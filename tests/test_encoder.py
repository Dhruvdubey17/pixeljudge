"""Encoder orchestration tests, with ffmpeg mocked at the seam.

The behaviours worth pinning here all cost real money if they break: a rung taller
than the master would produce a meaningless RD point, re-encoding an existing file
would throw away hours of CPU, and a manifest that does not round-trip would break
the measure stage that reads it.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pixeljudge.config import EncodeConfig, LadderConfig, Rung
from pixeljudge.encode import encoder as enc
from pixeljudge.encode.codecs import get_codec
from pixeljudge.errors import FfmpegError
from pixeljudge.io.ffmpeg import VideoInfo

LADDER = LadderConfig(
    name="test_ladder",
    codec="h264",
    rungs=[
        Rung(height=360, crf=24),
        Rung(height=720, crf=24),
        Rung(height=1080, crf=24),  # taller than the master below
    ],
)


def make_info(path: Path, height: int = 720, size: int = 500_000) -> VideoInfo:
    return VideoInfo(
        path=path,
        width=int(height * 16 / 9),
        height=height,
        fps=25.0,
        codec="h264",
        pix_fmt="yuv420p",
        duration_s=6.0,
        n_frames=150,
        size_bytes=size,
    )


@pytest.fixture()
def fake_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record ffmpeg calls, create the output file, and answer probes."""
    calls: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> None:
        calls.append(args)
        Path(args[-1]).write_bytes(b"\x00" * 500_000)

    def fake_probe(path: Path | str) -> VideoInfo:
        path = Path(path)
        # A source keeps its 720p; an output takes the height from its filename.
        height = 720
        for candidate in ("360p", "720p", "1080p", "144p"):
            if candidate in path.name:
                height = int(candidate.rstrip("p"))
        return make_info(path, height)

    monkeypatch.setattr(enc, "run_ffmpeg", fake_run)
    monkeypatch.setattr(enc, "probe", fake_probe)
    monkeypatch.setattr(enc, "available_encoders", lambda: frozenset({"libx264", "libx265"}))
    return calls


def test_output_path_carries_source_ladder_and_rung(tmp_path: Path) -> None:
    path = enc.output_path(
        Path("data/raw/big_buck_bunny.mp4"),
        LADDER,
        Rung(height=720, crf=24),
        get_codec("h264"),
        tmp_path,
    )
    assert path.name == "big_buck_bunny__test_ladder__720p_crf24.mp4"


def test_vp9_outputs_land_in_webm(tmp_path: Path) -> None:
    ladder = LadderConfig(name="vp9", codec="vp9", rungs=[Rung(height=720, crf=32)])
    path = enc.output_path(Path("clip.mp4"), ladder, ladder.rungs[0], get_codec("vp9"), tmp_path)
    assert path.suffix == ".webm"


def test_rungs_taller_than_the_master_are_skipped(
    fake_ffmpeg: list[list[str]], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    with caplog.at_level("WARNING"):
        rungs = enc.encode_ladder(source, LADDER, EncodeConfig(), tmp_path / "out")
    # Upscaling invents no detail, so a 1080p rung off a 720p master is meaningless.
    assert len(rungs) == 2
    assert "taller than the 720p master" in caplog.text


def test_encoding_is_ordered_smallest_first(fake_ffmpeg: list[list[str]], tmp_path: Path) -> None:
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    rungs = enc.encode_ladder(source, LADDER, EncodeConfig(), tmp_path / "out")
    assert [rung.height for rung in rungs] == [360, 720]


def test_existing_outputs_are_not_re_encoded(fake_ffmpeg: list[list[str]], tmp_path: Path) -> None:
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    out = tmp_path / "out"
    enc.encode_ladder(source, LADDER, EncodeConfig(), out)
    first_calls = len(fake_ffmpeg)
    second = enc.encode_ladder(source, LADDER, EncodeConfig(), out)
    assert len(fake_ffmpeg) == first_calls  # nothing re-ran
    assert all(rung.encode_seconds == 0.0 for rung in second)


def test_overwrite_forces_a_re_encode(fake_ffmpeg: list[list[str]], tmp_path: Path) -> None:
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    out = tmp_path / "out"
    enc.encode_ladder(source, LADDER, EncodeConfig(), out)
    before = len(fake_ffmpeg)
    enc.encode_ladder(source, LADDER, EncodeConfig(), out, overwrite=True)
    assert len(fake_ffmpeg) > before


def test_missing_encoder_is_reported_with_the_doctor_hint(
    fake_ffmpeg: list[list[str]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(enc, "available_encoders", lambda: frozenset({"libx264"}))
    ladder = LadderConfig(name="av1", codec="av1", rungs=[Rung(height=360, crf=32)])
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    with pytest.raises(FfmpegError, match="pixeljudge doctor"):
        enc.encode_ladder(source, ladder, EncodeConfig(), tmp_path / "out")


def test_measured_bitrate_comes_from_the_file_not_the_target(
    fake_ffmpeg: list[list[str]], tmp_path: Path
) -> None:
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    ladder = LadderConfig(name="rate", codec="h264", rungs=[Rung(height=360, bitrate_kbps=3000)])
    rung = enc.encode_ladder(source, ladder, EncodeConfig(), tmp_path / "out")[0]
    assert rung.target_bitrate_kbps == 3000
    # 500 kB over 6 s is about 667 kbps: the encoder missed, and we record reality.
    assert rung.actual_bitrate_kbps == pytest.approx(500_000 * 8 / 6 / 1000, rel=1e-3)


def test_manifest_round_trips(fake_ffmpeg: list[list[str]], tmp_path: Path) -> None:
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    out = tmp_path / "out"
    rungs = enc.encode_ladder(source, LADDER, EncodeConfig(), out)
    path = enc.write_manifest(rungs, enc.manifest_path(source, LADDER, out))
    assert enc.read_manifest(path) == rungs


def test_manifest_survives_a_none_valued_field(tmp_path: Path) -> None:
    # CRF-mode rungs have no target bitrate and vice versa; JSON must keep the nulls.
    record = enc.EncodedRung(
        source="s.mp4",
        ladder="l",
        codec="h264",
        rung="720p_crf24",
        path="o.mp4",
        width=1280,
        height=720,
        target_bitrate_kbps=None,
        crf=24,
        actual_bitrate_kbps=1234.5,
        size_bytes=1000,
        n_frames=None,
        encode_seconds=1.0,
    )
    path = enc.write_manifest([record], tmp_path / "m.json")
    reloaded = enc.read_manifest(path)[0]
    assert reloaded.target_bitrate_kbps is None
    assert reloaded.n_frames is None
    assert reloaded == replace(record)


def test_encode_command_includes_the_expected_flags(
    fake_ffmpeg: list[list[str]], tmp_path: Path
) -> None:
    source = tmp_path / "master.mp4"
    source.write_bytes(b"\x00")
    enc.encode_ladder(source, LADDER, EncodeConfig(), tmp_path / "out")
    command = " ".join(fake_ffmpeg[0])
    assert "-c:v libx264" in command
    assert "scale=-2:360:flags=bicubic" in command
    assert "-an" in command
    assert "-g 50" in command  # 2 s at 25 fps
