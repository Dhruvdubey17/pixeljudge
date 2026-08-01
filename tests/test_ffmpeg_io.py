"""Tests for the ffmpeg wrapper's pure logic and its failure messages.

No ffmpeg is required: ffprobe output is fed in as text, and the binary-discovery
paths are exercised through the environment variables that override them. What is
being tested is that a broken environment produces an *actionable* message, since
that is the first thing anyone cloning this project will hit.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from pixeljudge.errors import FfmpegError, MissingDependencyError
from pixeljudge.io import ffmpeg as ff


def test_missing_binary_message_says_how_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXELJUDGE_FFMPEG", raising=False)
    monkeypatch.setattr("pixeljudge.io.ffmpeg.shutil.which", lambda _name: None)
    with pytest.raises(MissingDependencyError) as excinfo:
        ff.ffmpeg_binary()
    message = str(excinfo.value)
    assert "brew install ffmpeg" in message
    assert "libvmaf" in message


def test_override_env_var_must_point_at_an_executable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    not_executable = tmp_path / "ffmpeg"
    not_executable.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("PIXELJUDGE_FFMPEG", str(not_executable))
    with pytest.raises(MissingDependencyError, match="not an executable file"):
        ff.ffmpeg_binary()


def test_ffmpeg_available_is_false_without_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PIXELJUDGE_FFMPEG", raising=False)
    monkeypatch.delenv("PIXELJUDGE_FFPROBE", raising=False)
    monkeypatch.setattr("pixeljudge.io.ffmpeg.shutil.which", lambda _name: None)
    assert ff.ffmpeg_available() is False


def test_run_reports_the_command_and_the_stderr_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    class Failed:
        returncode = 1
        stdout = ""
        stderr = "\n".join(f"line {i}" for i in range(40)) + "\nActual error: no such filter"

    monkeypatch.setattr("pixeljudge.io.ffmpeg.subprocess.run", lambda *a, **k: Failed())
    with pytest.raises(FfmpegError) as excinfo:
        ff.run(["/usr/bin/ffmpeg", "-i", "in.mp4"])
    message = str(excinfo.value)
    # The useful part of an ffmpeg failure is at the end of stderr.
    assert "Actual error: no such filter" in message
    assert "line 0" not in message  # the head is noise
    assert "-i in.mp4" in message


def test_run_turns_a_missing_binary_into_a_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise FileNotFoundError

    monkeypatch.setattr("pixeljudge.io.ffmpeg.subprocess.run", boom)
    with pytest.raises(MissingDependencyError, match="brew install"):
        ff.run(["/nope/ffmpeg"])


def test_run_turns_a_timeout_into_a_domain_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd="ffmpeg", timeout=5)

    monkeypatch.setattr("pixeljudge.io.ffmpeg.subprocess.run", boom)
    with pytest.raises(FfmpegError, match="timed out after 5"):
        ff.run(["/usr/bin/ffmpeg"], timeout=5)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("30000/1001", pytest.approx(29.97, abs=0.01)),
        ("25/1", 25.0),
        ("0/0", None),  # ffprobe's way of saying "unknown"
        ("", None),
        (None, None),
        ("24", 24.0),
    ],
)
def test_frame_rate_parsing(raw: object, expected: object) -> None:
    assert ff._parse_rational(raw) == expected


@pytest.mark.parametrize(
    ("raw", "duration", "fps", "expected"),
    [
        ("150", 6.0, 25.0, 150),
        ("N/A", 6.0, 25.0, 150),  # fall back to duration x fps
        (None, 6.0, 25.0, 150),
        ("0", 6.0, 25.0, 150),
        ("N/A", None, 25.0, None),  # nothing to fall back to
    ],
)
def test_frame_count_parsing(
    raw: object, duration: float | None, fps: float | None, expected: int | None
) -> None:
    assert ff._parse_frame_count(raw, duration, fps) == expected


def _probe_payload(**overrides: Any) -> str:
    stream = {
        "width": 1280,
        "height": 720,
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "avg_frame_rate": "25/1",
        "duration": "6.0",
        "nb_frames": "150",
    }
    stream.update(overrides)
    return json.dumps({"streams": [stream], "format": {"duration": "6.0", "size": "1500000"}})


def _fake_probe(monkeypatch: pytest.MonkeyPatch, payload: str) -> None:
    monkeypatch.setattr(ff, "ffprobe_binary", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        ff,
        "run",
        lambda *_a, **_k: type("P", (), {"stdout": payload, "stderr": "", "returncode": 0})(),
    )


def test_probe_reads_stream_properties(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    _fake_probe(monkeypatch, _probe_payload())
    info = ff.probe(clip)
    assert info.resolution == "1280x720"
    assert info.fps == 25.0
    assert info.n_frames == 150
    # Bitrate is derived from size and duration, not trusted from the header.
    assert info.bitrate_kbps == pytest.approx(1500000 * 8 / 6 / 1000)


def test_probe_of_a_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FfmpegError, match="cannot probe missing file"):
        ff.probe(tmp_path / "absent.mp4")


def test_probe_without_a_video_stream_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "audio_only.mp4"
    clip.write_bytes(b"\x00")
    _fake_probe(monkeypatch, json.dumps({"streams": [], "format": {}}))
    with pytest.raises(FfmpegError, match="no video stream"):
        ff.probe(clip)


def test_probe_with_unparseable_output_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    _fake_probe(monkeypatch, "not json at all")
    with pytest.raises(FfmpegError, match="unparseable JSON"):
        ff.probe(clip)


def test_probe_without_dimensions_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    _fake_probe(monkeypatch, json.dumps({"streams": [{"codec_name": "h264"}], "format": {}}))
    with pytest.raises(FfmpegError, match="no usable dimensions"):
        ff.probe(clip)


def test_zero_duration_gives_zero_bitrate_instead_of_dividing_by_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00")
    _fake_probe(monkeypatch, json.dumps({"streams": [{"width": 16, "height": 16}], "format": {}}))
    assert ff.probe(clip).bitrate_kbps == 0.0


def test_available_encoders_parses_the_listing(monkeypatch: pytest.MonkeyPatch) -> None:
    listing = (
        "Encoders:\n"
        " V..... libsvtav1            SVT-AV1 encoder\n"
        " V....D libx264              libx264 H.264\n"
        " A....D aac                  AAC\n"
        "not an encoder line\n"
    )
    monkeypatch.setattr(ff, "ffmpeg_binary", lambda: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ff,
        "run",
        lambda *_a, **_k: type("P", (), {"stdout": listing, "stderr": "", "returncode": 0})(),
    )
    ff.available_encoders.cache_clear()
    encoders = ff.available_encoders()
    ff.available_encoders.cache_clear()
    assert {"libsvtav1", "libx264", "aac"} <= encoders


def test_require_libvmaf_explains_the_consequence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ff, "has_libvmaf", lambda: False)
    with pytest.raises(MissingDependencyError, match="no libvmaf filter"):
        ff.require_libvmaf()


def test_environment_report_marks_a_missing_ffmpeg(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom() -> str:
        raise MissingDependencyError("ffmpeg was not found on PATH")

    monkeypatch.setattr(ff, "ffmpeg_binary", boom)
    report = ff.environment_report()
    assert report["ffmpeg"].startswith("MISSING")
