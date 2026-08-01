"""End-to-end tests that need a real ffmpeg with libvmaf.

Marked ``integration`` and skipped automatically when the binary is absent, so CI
(which deliberately has no ffmpeg) stays green while still proving that the offline
suite is genuinely offline.

These assert the properties that unit tests cannot: that a CRF ladder really does
produce monotonically better quality at higher bitrate, and that a measurement of a
clip against itself really does come out perfect. Both would still "pass" as code
while being completely wrong about video.

Run them with:  uv run pytest -m integration
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pixeljudge.config import EncodeConfig, VmafConfig, load_ladder
from pixeljudge.io.ffmpeg import ffmpeg_available, has_libvmaf, probe

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ffmpeg_available(), reason="needs the ffmpeg/ffprobe binaries"),
]


@pytest.fixture(scope="module")
def gradient() -> Path:
    clip = FIXTURES / "gradient.mp4"
    if not clip.exists():
        pytest.skip("run scripts/make_fixtures.py to generate the synthetic clips")
    return clip


def test_probe_reads_a_real_fixture(gradient: Path) -> None:
    info = probe(gradient)
    assert info.resolution == "320x180"
    assert info.n_frames == 30
    assert info.fps == pytest.approx(15.0)
    assert info.bitrate_kbps > 0


def test_encoding_produces_the_requested_resolutions(gradient: Path, tmp_path: Path) -> None:
    from pixeljudge.encode.encoder import encode_ladder

    ladder = load_ladder(REPO_ROOT / "configs" / "ladders" / "fixture_smoke_h264.yaml")
    rungs = encode_ladder(gradient, ladder, EncodeConfig(preset="veryfast"), tmp_path)
    assert len(rungs) == 3
    by_height = {rung.height: rung for rung in rungs}
    assert set(by_height) == {180, 144}
    # -2 width keeps 16:9 and rounds to an even number.
    assert by_height[144].width == 256
    assert all(Path(rung.path).exists() for rung in rungs)


def test_encoding_is_idempotent(gradient: Path, tmp_path: Path) -> None:
    from pixeljudge.encode.encoder import encode_ladder

    ladder = load_ladder(REPO_ROOT / "configs" / "ladders" / "fixture_smoke_h264.yaml")
    cfg = EncodeConfig(preset="veryfast")
    first = encode_ladder(gradient, ladder, cfg, tmp_path)
    second = encode_ladder(gradient, ladder, cfg, tmp_path)
    # The second pass must skip, not re-encode: a real sweep is hours of CPU.
    assert [r.size_bytes for r in first] == [r.size_bytes for r in second]
    assert all(rung.encode_seconds == 0.0 for rung in second)


@pytest.mark.skipif(not has_libvmaf(), reason="needs an ffmpeg built with libvmaf")
def test_measuring_a_clip_against_itself_is_perfect_on_fidelity(gradient: Path) -> None:
    """Identical input must saturate every fidelity metric.

    Note what is *not* asserted: VMAF = 100. It comes out at about 97.5 here, and
    that is correct behaviour rather than a bug — see the next test.
    """
    from pixeljudge.metrics.vqm import PSNR, SSIM, VMAF, measure_pair

    result = measure_pair(gradient, gradient)
    assert result.pooled[SSIM] == pytest.approx(1.0, abs=1e-6)
    assert result.pooled["float_ms_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert result.pooled["adm2"] == pytest.approx(1.0, abs=1e-3)
    # libvmaf caps PSNR at 60 dB rather than reporting infinity.
    assert result.pooled[PSNR] == pytest.approx(60.0, abs=0.01)
    assert result.pooled[VMAF] > 97.0
    assert result.n_frames == 30


@pytest.mark.skipif(not has_libvmaf(), reason="needs an ffmpeg built with libvmaf")
def test_vmaf_cannot_reach_100_without_motion(gradient: Path) -> None:
    """VMAF's motion feature depresses the score of static content.

    The motion feature is the mean absolute difference from the *previous* frame,
    so it is zero on frame 0 by definition, and near zero throughout a static clip.
    With every fidelity feature saturated at 1 and motion at 0, VMAF's SVM
    extrapolates to about 97.4 rather than 100: zero motion is nearly absent from
    the natural content it was trained on.

    Consequence worth knowing: on low-motion material VMAF has a ceiling below 100,
    so a "target VMAF 94" top rung is closer to that ceiling than it appears. The
    first frame of *any* measurement is affected for the same reason.
    """
    from pixeljudge.metrics.vqm import VMAF, measure_pair

    static_clip = measure_pair(gradient, gradient)
    moving_clip = measure_pair(FIXTURES / "motion.mp4", FIXTURES / "motion.mp4")

    # Frame 0 has no previous frame, so it is depressed in both clips.
    assert static_clip.per_frame[VMAF].iloc[0] < 98.0
    assert moving_clip.per_frame[VMAF].iloc[0] < 98.0
    # From frame 1 on, the clip with real motion reaches 100 and the static one
    # never does.
    assert moving_clip.per_frame[VMAF].iloc[1:].max() == pytest.approx(100.0, abs=1e-6)
    assert static_clip.per_frame[VMAF].max() < 98.0


@pytest.mark.skipif(not has_libvmaf(), reason="needs an ffmpeg built with libvmaf")
def test_quality_rises_with_bitrate(gradient: Path, tmp_path: Path) -> None:
    from pixeljudge.encode.encoder import encode_ladder
    from pixeljudge.metrics.vqm import VMAF, measure_pair

    ladder = load_ladder(REPO_ROOT / "configs" / "ladders" / "fixture_smoke_h264.yaml")
    rungs = encode_ladder(gradient, ladder, EncodeConfig(preset="veryfast"), tmp_path)
    measured = sorted(
        (rung.actual_bitrate_kbps, measure_pair(Path(rung.path), gradient).pooled[VMAF])
        for rung in rungs
    )
    qualities = [vmaf for _, vmaf in measured]
    # The single most important sanity check in the whole project: more bits must
    # buy more quality. If this fails, something is misaligned or mismeasured.
    assert qualities == sorted(qualities), measured
    assert qualities[-1] > qualities[0]


@pytest.mark.skipif(not has_libvmaf(), reason="needs an ffmpeg built with libvmaf")
def test_measurement_records_its_provenance(gradient: Path) -> None:
    from pixeljudge.metrics.vqm import measure_pair

    context = measure_pair(gradient, gradient, VmafConfig(pool="mean")).context
    # A VMAF number without its model and pooling method is not reproducible.
    assert context["vmaf_model"] == "vmaf_v0.6.1"
    assert context["pool"] == "mean"
    assert context["ffmpeg_version"]
    assert context["libvmaf_version"] != "unknown"


@pytest.mark.skipif(not has_libvmaf(), reason="needs an ffmpeg built with libvmaf")
def test_a_low_bitrate_encode_of_a_gradient_bands(gradient: Path, tmp_path: Path) -> None:
    """The project's central claim, tested on a real encode.

    A hard-compressed gradient should show measurable banding while the pooled
    metrics stay respectable. This asserts the *direction*: banding increases as
    quality drops, and it is detectable on real encoded pixels rather than only on
    synthetic images.
    """
    from pixeljudge.artifacts.scan import scan_clip
    from pixeljudge.config import LadderConfig, Rung
    from pixeljudge.encode.encoder import encode_ladder

    ladder = LadderConfig(
        name="banding_probe",
        codec="h264",
        rungs=[Rung(height=180, crf=20, label="high"), Rung(height=180, crf=45, label="low")],
    )
    rungs = {r.rung: r for r in encode_ladder(gradient, ladder, EncodeConfig(), tmp_path)}
    high = scan_clip(rungs["high"].path, n_sample=6)
    low = scan_clip(rungs["low"].path, n_sample=6)
    assert low.banding_max >= high.banding_max


def test_frames_can_be_read_from_a_webm(gradient: Path, tmp_path: Path) -> None:
    """VP9-in-WebM is the case OpenCV builds most often cannot decode.

    The scanner falls back to ffmpeg for exactly this, and without the fallback a
    whole codec would silently vanish from the artifact table.
    """
    from pixeljudge.artifacts.scan import read_frames, sample_frame_indices
    from pixeljudge.config import LadderConfig, Rung
    from pixeljudge.encode.encoder import encode_rung
    from pixeljudge.io.ffmpeg import available_encoders

    if "libvpx-vp9" not in available_encoders():
        pytest.skip("this ffmpeg has no VP9 encoder")
    ladder = LadderConfig(name="vp9_probe", codec="vp9", rungs=[Rung(height=180, crf=40)])
    encoded = encode_rung(gradient, ladder.rungs[0], ladder, EncodeConfig(), tmp_path)
    frames = read_frames(Path(encoded.path), sample_frame_indices(encoded.n_frames or 30, 4))
    assert len(frames) == 4
    assert all(frame.ndim == 3 for _, frame in frames)
