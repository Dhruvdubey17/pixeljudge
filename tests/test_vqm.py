"""Metrics tests that need no ffmpeg.

The parser is the interesting part: it is the boundary where another tool's
output becomes our data, so it is tested against a *real* libvmaf log captured
from ffmpeg 8.1.2 / libvmaf 3.2.0 (``tests/fixtures/libvmaf_log.json``) as well as
against hand-built payloads for the cases a real log will not conveniently
produce.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pixeljudge.config import VmafConfig
from pixeljudge.errors import MetricsError
from pixeljudge.metrics.vqm import (
    MS_SSIM,
    PSNR,
    SSIM,
    VMAF,
    build_filtergraph,
    build_libvmaf_options,
    harmonic_mean,
    parse_vmaf_log,
    pool_all,
    pool_metric,
)

FIXTURE_LOG = Path(__file__).parent / "fixtures" / "libvmaf_log.json"


@pytest.fixture(scope="module")
def real_log() -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(FIXTURE_LOG.read_text(encoding="utf-8"))
    return payload


def test_parses_real_libvmaf_log(real_log: dict[str, Any]) -> None:
    frame_table = parse_vmaf_log(real_log)
    assert len(frame_table) == 30
    assert list(frame_table["frame"]) == list(range(30))
    for column in (VMAF, PSNR, SSIM, MS_SSIM):
        assert column in frame_table.columns
    # Sanity ranges rather than exact values: this clip is a mild crf 33 encode.
    assert frame_table[VMAF].between(0, 100).all()
    assert frame_table[PSNR].between(20, 60).all()
    assert frame_table[SSIM].between(0.9, 1.0).all()


def test_real_log_keeps_vmaf_elementary_features(real_log: dict[str, Any]) -> None:
    # These are the features VMAF's own model fuses; free in the log and useful
    # later as an ablation baseline.
    frame_table = parse_vmaf_log(real_log)
    for column in ("adm2", "vif_scale0", "vif_scale3", "motion2"):
        assert column in frame_table.columns


def test_headline_metrics_come_first(real_log: dict[str, Any]) -> None:
    columns = list(parse_vmaf_log(real_log).columns)
    assert columns[:5] == ["frame", VMAF, PSNR, SSIM, MS_SSIM]


def test_psnr_alias_is_normalised() -> None:
    # Older libvmaf builds write "psnr" where newer ones write "psnr_y".
    payload = {"frames": [{"frameNum": 0, "metrics": {"psnr": 41.5, "vmaf": 88.0}}]}
    frame_table = parse_vmaf_log(payload)
    assert frame_table[PSNR].iloc[0] == pytest.approx(41.5)


def test_ssim_aliases_are_normalised() -> None:
    payload = {"frames": [{"frameNum": 0, "metrics": {"ssim": 0.98, "ms_ssim": 0.99}}]}
    frame_table = parse_vmaf_log(payload)
    assert frame_table[SSIM].iloc[0] == pytest.approx(0.98)
    assert frame_table[MS_SSIM].iloc[0] == pytest.approx(0.99)


def test_frame_number_defaults_to_position() -> None:
    payload = {"frames": [{"metrics": {"vmaf": 90.0}}, {"metrics": {"vmaf": 91.0}}]}
    assert list(parse_vmaf_log(payload)["frame"]) == [0, 1]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"frames": []},
        {"frames": "not-a-list"},
    ],
)
def test_missing_frames_raises(payload: dict[str, Any]) -> None:
    with pytest.raises(MetricsError, match="no frames"):
        parse_vmaf_log(payload)


def test_frame_without_metrics_raises() -> None:
    with pytest.raises(MetricsError, match="no metrics"):
        parse_vmaf_log({"frames": [{"frameNum": 0}]})


def test_frame_that_is_not_an_object_raises() -> None:
    with pytest.raises(MetricsError, match="not an object"):
        parse_vmaf_log({"frames": ["nope"]})


def test_unrecognised_metrics_only_raises() -> None:
    with pytest.raises(MetricsError, match="no recognised metrics"):
        parse_vmaf_log({"frames": [{"frameNum": 0, "metrics": {"psnr_cb": 40.0}}]})


def test_non_numeric_metric_raises() -> None:
    with pytest.raises(MetricsError, match="non-numeric"):
        parse_vmaf_log({"frames": [{"frameNum": 0, "metrics": {"vmaf": "nan-ish"}}]})


def test_harmonic_mean_matches_definition() -> None:
    # libvmaf's harmonic mean uses a +1 offset so a legitimate score of zero does
    # not divide by zero.
    values = [0.0, 100.0]
    expected = 2 / (1 / 1.0 + 1 / 101.0) - 1
    assert harmonic_mean(values) == pytest.approx(expected)


def test_harmonic_mean_punishes_bad_frames_more_than_the_mean() -> None:
    good_run = [95.0] * 9 + [10.0]
    assert harmonic_mean(good_run) < sum(good_run) / len(good_run)


def test_harmonic_mean_of_constant_series_is_that_constant() -> None:
    assert harmonic_mean([42.0] * 5) == pytest.approx(42.0)


def test_harmonic_mean_ignores_non_finite_values() -> None:
    assert harmonic_mean([float("inf"), 3.0, 3.0]) == pytest.approx(3.0)
    assert math.isnan(harmonic_mean([float("nan")]))


def test_pool_metric_reports_every_method() -> None:
    pooled = pool_metric([90.0, 80.0, 70.0])
    assert pooled["mean"] == pytest.approx(80.0)
    assert pooled["min"] == pytest.approx(70.0)
    assert pooled["max"] == pytest.approx(90.0)
    assert pooled["harmonic_mean"] < pooled["mean"]
    assert pooled["std"] > 0


def test_pool_metric_all_nan_is_nan() -> None:
    pooled = pool_metric([float("nan"), float("nan")])
    assert all(math.isnan(value) for value in pooled.values())


def test_pool_all_covers_every_metric_column(real_log: dict[str, Any]) -> None:
    pooled = pool_all(parse_vmaf_log(real_log))
    assert "frame" not in pooled
    assert set(pooled) >= {VMAF, PSNR, SSIM, MS_SSIM}
    assert 0 <= pooled[VMAF]["harmonic_mean"] <= 100


def test_filtergraph_puts_distorted_first_and_scales_it() -> None:
    graph = build_filtergraph(VmafConfig(), 1920, 1080, Path("/tmp/out.json"), ref_fps=25.0)
    # Input 0 must be the distorted clip: libvmaf judges its first input.
    assert graph.startswith("[0:v]settb=AVTB,setpts=N/25.000000/TB,scale=1920:1080:flags=bicubic")
    assert "[dist][ref]libvmaf=" in graph


def test_filtergraph_locks_both_inputs_to_the_frame_index() -> None:
    # The bug this prevents: WebM stores timestamps in milliseconds and MP4 does not,
    # so PTS-based pairing slips by a frame and every metric collapses. Regenerating
    # timestamps from the frame index puts frame k of both inputs on the same grid.
    graph = build_filtergraph(VmafConfig(), 1280, 720, Path("/tmp/out.json"), ref_fps=24.0)
    assert graph.count("settb=AVTB,setpts=N/24.000000/TB") == 2
    assert "PTS-STARTPTS" not in graph


def test_filtergraph_falls_back_when_the_frame_rate_is_unknown(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING"):
        graph = build_filtergraph(VmafConfig(), 1280, 720, Path("/tmp/out.json"), ref_fps=0.0)
    assert graph.count("setpts=PTS-STARTPTS") == 2
    assert "frame rate unknown" in caplog.text


def test_fractional_frame_rates_keep_enough_precision() -> None:
    # 30000/1001 must not be rounded to 30, or a 150-frame clip drifts by 5 frames.
    graph = build_filtergraph(VmafConfig(), 640, 360, Path("/tmp/o.json"), ref_fps=29.97003)
    assert "N/29.970030/TB" in graph


def test_filtergraph_uses_float_feature_names() -> None:
    graph = build_filtergraph(VmafConfig(), 640, 360, Path("/tmp/out.json"), ref_fps=25.0)
    assert "name=float_ssim" in graph
    assert "name=float_ms_ssim" in graph
    assert "name=ssim|" not in graph


def test_libvmaf_options_carry_model_and_pooling() -> None:
    options = build_libvmaf_options(
        VmafConfig(vmaf_model="vmaf_4k_v0.6.1", pool="mean"), Path("/tmp/o.json")
    )
    assert "model='version=vmaf_4k_v0.6.1'" in options
    assert "pool=mean" in options


def test_libvmaf_options_add_cambi_when_enabled() -> None:
    options = build_libvmaf_options(VmafConfig(enable_cambi=True), Path("/tmp/o.json"))
    assert "name=cambi" in options


def test_libvmaf_options_omit_defaults() -> None:
    options = build_libvmaf_options(VmafConfig(), Path("/tmp/o.json"))
    assert "n_threads" not in options  # 0 means "let libvmaf decide"
    assert "n_subsample" not in options  # 1 means "every frame"


def test_libvmaf_options_include_tuning_when_set() -> None:
    options = build_libvmaf_options(VmafConfig(n_threads=4, n_subsample=5), Path("/tmp/o.json"))
    assert "n_threads=4" in options
    assert "n_subsample=5" in options


def test_measure_many_keeps_going_after_one_failure(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Losing an hour of measurement because clip 2 of 3 was truncated would be a bad
    # trade, so a failure is logged and the batch continues.
    from pixeljudge.metrics import vqm

    def fake_measure(distorted, reference, cfg, **_kwargs):  # type: ignore[no-untyped-def]
        if "broken" in str(distorted):
            raise MetricsError("libvmaf produced no log file")
        return vqm.QualityResult(
            reference=str(reference),
            distorted=str(distorted),
            per_frame=pd.DataFrame({"frame": [0, 1], VMAF: [90.0, 92.0]}),
            pooled={VMAF: 91.0},
            pooled_all={VMAF: {"min": 90.0, "mean": 91.0}},
            context={"vmaf_model": "vmaf_v0.6.1"},
        )

    monkeypatch.setattr(vqm, "measure_pair", fake_measure)
    jobs = [
        (Path("a.mp4"), Path("ref.mp4"), {"rung": "a"}),
        (Path("broken.mp4"), Path("ref.mp4"), {"rung": "broken"}),
        (Path("c.mp4"), Path("ref.mp4"), {"rung": "c"}),
    ]
    with caplog.at_level("ERROR"):
        table = vqm.measure_many(jobs)
    assert list(table["rung"]) == ["a", "c"]
    assert "measurement failed for broken.mp4" in caplog.text
    # Extra columns from the caller survive alongside the measured ones.
    assert "ctx_vmaf_model" in table.columns


def test_measure_many_raises_when_everything_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    from pixeljudge.metrics import vqm

    def always_fails(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise MetricsError("nope")

    monkeypatch.setattr(vqm, "measure_pair", always_fails)
    with pytest.raises(MetricsError, match="no measurement succeeded"):
        vqm.measure_many([(Path("a.mp4"), Path("ref.mp4"), {})])


def test_summary_row_flattens_pooled_and_provenance() -> None:
    from pixeljudge.metrics.vqm import QualityResult

    result = QualityResult(
        reference="/data/raw/clip.mp4",
        distorted="/data/encoded/clip__720p.mp4",
        per_frame=pd.DataFrame({"frame": [0, 1], VMAF: [80.0, 90.0]}),
        pooled={VMAF: 84.7},
        pooled_all={VMAF: {"min": 80.0, "mean": 85.0}},
        context={"vmaf_model": "vmaf_v0.6.1", "pool": "harmonic_mean"},
    )
    row = result.summary_row()
    assert row["reference"] == "clip.mp4"  # basenames, not absolute paths
    assert row["distorted"] == "clip__720p.mp4"
    assert row["n_frames"] == 2
    assert row[VMAF] == 84.7
    assert row["vmaf_min"] == 80.0  # worst frame is worth keeping
    assert row["ctx_pool"] == "harmonic_mean"


def test_result_saves_frames_and_summary(tmp_path: Path) -> None:
    from pixeljudge.metrics.vqm import QualityResult

    result = QualityResult(
        reference="ref.mp4",
        distorted="dis.mp4",
        per_frame=pd.DataFrame({"frame": [0], VMAF: [90.0]}),
        pooled={VMAF: 90.0},
    )
    frames_path, summary_path = result.save(tmp_path, stem="clip")
    assert frames_path.name == "clip.frames.csv"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["pooled"][VMAF] == 90.0


def test_windows_style_log_path_is_escaped() -> None:
    # A drive letter would otherwise end the option at the colon.
    options = build_libvmaf_options(VmafConfig(), Path("C:/tmp/out.json"))
    assert r"C\:/tmp/out.json" in options
