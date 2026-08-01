"""CLI tests using Typer's runner. No ffmpeg, no encoding.

What is worth testing at this layer is the *guidance*: running a stage out of order,
or with nothing configured, should say what to do next rather than raise a
traceback or quietly do nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pixeljudge.cli import app
from pixeljudge.config import PipelineConfig
from pixeljudge.errors import PixelJudgeError

runner = CliRunner()

MINIMAL_CONFIG = """
paths:
  raw: {root}/raw
  encoded: {root}/encoded
  metrics: {root}/metrics
  datasets: {root}/datasets
  reports: {root}/reports
  models: {root}/models
  ladder_dir: {ladders}
ladders: [fixture_smoke_h264]
sources: []
"""


@pytest.fixture()
def config_file(tmp_path: Path) -> Path:
    ladders = Path(__file__).resolve().parents[1] / "configs" / "ladders"
    path = tmp_path / "pipeline.yaml"
    path.write_text(MINIMAL_CONFIG.format(root=tmp_path, ladders=ladders), encoding="utf-8")
    return path


def test_help_lists_every_stage() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in ("doctor", "encode", "measure", "scan", "ladder", "train", "report"):
        assert command in result.stdout


def test_no_arguments_shows_help_rather_than_doing_something() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.stdout


def test_bad_config_path_is_reported_clearly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["encode", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code != 0
    assert isinstance(result.exception, PixelJudgeError)
    assert "not found" in str(result.exception)


def test_encode_without_sources_explains_what_to_do(config_file: Path) -> None:
    result = runner.invoke(app, ["encode", "--config", str(config_file)])
    assert result.exit_code != 0
    assert "no sources given" in str(result.exception)


def test_encode_with_a_missing_source_names_the_file(config_file: Path) -> None:
    result = runner.invoke(app, ["encode", "--config", str(config_file), "-s", "ghost.mp4"])
    assert "ghost.mp4" in str(result.exception)


def test_measure_before_encode_says_to_encode_first(config_file: Path) -> None:
    result = runner.invoke(app, ["measure", "--config", str(config_file)])
    assert "run 'pixeljudge encode' first" in str(result.exception)


def test_report_with_nothing_cached_says_what_to_run(config_file: Path) -> None:
    result = runner.invoke(app, ["report", "--config", str(config_file)])
    assert "nothing to report yet" in str(result.exception)


def test_ladder_requires_a_source(config_file: Path) -> None:
    result = runner.invoke(app, ["ladder", "--config", str(config_file)])
    assert "--source is required" in str(result.exception)


def test_report_writes_tables_from_cached_metrics(config_file: Path, tmp_path: Path) -> None:
    """The whole point of ``report``: regenerate from cache, encode nothing."""
    import pandas as pd

    cfg = PipelineConfig()
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "source": ["clip"] * 5,
            "distorted": [f"clip__crf{c}.mp4" for c in (20, 26, 32, 38, 44)],
            "ladder": ["crf_sweep_h264"] * 5,
            "codec": ["h264"] * 5,
            "rung": [f"720p_crf{c}" for c in (20, 26, 32, 38, 44)],
            "crf": [20, 26, 32, 38, 44],
            "actual_bitrate_kbps": [8000.0, 4000.0, 2000.0, 1000.0, 500.0],
            "vmaf": [97.0, 93.0, 86.0, 74.0, 58.0],
            "psnr_y": [46.0, 43.0, 39.0, 35.0, 31.0],
        }
    ).to_csv(metrics_dir / "all_metrics.csv", index=False)

    result = runner.invoke(app, ["report", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    reports = tmp_path / "reports"
    assert (reports / "ladder_summary.csv").exists()
    assert (reports / "rd_curves_codecs_clip.png").exists()
    # A single codec cannot produce a BD-Rate: there is nothing to compare against.
    assert not (reports / "bd_rate.csv").exists()
    assert cfg.paths.reports == Path("reports")  # defaults untouched by the run


def test_report_computes_bd_rate_when_two_codecs_are_present(
    config_file: Path, tmp_path: Path
) -> None:
    import pandas as pd

    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for codec, factor in (("h264", 1.0), ("hevc", 0.5)):
        for crf, rate, vmaf in zip(
            (20, 26, 32, 38, 44),
            (8000.0, 4000.0, 2000.0, 1000.0, 500.0),
            (97.0, 93.0, 86.0, 74.0, 58.0),
            strict=True,
        ):
            rows.append(
                {
                    "source": "clip",
                    "distorted": f"clip__{codec}__crf{crf}.mp4",
                    "ladder": f"crf_sweep_{codec}",
                    "codec": codec,
                    "rung": f"720p_crf{crf}",
                    "crf": crf,
                    "actual_bitrate_kbps": rate * factor,
                    "vmaf": vmaf,
                    "psnr_y": 30.0 + vmaf / 6,
                }
            )
    pd.DataFrame(rows).to_csv(metrics_dir / "all_metrics.csv", index=False)

    result = runner.invoke(app, ["report", "--config", str(config_file)])
    assert result.exit_code == 0, result.output
    bd = pd.read_csv(tmp_path / "reports" / "bd_rate.csv")
    hevc = bd[(bd["codec"] == "hevc") & (bd["source"] == "clip")]
    # Half the bitrate at identical quality is exactly -50%.
    assert hevc["bd_rate_pct"].iloc[0] == pytest.approx(-50.0, abs=0.5)
