"""Config validation tests.

The point of these is not that pydantic works, it is that our *rules* work: a
rung must name exactly one rate target, the SSIM feature names must be the
float_ variants, and a bad file must produce an error that names the file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pixeljudge.config import (
    LadderConfig,
    PipelineConfig,
    Rung,
    VmafConfig,
    load_ladder,
    load_pipeline_config,
)
from pixeljudge.errors import ConfigError

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_rung_requires_exactly_one_rate_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Rung(height=720, bitrate_kbps=3000, crf=23)
    with pytest.raises(ValueError, match="exactly one"):
        Rung(height=720)


def test_rung_name_describes_the_rate_mode() -> None:
    assert Rung(height=720, bitrate_kbps=3000).name == "720p_3000k"
    assert Rung(height=720, crf=26).name == "720p_crf26"
    assert Rung(height=720, crf=26, label="custom").name == "custom"


def test_vmaf_config_rejects_plain_ssim_names() -> None:
    # 'ssim' looks right but is rejected by several libvmaf builds, so we refuse
    # it at config time rather than 40 minutes into an encode.
    with pytest.raises(ValueError, match="float_ssim"):
        VmafConfig(features=["psnr", "ssim"])


def test_vmaf_config_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown vmaf model"):
        VmafConfig(vmaf_model="vmaf_v9.9.9")


def test_vmaf_config_defaults() -> None:
    cfg = VmafConfig()
    assert cfg.pool == "harmonic_mean"
    assert cfg.features == ["psnr", "float_ssim", "float_ms_ssim"]
    assert cfg.n_subsample == 1


def test_ladder_needs_rungs() -> None:
    with pytest.raises(ValueError, match="at least one rung"):
        LadderConfig(name="empty", codec="h264", rungs=[])


def test_ladder_sorted_by_height() -> None:
    ladder = LadderConfig(
        name="x",
        codec="h264",
        rungs=[Rung(height=1080, crf=20), Rung(height=360, crf=20)],
    )
    assert [r.height for r in ladder.sorted_by_height()] == [360, 1080]


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_pipeline_config(tmp_path / "nope.yaml")


def test_malformed_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("vmaf: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid YAML"):
        load_pipeline_config(bad)


def test_non_mapping_yaml_raises_config_error(tmp_path: Path) -> None:
    bad = tmp_path / "list.yaml"
    bad.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_pipeline_config(bad)


def test_validation_error_names_the_file_and_field(tmp_path: Path) -> None:
    bad = tmp_path / "ladder.yaml"
    bad.write_text("codec: h264\nrungs:\n  - {height: 720}\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_ladder(bad)
    message = str(excinfo.value)
    assert "ladder.yaml" in message
    assert "rungs" in message


def test_ladder_name_defaults_to_filename(tmp_path: Path) -> None:
    path = tmp_path / "my_ladder.yaml"
    path.write_text("codec: h264\nrungs:\n  - {height: 720, crf: 24}\n", encoding="utf-8")
    assert load_ladder(path).name == "my_ladder"


def test_repo_pipeline_config_loads_and_its_ladders_resolve() -> None:
    # Guards against a typo in the shipped configs, which is the most likely
    # real-world config failure.
    cfg = load_pipeline_config(REPO_ROOT / "configs" / "pipeline.yaml")
    cfg.paths.ladder_dir = REPO_ROOT / "configs" / "ladders"
    ladders = cfg.load_ladders()
    assert [ladder.name for ladder in ladders] == cfg.ladders
    assert all(ladder.rungs for ladder in ladders)


@pytest.mark.parametrize(
    "ladder_file",
    sorted((REPO_ROOT / "configs" / "ladders").glob("*.yaml")),
    ids=lambda p: p.stem,
)
def test_every_shipped_ladder_is_valid(ladder_file: Path) -> None:
    ladder = load_ladder(ladder_file)
    assert ladder.name == ladder_file.stem
    assert len(ladder.rungs) >= 1


def test_paths_ensure_creates_directories(tmp_path: Path) -> None:
    cfg = PipelineConfig()
    cfg.paths.raw = tmp_path / "raw"
    cfg.paths.encoded = tmp_path / "encoded"
    cfg.paths.metrics = tmp_path / "metrics"
    cfg.paths.datasets = tmp_path / "datasets"
    cfg.paths.reports = tmp_path / "reports"
    cfg.paths.models = tmp_path / "models"
    cfg.paths.ensure()
    assert cfg.paths.encoded.is_dir()
    assert cfg.paths.reports.is_dir()


def test_resolve_source_respects_absolute_paths(tmp_path: Path) -> None:
    cfg = PipelineConfig()
    cfg.paths.raw = tmp_path / "raw"
    assert cfg.resolve_source("clip.mp4") == tmp_path / "raw" / "clip.mp4"
    absolute = tmp_path / "elsewhere" / "clip.mp4"
    assert cfg.resolve_source(str(absolute)) == absolute
