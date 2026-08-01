"""Configuration models and YAML loaders.

Why pydantic instead of plain dicts: a ladder file with ``bitrate: 3000`` where
we expected ``bitrate_kbps`` should fail immediately with a message naming the
file and the field, not twenty seconds later inside an ffmpeg command line.
Validation happens once, at load time, and everything downstream can trust the
objects it receives.

Note on naming: pydantic reserves the ``model_`` prefix for its own methods, so
the VMAF model field is called ``vmaf_model`` rather than ``model_name``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .errors import ConfigError

CodecName = Literal["h264", "hevc", "vp9", "av1"]
PoolMethod = Literal["mean", "harmonic_mean", "min"]

# The three VMAF models we support. A VMAF score is only meaningful together
# with the model that produced it, so the name is carried into every result.
KNOWN_VMAF_MODELS = ("vmaf_v0.6.1", "vmaf_4k_v0.6.1", "vmaf_v0.6.1neg")

DEFAULT_FEATURES = ("psnr", "float_ssim", "float_ms_ssim")

T = TypeVar("T", bound=BaseModel)


class VmafConfig(BaseModel):
    """Options for the single libvmaf pass that produces every metric."""

    vmaf_model: str = "vmaf_v0.6.1"
    pool: PoolMethod = "harmonic_mean"
    # 'float_ssim'/'float_ms_ssim' rather than 'ssim'/'ms_ssim': the plain names
    # are rejected by several libvmaf builds.
    features: list[str] = Field(default_factory=lambda: list(DEFAULT_FEATURES))
    n_threads: int = Field(default=0, ge=0)  # 0 lets libvmaf pick
    n_subsample: int = Field(default=1, ge=1)  # 1 = score every frame
    # CAMBI is Netflix's dedicated banding index. Off by default because it is
    # not part of the VMAF score and costs extra time; useful as a reference
    # point for our own banding proxy.
    enable_cambi: bool = False

    @field_validator("vmaf_model")
    @classmethod
    def _known_model(cls, value: str) -> str:
        if value not in KNOWN_VMAF_MODELS:
            raise ValueError(f"unknown vmaf model {value!r}; expected one of {KNOWN_VMAF_MODELS}")
        return value

    @field_validator("features")
    @classmethod
    def _non_empty_features(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("features must list at least one libvmaf feature")
        bad = [f for f in value if f in {"ssim", "ms_ssim"}]
        if bad:
            raise ValueError(
                f"use float_ssim/float_ms_ssim instead of {bad}: the plain names "
                "fail on several libvmaf builds"
            )
        return value


class EncodeConfig(BaseModel):
    """Encoder knobs that are not part of a ladder rung."""

    preset: str = "medium"  # x264/x265 speed/efficiency trade-off
    av1_preset: int = Field(default=8, ge=0, le=13)  # SVT-AV1: higher = faster
    vp9_cpu_used: int = Field(default=2, ge=0, le=8)
    gop_seconds: float = Field(default=2.0, gt=0)  # segment-friendly keyframes
    scale_flags: str = "bicubic"
    pix_fmt: str = "yuv420p"
    # Streaming rungs are rate-capped, not free-running VBR: these multipliers
    # mirror the -maxrate/-bufsize a packager would apply.
    maxrate_multiplier: float = Field(default=2.0, gt=0)
    bufsize_multiplier: float = Field(default=2.0, gt=0)


class Rung(BaseModel):
    """One entry of an encoding ladder: a resolution plus a rate target.

    Exactly one of ``bitrate_kbps`` (average-bitrate mode, what a streaming
    service ships) or ``crf`` (constant quality, better for codec comparison)
    must be set.
    """

    height: int = Field(gt=0)
    width: int | None = Field(default=None, gt=0)
    bitrate_kbps: int | None = Field(default=None, gt=0)
    crf: int | None = Field(default=None, ge=0, le=63)
    label: str | None = None

    @model_validator(mode="after")
    def _exactly_one_rate_target(self) -> Rung:
        if (self.bitrate_kbps is None) == (self.crf is None):
            raise ValueError("set exactly one of bitrate_kbps or crf on each rung")
        return self

    @property
    def name(self) -> str:
        """Short identifier used in filenames and plot legends."""
        if self.label:
            return self.label
        rate = f"{self.bitrate_kbps}k" if self.bitrate_kbps is not None else f"crf{self.crf}"
        return f"{self.height}p_{rate}"


class LadderConfig(BaseModel):
    """A named set of rungs for one codec, i.e. one platform's delivery recipe."""

    name: str
    codec: CodecName
    description: str = ""
    rungs: list[Rung]

    @field_validator("rungs")
    @classmethod
    def _needs_rungs(cls, value: list[Rung]) -> list[Rung]:
        if not value:
            raise ValueError("a ladder needs at least one rung")
        return value

    def sorted_by_height(self) -> list[Rung]:
        return sorted(self.rungs, key=lambda r: r.height)


class PathsConfig(BaseModel):
    """Where things live on disk. All runtime dirs are gitignored."""

    raw: Path = Path("data/raw")
    encoded: Path = Path("data/encoded")
    metrics: Path = Path("data/metrics")
    datasets: Path = Path("data/datasets")
    reports: Path = Path("reports")
    models: Path = Path("models")
    ladder_dir: Path = Path("configs/ladders")

    def ensure(self) -> None:
        """Create every output directory. Called before long-running work."""
        for path in (
            self.raw,
            self.encoded,
            self.metrics,
            self.datasets,
            self.reports,
            self.models,
        ):
            path.mkdir(parents=True, exist_ok=True)


class LiveNflxConfig(BaseModel):
    """The subjective-regression run on the LIVE-Netflix Video QoE Database.

    ``scope`` is the field that decides whether the resulting numbers mean what
    they appear to mean, so it is required in the file rather than defaulted
    silently: ``compression_only`` keeps the continuous-playback conditions, where
    compression metrics and the human score describe the same thing, and ``all``
    adds the rebuffering conditions, where they do not.
    """

    release_dir: Path = Path("data/datasets/LIVE_NFLX_PublicData_VideoATLAS_Release")
    split_matrix: Path = Path("data/datasets/TrainingMatrix_LIVENetflix_1000_trials.mat")
    scope: Literal["compression_only", "all"]
    # Quality models to pool into features. Defaults to the four PixelJudge can
    # also compute itself, which is what makes the Stage 3 comparison possible.
    metrics: list[str] = Field(default_factory=lambda: ["vmaf", "psnr", "ssim", "ms_ssim"])
    poolings: list[str] = Field(default_factory=lambda: ["mean", "hmean"])
    # 'group_kfold' is exhaustive and is the project's own protocol;
    # 'released_splits' replays the release's sampled 80/20 trials so the numbers
    # line up with the published ones.
    split_mode: Literal["group_kfold", "released_splits"] = "group_kfold"
    n_splits: int = Field(default=5, ge=2)
    n_trials: int = Field(default=100, ge=1)
    # Single-metric baselines, named by their pooled feature column.
    baselines: list[str] = Field(default_factory=lambda: ["vmaf_mean", "psnr_mean", "ssim_mean"])
    random_seed: int = 1234

    @field_validator("metrics", "poolings")
    @classmethod
    def _non_empty(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("must list at least one entry")
        return value


class PipelineConfig(BaseModel):
    """Top-level config: paths, measurement options, and which ladders to run."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    vmaf: VmafConfig = Field(default_factory=VmafConfig)
    encode: EncodeConfig = Field(default_factory=EncodeConfig)
    ladders: list[str] = Field(default_factory=list)
    # Reference clips (relative to paths.raw unless absolute).
    sources: list[str] = Field(default_factory=list)
    # Frames sampled per clip by the artifact scanner.
    artifact_sample_frames: int = Field(default=12, ge=1)
    random_seed: int = 1234

    def ladder_path(self, name: str) -> Path:
        return self.paths.ladder_dir / f"{name}.yaml"

    def load_ladders(self) -> list[LadderConfig]:
        """Load every ladder named in ``ladders``, in order."""
        return [load_ladder(self.ladder_path(name)) for name in self.ladders]

    def resolve_source(self, source: str) -> Path:
        path = Path(source)
        return path if path.is_absolute() else self.paths.raw / path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")
    return raw


def _build(model: type[T], data: dict[str, Any], source: Path | str) -> T:
    """Validate ``data`` into ``model``, reporting the file that was wrong."""
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ConfigError(f"invalid config in {source}: {details}") from exc


def load_pipeline_config(path: Path | str = Path("configs/pipeline.yaml")) -> PipelineConfig:
    """Load the top-level pipeline config."""
    path = Path(path)
    return _build(PipelineConfig, _read_yaml(path), path)


def load_livenflx_config(path: Path | str = Path("configs/livenflx.yaml")) -> LiveNflxConfig:
    """Load the LIVE-Netflix regression config."""
    path = Path(path)
    return _build(LiveNflxConfig, _read_yaml(path), path)


def load_ladder(path: Path | str) -> LadderConfig:
    """Load one ladder YAML file."""
    path = Path(path)
    data = _read_yaml(path)
    # Convenience: a ladder file may omit `name` and inherit it from the filename.
    data.setdefault("name", path.stem)
    return _build(LadderConfig, data, path)
