"""Video quality measurement: one libvmaf pass gives us every metric at once.

Running PSNR, SSIM, MS-SSIM and VMAF as four separate ffmpeg passes would decode
both clips four times. libvmaf can compute the others as "features" alongside
VMAF, so we decode once and parse one JSON log.

Everything below the filtergraph builder is the accumulated list of things that
go wrong, written down as code:

1. **Input order.** libvmaf judges its *first* input against its second. Swap them
   and the numbers still look plausible, which is what makes it dangerous.
2. **Timestamps.** libvmaf pairs frames by presentation time, not by index. A
   distorted clip whose first PTS is not zero gets compared against the wrong
   reference frames and scores mysteriously badly. ``setpts=PTS-STARTPTS`` on both
   inputs removes the whole class of bug.
3. **Resolution.** Full-reference metrics need identical dimensions, so a 360p
   rung is scaled back up to the master's resolution before comparison. That is
   not a hack: it is what a real player does when it fills a 1080p screen with a
   360p stream, so it is the quality the viewer actually sees.
4. **Feature names.** ``float_ssim``/``float_ms_ssim``, not ``ssim``/``ms_ssim``.
   The plain names are rejected by several libvmaf builds.
5. **Provenance.** A VMAF number without its model, pooling method and library
   version is not a result, it is a rumour. Every measurement records all three.
"""

from __future__ import annotations

import json
import math
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import VmafConfig
from ..errors import FfmpegError, MetricsError
from ..io.ffmpeg import (
    ffmpeg_version,
    has_libvmaf,
    probe,
    require_libvmaf,
    run_ffmpeg,
)
from ..logging_conf import get_logger

log = get_logger(__name__)

# Metric column names we normalise to, whatever the build calls them.
VMAF = "vmaf"
PSNR = "psnr_y"
SSIM = "float_ssim"
MS_SSIM = "float_ms_ssim"
CAMBI = "cambi"

# libvmaf has changed these key names between versions; map everything onto ours.
_KEY_ALIASES: dict[str, str] = {
    "psnr": PSNR,
    "psnr_y": PSNR,
    "float_psnr": PSNR,
    "ssim": SSIM,
    "float_ssim": SSIM,
    "ms_ssim": MS_SSIM,
    "float_ms_ssim": MS_SSIM,
    "vmaf": VMAF,
    "cambi": CAMBI,
}

# VMAF's own elementary features come free in the log: they are what its SVM sees
# before fusion. Keeping them gives the regression phase something to ablate
# against ("is fused VMAF better than its own ingredients?") at zero extra cost.
ELEMENTARY_ALIASES: dict[str, str] = {
    "integer_adm2": "adm2",  # detail loss, the biggest single contributor
    "integer_vif_scale0": "vif_scale0",  # information fidelity, coarse to fine
    "integer_vif_scale1": "vif_scale1",
    "integer_vif_scale2": "vif_scale2",
    "integer_vif_scale3": "vif_scale3",
    "integer_motion2": "motion2",  # temporal difference; motion masks distortion
}
_KEY_ALIASES.update(ELEMENTARY_ALIASES)

POOL_METHODS = ("mean", "harmonic_mean", "min", "max")


@dataclass(frozen=True)
class QualityResult:
    """Per-frame scores, pooled scores, and the provenance needed to trust them."""

    reference: str
    distorted: str
    per_frame: pd.DataFrame
    pooled: dict[str, float]
    pooled_all: dict[str, dict[str, float]] = field(default_factory=dict)
    context: dict[str, str] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(len(self.per_frame))

    @property
    def metrics(self) -> list[str]:
        return [column for column in self.per_frame.columns if column != "frame"]

    def summary_row(self) -> dict[str, Any]:
        """One flat row for a feature table, provenance included."""
        row: dict[str, Any] = {
            "reference": Path(self.reference).name,
            "distorted": Path(self.distorted).name,
            "n_frames": self.n_frames,
        }
        row.update(self.pooled)
        # Worst-frame VMAF is worth keeping: a clip can pool well and still have a
        # stretch that ruins the viewing experience.
        if VMAF in self.pooled_all:
            row["vmaf_min"] = self.pooled_all[VMAF]["min"]
            row["vmaf_mean"] = self.pooled_all[VMAF]["mean"]
        row.update({f"ctx_{key}": value for key, value in self.context.items()})
        return row

    def save(self, out_dir: Path, stem: str | None = None) -> tuple[Path, Path]:
        """Write ``<stem>.frames.csv`` and ``<stem>.summary.json``."""
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = stem or Path(self.distorted).stem
        frames_path = out_dir / f"{stem}.frames.csv"
        summary_path = out_dir / f"{stem}.summary.json"
        self.per_frame.to_csv(frames_path, index=False)
        summary_path.write_text(
            json.dumps(
                {
                    "reference": self.reference,
                    "distorted": self.distorted,
                    "n_frames": self.n_frames,
                    "pooled": self.pooled,
                    "pooled_all": self.pooled_all,
                    "context": self.context,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return frames_path, summary_path


def build_libvmaf_options(cfg: VmafConfig, log_path: Path) -> str:
    """The option string for the libvmaf filter.

    Values that themselves contain ``=`` or ``|`` are wrapped in single quotes,
    which is how ffmpeg's filter parser wants nested key-value lists.
    """
    features = list(cfg.features)
    if cfg.enable_cambi and "cambi" not in features:
        features.append("cambi")
    feature_list = "|".join(f"name={name}" for name in features)

    options = [
        f"log_path={_escape_path(log_path)}",
        "log_fmt=json",
        f"model='version={cfg.vmaf_model}'",
        f"feature='{feature_list}'",
        f"pool={cfg.pool}",
    ]
    if cfg.n_threads:
        options.append(f"n_threads={cfg.n_threads}")
    if cfg.n_subsample > 1:
        options.append(f"n_subsample={cfg.n_subsample}")
    return "libvmaf=" + ":".join(options)


def _escape_path(path: Path) -> str:
    """ffmpeg filter values treat ``:`` and ``\\`` specially."""
    return str(path).replace("\\", "/").replace(":", r"\:")


def timestamp_filter(ref_fps: float | None) -> str:
    """Filters that put both inputs on one timestamp grid.

    ``setpts=PTS-STARTPTS`` alone is **not enough**, which cost this project a set of
    wrong numbers. It zeroes the start but leaves each container's own timebase in
    place, and Matroska/WebM stores timestamps in milliseconds while MP4 uses a
    fine-grained tick. At 24 fps the WebM frame times round to 0, 42, 83, 125 ms while
    the reference sits at exact 1/24 s intervals, so libvmaf's frame pairing (nearest
    lower-or-equal timestamp) slips by one frame intermittently and invents an extra
    pair at the end. Every metric collapses: on one VP9 rung, PSNR read 39.0 dB
    instead of 47.3 and VMAF 71 instead of 99.

    So when the reference frame rate is known we regenerate timestamps from the frame
    *index* on a shared timebase (``settb=AVTB``), which makes frame k of both inputs
    land on exactly the same timestamp. Pairing by index is also the only thing a
    full-reference comparison of the same content can mean.
    """
    if ref_fps and ref_fps > 0:
        return f"settb=AVTB,setpts=N/{ref_fps:.6f}/TB"
    # Unknown frame rate: fall back to zeroing the start, which is better than nothing.
    log.warning("reference frame rate unknown; falling back to PTS-STARTPTS alignment")
    return "setpts=PTS-STARTPTS"


def build_filtergraph(
    cfg: VmafConfig,
    ref_width: int,
    ref_height: int,
    log_path: Path,
    scale_flags: str = "bicubic",
    ref_fps: float | None = None,
) -> str:
    """Assemble the full graph. Input 0 is the distorted clip, input 1 the reference.

    Kept as a pure string builder so the argument order, the escaping and the
    timestamp handling can be unit-tested without ffmpeg anywhere near the test.
    """
    align = timestamp_filter(ref_fps)
    return (
        f"[0:v]{align},scale={ref_width}:{ref_height}:flags={scale_flags},"
        f"format=yuv420p[dist];"
        f"[1:v]{align},format=yuv420p[ref];"
        f"[dist][ref]{build_libvmaf_options(cfg, log_path)}"
    )


def measure_pair(
    distorted: Path | str,
    reference: Path | str,
    cfg: VmafConfig | None = None,
    *,
    scale_flags: str = "bicubic",
    timeout: float | None = None,
) -> QualityResult:
    """Measure one distorted clip against its reference.

    Raises :class:`~pixeljudge.errors.MissingDependencyError` when the installed
    ffmpeg has no libvmaf, and :class:`MetricsError` when the log is unusable.
    """
    cfg = cfg or VmafConfig()
    distorted, reference = Path(distorted), Path(reference)
    require_libvmaf()

    ref_info = probe(reference)
    dist_info = probe(distorted)
    if ref_info.n_frames and dist_info.n_frames and ref_info.n_frames != dist_info.n_frames:
        # Not fatal: libvmaf pairs by timestamp, so a one-frame difference is
        # harmless. A large gap means the clips are not the same content.
        log.warning(
            "frame count mismatch: reference %s has %s frames, %s has %s",
            reference.name,
            ref_info.n_frames,
            distorted.name,
            dist_info.n_frames,
        )

    with tempfile.TemporaryDirectory(prefix="pixeljudge-vmaf-") as tmp:
        log_path = Path(tmp) / "vmaf.json"
        graph = build_filtergraph(
            cfg,
            ref_info.width,
            ref_info.height,
            log_path,
            scale_flags=scale_flags,
            ref_fps=ref_info.fps,
        )
        log.info(
            "measure %s vs %s (%s, model %s)",
            distorted.name,
            reference.name,
            f"{dist_info.resolution} -> {ref_info.resolution}",
            cfg.vmaf_model,
        )
        run_ffmpeg(
            [
                "-i",
                str(distorted),
                "-i",
                str(reference),
                "-lavfi",
                graph,
                "-f",
                "null",
                "-",
            ],
            timeout=timeout,
        )
        if not log_path.exists():
            raise MetricsError(
                "libvmaf produced no log file; the filter ran but wrote nothing to " f"{log_path}"
            )
        payload = json.loads(log_path.read_text(encoding="utf-8"))

    per_frame = parse_vmaf_log(payload)
    if ref_info.n_frames and len(per_frame) != ref_info.n_frames:
        # The tell that caught a real misalignment bug: libvmaf scored 145 pairs for a
        # 144-frame reference, because the two containers' timebases disagreed. If the
        # pair count is not the reference's frame count, the comparison is suspect and
        # every metric below is affected.
        log.warning(
            "%s: libvmaf scored %d frame pairs against a %d-frame reference. "
            "The clips may be misaligned; treat these numbers with suspicion.",
            distorted.name,
            len(per_frame),
            ref_info.n_frames,
        )
    pooled_all = pool_all(per_frame)
    pooled = {metric: values[cfg.pool] for metric, values in pooled_all.items()}
    context = {
        "vmaf_model": cfg.vmaf_model,
        "pool": cfg.pool,
        "features": ",".join(cfg.features),
        "frame_alignment": "index" if ref_info.fps > 0 else "pts-startpts",
        "ffmpeg_version": ffmpeg_version(),
        "libvmaf_version": str(payload.get("version", "unknown")),
        "reference_resolution": ref_info.resolution,
        "distorted_resolution": dist_info.resolution,
        "scale_flags": scale_flags,
    }
    result = QualityResult(
        reference=str(reference),
        distorted=str(distorted),
        per_frame=per_frame,
        pooled=pooled,
        pooled_all=pooled_all,
        context=context,
    )
    log.info(
        "  %s",
        ", ".join(f"{name}={value:.4g}" for name, value in sorted(result.pooled.items())),
    )
    return result


def measure_many(
    jobs: Sequence[tuple[Path, Path, Mapping[str, Any]]],
    cfg: VmafConfig | None = None,
    *,
    scale_flags: str = "bicubic",
    per_frame_dir: Path | None = None,
) -> pd.DataFrame:
    """Measure a batch of (distorted, reference, extra columns) jobs.

    Deliberately decoupled from the encoder's own record type: anything that can
    produce a path pair and a dict of context columns can drive this, which keeps
    the metrics package independent of how the files were produced.

    A failure on one clip logs and continues. Losing an hour of measurement
    because clip 19 of 40 was truncated would be a bad trade.
    """
    cfg = cfg or VmafConfig()
    rows: list[dict[str, Any]] = []
    for distorted, reference, extra in jobs:
        try:
            result = measure_pair(distorted, reference, cfg, scale_flags=scale_flags)
        except (MetricsError, FfmpegError) as exc:
            log.error("measurement failed for %s: %s", Path(distorted).name, exc)
            continue
        if per_frame_dir is not None:
            result.save(per_frame_dir)
        row = dict(extra)
        row.update(result.summary_row())
        rows.append(row)
    if not rows:
        raise MetricsError("no measurement succeeded; nothing to report")
    return pd.DataFrame(rows)


def parse_vmaf_log(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Turn a libvmaf JSON log into a per-frame DataFrame.

    Defensive on purpose: this is the boundary where another tool's output becomes
    our data. Different libvmaf versions name PSNR ``psnr`` or ``psnr_y``, some
    omit the pooled block entirely, and a truncated run leaves an empty frame
    list. Each of those gets a clear error or a normalised column instead of a
    ``KeyError`` three modules later.
    """
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise MetricsError(
            "libvmaf log contains no frames; the measurement produced nothing usable"
        )

    rows: list[dict[str, float]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise MetricsError(f"libvmaf log frame {index} is not an object")
        metrics = frame.get("metrics")
        if not isinstance(metrics, Mapping) or not metrics:
            raise MetricsError(f"libvmaf log frame {index} has no metrics block")
        row: dict[str, float] = {"frame": float(frame.get("frameNum", index))}
        for key, value in metrics.items():
            column = _KEY_ALIASES.get(str(key))
            if column is None:
                continue  # a feature we did not ask for (e.g. psnr_cb)
            row[column] = _as_float(value, column, index)
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.columns.size <= 1:
        raise MetricsError(
            "libvmaf log had frames but no recognised metrics; expected one of "
            f"{sorted(set(_KEY_ALIASES.values()))}"
        )
    table["frame"] = table["frame"].astype(int)
    headline = [c for c in (VMAF, PSNR, SSIM, MS_SSIM, CAMBI) if c in table.columns]
    ordered = ["frame", *headline]
    return table[ordered + [c for c in table.columns if c not in ordered]]


def _as_float(value: Any, column: str, frame_index: int) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricsError(
            f"non-numeric {column} value {value!r} in libvmaf log frame {frame_index}"
        ) from exc
    return number


def harmonic_mean(values: Sequence[float]) -> float:
    """Harmonic mean the way libvmaf does it, with a +1 offset.

    The offset exists because a legitimate score can be zero (VMAF 0 for a
    destroyed frame, SSIM 0 for no structural similarity) and the plain harmonic
    mean divides by it. Offsetting by one keeps the "punish the worst frames"
    behaviour without the singularity.
    """
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return float("nan")
    total = sum(1.0 / (v + 1.0) for v in finite)
    if total == 0:
        return float("nan")
    return len(finite) / total - 1.0


def pool_metric(values: Iterable[float]) -> dict[str, float]:
    """Every pooling method for one metric, so the choice is visible in the output."""
    series = pd.Series(list(values), dtype="float64")
    finite = series[series.apply(math.isfinite)]
    if finite.empty:
        return dict.fromkeys((*POOL_METHODS, "std"), float("nan"))
    return {
        "mean": float(finite.mean()),
        "harmonic_mean": harmonic_mean(finite.tolist()),
        "min": float(finite.min()),
        "max": float(finite.max()),
        "std": float(finite.std(ddof=0)),
    }


def pool_all(per_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Pool every metric column.

    We pool from the per-frame data rather than reading the log's ``pooled_metrics``
    block: not every build writes it, the harmonic-mean definition has changed
    between versions, and computing it ourselves means one definition across the
    whole project.
    """
    return {
        column: pool_metric(per_frame[column]) for column in per_frame.columns if column != "frame"
    }


def vmaf_available() -> bool:
    """Convenience for tests and the CLI."""
    return has_libvmaf()
