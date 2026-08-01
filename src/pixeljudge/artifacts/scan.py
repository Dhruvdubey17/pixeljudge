"""Sample frames from a clip, score them, and keep the worst ones as evidence.

Scanning every frame of every rung would be wasteful: artifacts are properties of
the encode, not of one moment, so a dozen evenly spread frames characterise a clip
well enough to rank it. What we do keep per clip is the worst frame, because the
deliverable is not a number, it is "look at this frame while VMAF tells you it is
fine".

**The detectors are no-reference, but the question is full-reference.** That
distinction cost this project a false result. The Sintel master scores 52 for banding
all by itself: it is stylised animation with graded skies, and those contours are in
the *content*. Its encodes score about the same, so an absolute banding score flags
them all - including a rung at VMAF 99.7, which looks like a spectacular metric blind
spot and is nothing of the kind. VMAF is doing its job correctly there: the encode
really is faithful to a master that already has contours.

So whenever a reference is available, this module scores it too and reports the
**delta**. "Banding the encode introduced" is the only version of the question that a
full-reference claim can rest on.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..errors import FfmpegError
from ..io.ffmpeg import probe, run_ffmpeg
from ..logging_conf import get_logger
from .detectors import ArtifactScores, score_frame

log = get_logger(__name__)

# A frame's worth of introduced banding below this is noise, not an artifact.
DEFAULT_DELTA_FLOOR = 5.0


@dataclass(frozen=True)
class ClipArtifacts:
    """Aggregated artifact scores for one clip.

    The ``*_delta_*`` fields are populated only when a reference was supplied. They are
    computed per frame (both clips are sampled at the same deterministic indices) and
    then pooled, so ``banding_delta_max`` is the worst *introduced* banding rather than
    the difference of two maxima.
    """

    path: str
    n_sampled: int
    banding_mean: float
    banding_max: float
    banding_worst_frame: int
    blocking_mean: float
    blocking_max: float
    blur_mean: float
    flat_fraction_mean: float
    reference: str | None = None
    reference_banding_mean: float | None = None
    banding_delta_mean: float | None = None
    banding_delta_max: float | None = None
    banding_delta_worst_frame: int | None = None
    blocking_delta_mean: float | None = None

    def as_row(self) -> dict[str, object]:
        return dict(asdict(self))


def sample_frame_indices(n_frames: int, k: int) -> list[int]:
    """Pick ``k`` frame indices spread evenly through a clip of ``n_frames``.

    Interior sampling on purpose: the first frame of an encode is a keyframe and is
    unrepresentatively clean, and the last frame is where a truncated file tends to be
    broken. Sampling at the midpoints of ``k`` equal segments avoids both while staying
    deterministic, which matters twice over: a test asserting "this clip bands" has to
    mean the same thing on every run, and a distorted clip and its reference must be
    sampled at the *same* frames for a per-frame delta to mean anything.
    """
    if n_frames <= 0 or k <= 0:
        return []
    if k >= n_frames:
        return list(range(n_frames))
    return [min(n_frames - 1, int((i + 0.5) * n_frames / k)) for i in range(k)]


def _frames_via_opencv(path: Path, indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        return
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                log.debug("opencv could not read frame %d of %s", index, path.name)
                continue
            yield index, frame
    finally:
        capture.release()


def _frames_via_ffmpeg(path: Path, indices: Sequence[int]) -> Iterator[tuple[int, np.ndarray]]:
    """Extract the requested frames as PNGs with ffmpeg, then read them back.

    The ``select`` filter takes an expression over the frame number ``n``.
    ``-fps_mode passthrough`` stops ffmpeg from duplicating or dropping frames to fit
    an output rate, which would break the mapping from output file back to frame
    index. (It is the modern spelling of ``-vsync 0``, which ffmpeg 8 warns about.)
    """
    if not indices:
        return
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    with tempfile.TemporaryDirectory(prefix="pixeljudge-frames-") as tmp:
        pattern = Path(tmp) / "frame_%04d.png"
        run_ffmpeg(
            [
                "-i",
                str(path),
                "-vf",
                f"select='{expression}'",
                "-fps_mode",
                "passthrough",
                str(pattern),
            ]
        )
        extracted = sorted(Path(tmp).glob("frame_*.png"))
        for index, image_path in zip(indices, extracted, strict=False):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            yield index, image


def read_frames(path: Path, indices: Sequence[int]) -> list[tuple[int, np.ndarray]]:
    """Frames at the given indices, whichever decoder can actually produce them.

    OpenCV first, ffmpeg as a fallback. That is not belt-and-braces: OpenCV builds
    routinely lack an AV1 decoder, and we encode AV1, so without the fallback the scan
    would silently skip a whole codec.
    """
    frames = list(_frames_via_opencv(path, indices))
    if len(frames) < len(indices):
        log.debug(
            "opencv returned %d/%d frames for %s; falling back to ffmpeg",
            len(frames),
            len(indices),
            path.name,
        )
        try:
            frames = list(_frames_via_ffmpeg(path, indices))
        except FfmpegError as exc:
            log.error("could not extract frames from %s: %s", path.name, exc)
    return frames


def scan_frames(
    path: Path | str, n_sample: int = 12
) -> list[tuple[int, ArtifactScores, np.ndarray]]:
    """Score sampled frames of one clip, keeping the pixels for evidence."""
    path = Path(path)
    info = probe(path)
    indices = sample_frame_indices(info.n_frames or 0, n_sample)
    frames = read_frames(path, indices)
    if not frames:
        raise FfmpegError(f"no frames could be decoded from {path}")
    return [(index, score_frame(frame), frame) for index, frame in frames]


def scan_clip(
    path: Path | str,
    *,
    n_sample: int = 12,
    evidence_dir: Path | None = None,
    reference: Path | str | None = None,
    reference_scores: list[tuple[int, ArtifactScores, np.ndarray]] | None = None,
) -> ClipArtifacts:
    """Score one clip, optionally against a reference to get introduced-artifact deltas.

    When a reference is given, the evidence frame saved is the one where the encode
    introduced the *most* banding, not the one with the highest absolute score. On
    content whose master already has contours those are different frames, and only the
    first answers "what did the encoder do".
    """
    path = Path(path)
    scored = scan_frames(path, n_sample)
    banding = np.array([s.banding for _, s, _ in scored])
    blocking = np.array([s.blocking for _, s, _ in scored])
    blur = np.array([s.blur for _, s, _ in scored])
    flat = np.array([s.flat_fraction for _, s, _ in scored])
    worst = int(np.argmax(banding))

    deltas: dict[str, float | int | str | None] = {
        "reference": None,
        "reference_banding_mean": None,
        "banding_delta_mean": None,
        "banding_delta_max": None,
        "banding_delta_worst_frame": None,
        "blocking_delta_mean": None,
    }
    evidence_index = scored[worst][0]
    evidence_frame = scored[worst][2]

    if reference is not None:
        reference = Path(reference)
        if reference_scores is None:
            reference_scores = scan_frames(reference, n_sample)
        paired = _pair_by_index(scored, reference_scores)
        if paired:
            banding_delta = np.array([d.banding - r.banding for _, d, r in paired])
            blocking_delta = np.array([d.blocking - r.blocking for _, d, r in paired])
            worst_delta = int(np.argmax(banding_delta))
            deltas.update(
                {
                    "reference": str(reference),
                    "reference_banding_mean": float(
                        np.mean([s.banding for _, s, _ in reference_scores])
                    ),
                    "banding_delta_mean": float(banding_delta.mean()),
                    "banding_delta_max": float(banding_delta.max()),
                    "banding_delta_worst_frame": int(paired[worst_delta][0]),
                    "blocking_delta_mean": float(blocking_delta.mean()),
                }
            )
            evidence_index = paired[worst_delta][0]
            evidence_frame = next(f for i, _, f in scored if i == evidence_index)
        else:
            log.warning(
                "%s and %s share no sampled frame indices; no delta computed",
                path.name,
                reference.name,
            )

    if evidence_dir is not None:
        save_evidence_frame(
            evidence_frame, evidence_dir / f"{path.stem}__banding_frame{evidence_index}.png"
        )
        # Save the master's matching frame too. A picture of a banded encode proves
        # nothing on its own: the only way to show the *encoder* did it is to put the
        # same frame of the master beside it.
        if reference_scores is not None:
            match = next((f for i, _, f in reference_scores if i == evidence_index), None)
            if match is not None:
                save_evidence_frame(
                    match, evidence_dir / f"{path.stem}__reference_frame{evidence_index}.png"
                )

    return ClipArtifacts(
        path=str(path),
        n_sampled=len(scored),
        banding_mean=float(banding.mean()),
        banding_max=float(banding.max()),
        banding_worst_frame=int(scored[worst][0]),
        blocking_mean=float(blocking.mean()),
        blocking_max=float(blocking.max()),
        blur_mean=float(blur.mean()),
        flat_fraction_mean=float(flat.mean()),
        **deltas,  # type: ignore[arg-type]
    )


def _pair_by_index(
    distorted: list[tuple[int, ArtifactScores, np.ndarray]],
    reference: list[tuple[int, ArtifactScores, np.ndarray]],
) -> list[tuple[int, ArtifactScores, ArtifactScores]]:
    """Match frames by index. Both clips are sampled deterministically, so the indices
    line up unless one of them failed to decode a frame."""
    by_index = {index: scores for index, scores, _ in reference}
    return [(index, scores, by_index[index]) for index, scores, _ in distorted if index in by_index]


def save_evidence_frame(frame: np.ndarray, destination: Path) -> Path:
    """Write a frame as PNG. Lossless, because the point is to look at artifacts."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), frame):
        raise FfmpegError(f"could not write evidence frame to {destination}")
    return destination


def scan_many(
    paths: Sequence[Path | str],
    *,
    references: Mapping[str, str] | None = None,
    n_sample: int = 12,
    evidence_dir: Path | None = None,
) -> pd.DataFrame:
    """Scan a batch of clips into one table, skipping (loudly) any that fail.

    ``references`` maps a distorted path to its master. Each master is scanned once and
    reused across all of its rungs, which matters: a four-codec sweep has thirty-five
    rungs per master.
    """
    references = references or {}
    cache: dict[str, list[tuple[int, ArtifactScores, np.ndarray]]] = {}
    rows: list[dict[str, object]] = []

    for path in paths:
        reference = references.get(str(path))
        try:
            if reference is not None and reference not in cache:
                cache[reference] = scan_frames(reference, n_sample)
            rows.append(
                scan_clip(
                    path,
                    n_sample=n_sample,
                    evidence_dir=evidence_dir,
                    reference=reference,
                    reference_scores=cache.get(reference) if reference else None,
                ).as_row()
            )
        except FfmpegError as exc:
            log.error("artifact scan failed for %s: %s", Path(path).name, exc)
    return pd.DataFrame(rows)


def find_metric_blind_spots(
    metrics: pd.DataFrame,
    artifacts: pd.DataFrame,
    *,
    vmaf_column: str = "vmaf",
    psnr_column: str = "psnr_y",
    vmaf_floor: float = 80.0,
    psnr_floor: float = 38.0,
    banding_column: str = "banding_delta_max",
    banding_quantile: float = 0.75,
    banding_floor: float | None = DEFAULT_DELTA_FLOOR,
) -> pd.DataFrame:
    """Find clips the metrics call good while the encode visibly introduced banding.

    The rule is deliberately blunt and stated up front rather than tuned until it
    produced something. A clip counts when VMAF is above ``vmaf_floor`` and PSNR above
    ``psnr_floor`` - both comfortably "this is fine" territory - and the banding it
    *added* relative to its master clears the bar.

    The floors come from how these metrics are normally read: VMAF 80+ is "good", and
    PSNR above roughly 38-40 dB is usually called high quality. Neither is a perceptual
    guarantee, which is the point.

    ``banding_column`` defaults to the **delta** against the reference, not the absolute
    score. Using the absolute score here produces a confident false positive on
    stylised content: the Sintel master scores 52 for banding on its own, so every one
    of its rungs looks banded, including one at VMAF 99.7 where the encoder did nothing
    wrong at all. Absolute banding answers "does this frame have contours"; only the
    delta answers "did compression put them there".

    The bar is an absolute floor by default (``DEFAULT_DELTA_FLOOR``), because a delta
    is already on a meaningful scale. Pass ``banding_floor=None`` to use the population
    quantile instead, which is only sensible with plenty of clips.
    """
    if metrics.empty or artifacts.empty:
        return pd.DataFrame()

    merged = metrics.merge(
        artifacts.assign(distorted=[Path(p).name for p in artifacts["path"]]),
        on="distorted",
        how="inner",
        suffixes=("", "_artifact"),
    )
    if merged.empty:
        log.warning("no clips matched between the metrics and artifact tables")
        return merged
    if banding_column not in merged.columns or merged[banding_column].isna().all():
        log.warning(
            "%s is not available (was the scan run without references?); "
            "cannot separate introduced banding from banding already in the master",
            banding_column,
        )
        return merged.iloc[0:0]

    threshold = (
        float(banding_floor)
        if banding_floor is not None
        else float(merged[banding_column].quantile(banding_quantile))
    )
    blind = merged[
        (merged[vmaf_column] >= vmaf_floor)
        & (merged.get(psnr_column, pd.Series(np.inf, index=merged.index)) >= psnr_floor)
        & (merged[banding_column] >= threshold)
    ].copy()
    blind["banding_threshold"] = round(threshold, 3)
    return blind.sort_values(banding_column, ascending=False).reset_index(drop=True)
