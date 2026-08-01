"""Reading the LIVE-Netflix Video QoE Database (VideoATLAS release).

This dataset closes the one thing NFLX-P could not: it ships human scores *and*
pre-computed per-frame quality vectors in the same files, so the metrics-to-opinion
regression can be built and evaluated without waiting on a media access grant.

**What is actually on disk.** 112 MATLAB files named ``content_{1..14}_seq_{0..7}``
- 14 source contents crossed with 8 playout patterns. Each file holds one video's
per-frame vectors (``VMAF_vec``, ``PSNR_vec``, ``SSIM_vec``, ``MSSIM_vec``, ...),
the pre-pooled scalars the release's own scripts use, the QoE covariates and the
subjective score. Note ``MSSIM``, not ``MS_SSIM``; the keys are upper-case.

**The score is not a MOS.** ``final_subj_score`` is Z-scored per viewing session
and per subject, so it is centred near zero and runs roughly -1.6 to 1.6, not 1-5.
It is exposed as ``subj_score`` rather than ``mos`` because renaming it would
misdescribe its units. Correlation coefficients are unaffected by the scale; RMSE
is in Z-score units and only comparable within this dataset.

**Why the condition scope matters more than anything else here.** LIVE-Netflix is a
*Quality of Experience* database. Its 8 playout patterns include rebuffering
(stalls) as well as bitrate adaptation, and the human score reflects all of it. The
features this project computes - PSNR, SSIM, MS-SSIM, VMAF - measure compression
distortion. They cannot see a stall: a frozen frame is not a distorted frame.

The dataset itself makes that boundary literal. For the 56 videos with a stall,
``Nframes`` exceeds the length of every quality vector by exactly ``ds * vid_fps``,
because the vectors are computed on the stall-removed video (a full-reference
metric needs its two inputs time-aligned). So for those conditions there is a
measured stretch of the viewer's experience that no vector covers.

Hence :data:`CONDITION_SCOPES`:

``compression_only``
    ``ns == 0``: the four continuous-playback patterns (0, 2, 4, 7), 56 videos over
    all 14 contents. Bitrate adaptation is *kept* - the metrics see it fine, as a
    dip in the per-frame quality - only rebuffering is excluded. Features and label
    then describe the same phenomenon. This is the default and the honest scope for
    a compression-quality correlation.
``all``
    All 112 videos. Available for completeness, but a compression metric is not
    expected to predict a score that stalls helped produce, and any number from
    this scope must be labelled as QoE, not compression quality.

The scope is derived from the ``ns`` field rather than a hardcoded list of pattern
indices, so it follows the data.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ..errors import DatasetError
from ..logging_conf import get_logger

log = get_logger(__name__)

ConditionScope = Literal["compression_only", "all"]
CONDITION_SCOPES: tuple[str, ...] = ("compression_only", "all")

# The label column. Named for what it is - a Z-scored subjective score - and not
# 'mos', because it is not on a mean-opinion 1-5 scale. See the module docstring.
LABEL_COLUMN = "subj_score"
GROUP_COLUMN = "content"

# On-disk metric key -> the name used everywhere in this project. The release
# writes upper-case keys and calls MS-SSIM 'MSSIM'; PixelJudge's own measurement
# output uses libvmaf's lower-case names, and matching them is what lets the
# Stage 3 comparison line up column-for-column without a translation layer.
METRIC_KEYS: dict[str, str] = {
    "VMAF": "vmaf",
    "PSNR": "psnr",
    "SSIM": "ssim",
    "MSSIM": "ms_ssim",
    "STRRED": "strred",
    "NIQE": "niqe",
}

# The four metrics PixelJudge can also compute itself, so they are the ones the
# regression uses by default. STRRED and NIQE are parsed and kept in the long
# frame - they are the release's own headline features and useful context - but
# reproducing them is out of scope for a libvmaf-based pipeline.
FULL_REFERENCE_METRICS: tuple[str, ...] = ("vmaf", "psnr", "ssim", "ms_ssim")

# Per-video scalars worth carrying: the QoE covariates that define the condition,
# plus the timing fields needed to verify the stall-alignment relationship.
SCALAR_KEYS: dict[str, str] = {
    "final_subj_score": LABEL_COLUMN,
    "ns": "n_stalls",  # number of rebuffering events
    "ds": "stall_seconds",  # total duration of those stalls
    "lt": "low_bitrate_seconds",  # time spent encoded below 250 kbps
    "tsl": "time_since_last_impairment",
    "Nframes": "n_playout_frames",  # includes frames frozen during a stall
    "vid_fps": "fps",
}

_FILENAME = re.compile(r"^content_(\d+)_seq_(\d+)$")

# Fewest held-out videos a trial can have and still yield a correlation worth
# reporting; below three, Spearman is undefined or meaningless.
MIN_TEST_ROWS = 3


def _scalar(payload: dict[str, Any], key: str, source: str) -> float:
    """Pull a 1x1 MATLAB array out as a plain float.

    ``loadmat`` wraps every scalar in a 2-D array, so ``payload['ns']`` is
    ``array([[0]], dtype=uint8)`` rather than ``0``. Unwrapping in one place keeps
    the rest of the module free of ``[0][0]`` indexing.
    """
    if key not in payload:
        raise DatasetError(f"{source}: missing field {key!r}")
    flat = np.ravel(np.asarray(payload[key]))
    if flat.size != 1:
        raise DatasetError(f"{source}: expected {key!r} to be a single value, got {flat.size}")
    return float(flat[0])


def _vector(payload: dict[str, Any], key: str, source: str) -> np.ndarray:
    if key not in payload:
        raise DatasetError(f"{source}: missing per-frame vector {key!r}")
    values = np.ravel(np.asarray(payload[key], dtype=float))
    if values.size == 0:
        raise DatasetError(f"{source}: {key!r} is empty")
    return values


def read_video_mat(path: Path | str) -> dict[str, Any]:
    """Parse one ``content_C_seq_S.mat`` into scalars plus per-frame vectors.

    Returns a dict with the identity fields, the scalars from
    :data:`SCALAR_KEYS`, and one ``numpy`` array per metric under its project
    name. The per-frame arrays are checked for equal length here rather than
    downstream: within a file every metric is scored on the same frames, and a
    ragged set means the file is damaged, not that pooling should paper over it.
    """
    from scipy.io import loadmat

    path = Path(path)
    match = _FILENAME.match(path.stem)
    if match is None:
        raise DatasetError(
            f"{path.name} does not follow the release's content_C_seq_S naming convention"
        )
    content_index, sequence_index = int(match.group(1)), int(match.group(2))

    try:
        payload: dict[str, Any] = loadmat(str(path))
    except (ValueError, TypeError, OSError) as exc:
        raise DatasetError(f"{path} is not a readable MATLAB file: {exc}") from exc

    record: dict[str, Any] = {
        "video_id": path.stem,
        GROUP_COLUMN: f"content_{content_index:02d}",
        "content_index": content_index,
        # 'condition' is the playout pattern: the release's 8 fixed streaming
        # scenarios, identical across contents, so the index is the label.
        "condition": f"seq_{sequence_index}",
        "condition_index": sequence_index,
    }
    for key, column in SCALAR_KEYS.items():
        record[column] = _scalar(payload, key, path.name)

    lengths: dict[str, int] = {}
    for release_key, name in METRIC_KEYS.items():
        values = _vector(payload, f"{release_key}_vec", path.name)
        record[name] = values
        lengths[name] = values.size
    if len(set(lengths.values())) != 1:
        raise DatasetError(f"{path.name}: per-frame vectors disagree in length: {lengths}")

    record["n_frames"] = next(iter(lengths.values()))
    return record


def load_livenflx(
    release_dir: Path | str,
    *,
    scope: ConditionScope = "compression_only",
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Every in-scope video as a tidy long frame: one row per frame per metric.

    Columns: ``video_id``, ``content``, ``condition``, ``metric``, ``frame``,
    ``value``, plus the subjective score and QoE covariates repeated on each row.
    Repeating ``content`` and the label on every row is deliberate redundancy - it
    means no downstream step can pool or split without the grouping key in hand,
    which is the thing that prevents content leakage later.
    """
    release_dir = Path(release_dir)
    if not release_dir.is_dir():
        raise DatasetError(
            f"LIVE-Netflix release directory not found: {release_dir}. "
            "See DATA_CARD.md for how to obtain the VideoATLAS release."
        )
    if scope not in CONDITION_SCOPES:
        raise DatasetError(f"unknown condition scope {scope!r}; expected one of {CONDITION_SCOPES}")

    files = sorted(release_dir.glob("content_*_seq_*.mat"))
    if not files:
        raise DatasetError(f"{release_dir} contains no content_*_seq_*.mat files")

    wanted = list(metrics) if metrics is not None else list(METRIC_KEYS.values())
    unknown = [name for name in wanted if name not in METRIC_KEYS.values()]
    if unknown:
        raise DatasetError(
            f"unknown metric(s) {unknown}; available: {sorted(METRIC_KEYS.values())}"
        )

    records = [read_video_mat(path) for path in files]
    kept = [record for record in records if _in_scope(record, scope)]
    if not kept:
        raise DatasetError(f"no video in {release_dir} matched scope {scope!r}")

    _log_scope(records, kept, scope)
    _check_stall_alignment(records)

    frames: list[pd.DataFrame] = []
    for record in kept:
        # Everything that is constant for the video, broadcast down its frames.
        shared = {
            key: value
            for key, value in record.items()
            if key not in METRIC_KEYS.values() and key != "n_frames"
        }
        for name in wanted:
            values = record[name]
            frames.append(
                pd.DataFrame(
                    {
                        **shared,
                        "metric": name,
                        "frame": np.arange(values.size, dtype=int),
                        "value": values,
                    }
                )
            )

    table = pd.concat(frames, ignore_index=True)
    log.info(
        "loaded %d frame-metric rows: %d videos, %d contents, %d metrics (scope=%s)",
        len(table),
        table["video_id"].nunique(),
        table[GROUP_COLUMN].nunique(),
        table["metric"].nunique(),
        scope,
    )
    return table


def _in_scope(record: dict[str, Any], scope: str) -> bool:
    """Continuous playback means no rebuffering events, i.e. ``ns == 0``.

    Read from the data rather than from a list of pattern indices, so the rule
    stays true if the release ever adds patterns.
    """
    if scope == "all":
        return True
    return bool(record["n_stalls"] == 0)


def _log_scope(records: list[dict[str, Any]], kept: list[dict[str, Any]], scope: str) -> None:
    dropped = len(records) - len(kept)
    if dropped:
        conditions = sorted({record["condition"] for record in records if record not in kept})
        log.info(
            "scope=%s dropped %d of %d videos containing rebuffering (%s); "
            "compression metrics cannot observe a stall, so those rows would ask "
            "the features to explain something they never measured",
            scope,
            dropped,
            len(records),
            ", ".join(conditions),
        )
    elif scope == "all":
        log.warning(
            "scope=all keeps %d videos including rebuffering conditions; any correlation "
            "from this scope describes QoE, not compression quality",
            len(kept),
        )


def _check_stall_alignment(records: list[dict[str, Any]]) -> None:
    """Verify ``Nframes - len(vec) == ds * fps`` on the stalled videos.

    Not a formality. The gap between the playout frame count and the vector length
    is the dataset's own statement of what the quality vectors do and do not cover,
    and it is the evidence behind the default scope. If the relationship ever stops
    holding, the assumption that ``ns == 0`` implies full coverage has to be
    revisited rather than trusted.
    """
    mismatched = []
    for record in records:
        gap = record["n_playout_frames"] - record["n_frames"]
        expected = record["stall_seconds"] * record["fps"]
        if abs(gap - expected) > 1.0:
            mismatched.append(f"{record['video_id']} (gap {gap:.0f}, expected {expected:.0f})")
    if mismatched:
        log.warning(
            "%d video(s) where the playout/vector frame gap is not the stall duration: %s. "
            "The quality vectors may not be stall-aligned as documented",
            len(mismatched),
            ", ".join(mismatched[:3]),
        )


def load_release_splits(path: Path | str) -> np.ndarray:
    """The release's 1000 pre-generated 80/20 content splits, as a boolean matrix.

    Shape ``(112, 1000)``; ``True`` at ``(i, j)`` means video ``i`` is a *training*
    video on trial ``j``. Row order is the release's natural sort of the filenames,
    i.e. ``i = (content_index - 1) * 8 + sequence_index``, which
    :func:`released_split_masks` relies on.

    Worth using because it is how the published LIVE-Netflix numbers were produced,
    so results from this split mode are comparable to the paper's; the project's own
    :class:`~sklearn.model_selection.GroupKFold` is the default because it is
    exhaustive rather than sampled.
    """
    from scipy.io import loadmat

    path = Path(path)
    if not path.exists():
        raise DatasetError(f"split matrix not found: {path}")
    payload = loadmat(str(path))
    keys = [key for key in payload if not key.startswith("__")]
    if len(keys) != 1:
        raise DatasetError(f"{path}: expected exactly one variable, found {keys}")
    matrix = np.asarray(payload[keys[0]])
    if matrix.ndim != 2:
        raise DatasetError(f"{path}: expected a 2-D matrix, got shape {matrix.shape}")
    return matrix.astype(bool)


def released_split_masks(
    table: pd.DataFrame, matrix: np.ndarray, *, n_trials: int | None = None
) -> list[np.ndarray]:
    """Turn the release's split matrix into train masks over ``table``'s rows.

    ``table`` is a per-video feature table, which after scoping may hold a subset
    of the 112 rows the matrix describes. The mapping is by video identity
    (``content_index``, ``condition_index``) rather than by position, so scoping to
    the continuous-playback conditions cannot silently shift every row against the
    wrong column of the matrix.

    Trials that come out degenerate after scoping are dropped rather than allowed
    to produce a meaningless number. The two sides need different things: the
    training side needs at least two contents, because hyperparameters are tuned
    with a content-grouped inner split; the test side only needs enough rows to
    compute a correlation over.
    """
    required = {"content_index", "condition_index"}
    missing = required - set(table.columns)
    if missing:
        raise DatasetError(f"feature table lacks {sorted(missing)}, needed to map release splits")

    positions = (table["content_index"].to_numpy(int) - 1) * 8 + table["condition_index"].to_numpy(
        int
    )
    if positions.max() >= matrix.shape[0]:
        raise DatasetError(
            f"video index {positions.max()} is outside the split matrix ({matrix.shape[0]} rows)"
        )

    limit = matrix.shape[1] if n_trials is None else min(n_trials, matrix.shape[1])
    contents = table[GROUP_COLUMN].to_numpy()
    masks: list[np.ndarray] = []
    for trial in range(limit):
        mask = matrix[positions, trial]
        if len(np.unique(contents[mask])) < 2 or int((~mask).sum()) < MIN_TEST_ROWS:
            continue
        masks.append(mask)

    if not masks:
        raise DatasetError("no usable trial in the release split matrix after applying the scope")
    if len(masks) < limit:
        log.info(
            "using %d of %d release trials; the rest were degenerate after scoping",
            len(masks),
            limit,
        )
    return masks
