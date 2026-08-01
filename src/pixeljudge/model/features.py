"""Collapsing per-frame quality vectors into one feature row per clip.

A subjective score is given for a whole clip, so the per-frame vectors have to be
pooled before anything can be regressed against it. Pooling is not a detail: it is
where most of the information about *how* quality varied over time is thrown away,
and which summary you keep changes the answer.

Two are kept per metric, both from :func:`~pixeljudge.metrics.vqm.pool_metric` so
they are computed by exactly the same code that pools PixelJudge's own libvmaf
output - which is what makes the Stage 3 comparison a like-for-like one:

``mean``
    The plain average. What the LIVE-Netflix release's own scripts use for their
    headline results, so keeping it is what makes the numbers here comparable to
    the published ones.
``hmean``
    libvmaf's harmonic mean, with its ``+1`` offset to survive zero-valued frames.
    It sits below the arithmetic mean and is dragged down by the worst frames,
    which is closer to how viewers weight a clip: a few seconds of visible damage
    is remembered more than a lot of seconds of adequacy.

Both are offered to every model rather than one being chosen up front. Which pooling
carries the signal is a question the cross-validation can answer; guessing would
just be a guess with fewer numbers attached.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..errors import DatasetError
from ..logging_conf import get_logger
from ..metrics.vqm import pool_metric
from .livenflx import GROUP_COLUMN, LABEL_COLUMN

log = get_logger(__name__)

# Pooling name in the output column -> the key pool_metric returns.
POOLINGS: dict[str, str] = {"mean": "mean", "hmean": "harmonic_mean"}

# Columns that describe the video rather than its quality, carried through so the
# feature table stays self-describing (and so grouping and scoping still work on
# it without a join back to the loader's output).
IDENTITY_COLUMNS: tuple[str, ...] = (
    "video_id",
    GROUP_COLUMN,
    "content_index",
    "condition",
    "condition_index",
    LABEL_COLUMN,
    "n_stalls",
    "stall_seconds",
    "low_bitrate_seconds",
    "fps",
)


def feature_name(metric: str, pooling: str) -> str:
    """``('vmaf', 'hmean') -> 'vmaf_hmean'``, in one place so nothing drifts."""
    return f"{metric}_{pooling}"


def pool_to_features(
    long_table: pd.DataFrame,
    *,
    poolings: Sequence[str] = tuple(POOLINGS),
) -> pd.DataFrame:
    """Long per-frame table in, one row per video out.

    Emits ``{metric}_{pooling}`` for every metric present, alongside the identity
    columns. Videos are sorted by content then condition so the row order is
    reproducible - which matters because the release's split matrix is positional.
    """
    required = {"video_id", "metric", "frame", "value", GROUP_COLUMN, LABEL_COLUMN}
    missing = required - set(long_table.columns)
    if missing:
        raise DatasetError(f"long table is missing {sorted(missing)}")

    unknown = [name for name in poolings if name not in POOLINGS]
    if unknown:
        raise DatasetError(f"unknown pooling(s) {unknown}; available: {sorted(POOLINGS)}")

    identity = [column for column in IDENTITY_COLUMNS if column in long_table.columns]
    rows: list[dict[str, object]] = []

    for video_id, video in long_table.groupby("video_id", sort=False):
        # Identity fields are constant within a video by construction; take the
        # first row's, but check rather than assume - a silently varying label
        # would mean the frames of two videos had been merged.
        head = video.iloc[0]
        for column in (GROUP_COLUMN, LABEL_COLUMN):
            if video[column].nunique() != 1:
                raise DatasetError(f"{video_id}: {column} is not constant across its rows")

        row: dict[str, object] = {column: head[column] for column in identity}
        for metric, group in video.groupby("metric", sort=False):
            pooled = pool_metric(group.sort_values("frame")["value"])
            for pooling in poolings:
                row[feature_name(str(metric), pooling)] = pooled[POOLINGS[pooling]]
        rows.append(row)

    table = pd.DataFrame(rows)
    if GROUP_COLUMN in table.columns and "condition_index" in table.columns:
        table = table.sort_values([GROUP_COLUMN, "condition_index"]).reset_index(drop=True)

    feature_columns = sorted(set(table.columns) - set(identity))
    log.info(
        "pooled %d videos over %d contents into %d features (%s)",
        len(table),
        table[GROUP_COLUMN].nunique(),
        len(feature_columns),
        ", ".join(poolings),
    )
    return table


def feature_columns(
    table: pd.DataFrame, *, metrics: Sequence[str], poolings: Sequence[str]
) -> list[str]:
    """The ``{metric}_{pooling}`` columns that actually exist in *table*.

    Requested combinations that are absent are reported rather than skipped: a
    typo in a config's metric list would otherwise quietly train a smaller model
    than the one the config describes.
    """
    wanted = [feature_name(metric, pooling) for metric in metrics for pooling in poolings]
    missing = [column for column in wanted if column not in table.columns]
    if missing:
        raise DatasetError(
            f"feature table is missing {missing}; present: "
            f"{sorted(c for c in table.columns if '_' in c)}"
        )
    return wanted


def compare_feature_tables(
    theirs: pd.DataFrame,
    ours: pd.DataFrame,
    *,
    on: str = "video_id",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Agreement between two pooled feature tables for the same clips.

    The independent-validation check: the LIVE-Netflix release's own quality
    vectors on one side, PixelJudge's libvmaf measurements of the same videos on
    the other. Run *before* re-training, because if the two disagree there is
    nothing to learn from re-running a regression on the second one.

    Reports rank agreement (Spearman - the thing that actually matters, since a
    constant offset between two VMAF implementations would not change any
    conclusion), linear agreement, and the raw differences, which are what
    distinguish "a different VMAF model version" from "the videos are misaligned".

    Expect Spearman above about 0.95 on VMAF. A materially lower number is a real
    discrepancy - most likely a VMAF model-version or pooling mismatch, or a
    scaling/timestamp problem - and finding it is this check doing its job.
    """
    from scipy.stats import pearsonr, spearmanr

    for frame, label in ((theirs, "theirs"), (ours, "ours")):
        if on not in frame.columns:
            raise DatasetError(f"{label} table has no {on!r} column to join on")

    merged = theirs.merge(ours, on=on, suffixes=("_theirs", "_ours"))
    if merged.empty:
        raise DatasetError(
            f"no clip in common between the two tables on {on!r}; "
            "check that the video identifiers match"
        )
    if len(merged) < len(theirs):
        log.warning(
            "only %d of %d clips could be matched; comparing the overlap",
            len(merged),
            len(theirs),
        )

    candidates = (
        list(columns)
        if columns is not None
        else sorted(
            column
            for column in theirs.columns
            if f"{column}_theirs" in merged.columns and f"{column}_ours" in merged.columns
        )
    )

    rows: list[dict[str, object]] = []
    for column in candidates:
        left = pd.to_numeric(merged[f"{column}_theirs"], errors="coerce")
        right = pd.to_numeric(merged[f"{column}_ours"], errors="coerce")
        usable = left.notna() & right.notna()
        if int(usable.sum()) < 3:
            log.warning("%s: fewer than three comparable values, skipped", column)
            continue
        a, b = left[usable].to_numpy(float), right[usable].to_numpy(float)
        difference = b - a
        rows.append(
            {
                "feature": column,
                "n": int(usable.sum()),
                "spearman": round(float(spearmanr(a, b).statistic), 4),
                "pearson": round(float(pearsonr(a, b).statistic), 4),
                "mean_difference": round(float(difference.mean()), 4),
                "mean_abs_difference": round(float(np.abs(difference).mean()), 4),
                "max_abs_difference": round(float(np.abs(difference).max()), 4),
            }
        )

    if not rows:
        raise DatasetError("no feature column was comparable between the two tables")
    return pd.DataFrame(rows)
