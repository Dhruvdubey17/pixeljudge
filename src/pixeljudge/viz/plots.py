"""Plots and tables for the report.

Deliberately plain matplotlib: no style packages, no seaborn, nothing that changes
what the numbers mean. Every function takes a DataFrame and a destination path,
writes one file, and returns that path, so the ``report`` command is a list of
calls and nothing else.

Conventions that make these figures readable at a glance:

* Bitrate is always on a log x-axis. Rate-distortion curves are roughly
  logarithmic, so linear axes squash the interesting low-bitrate end into nothing.
* Quality is always on the y-axis, higher up meaning better, so "up and to the
  left" is unambiguously the better codec.
* Nothing is invented. If a column is missing, the function says so rather than
  plotting a blank panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

# Non-interactive backend: the report is generated in a terminal and in CI, where
# there is no display to attach to.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from ..errors import DatasetError  # noqa: E402
from ..logging_conf import get_logger  # noqa: E402

log = get_logger(__name__)

FIGSIZE = (9.0, 5.5)
DPI = 140
QUALITY_LABELS = {
    "vmaf": "VMAF",
    "psnr_y": "PSNR-Y (dB)",
    "float_ssim": "SSIM",
    "float_ms_ssim": "MS-SSIM",
}


def _require(frame: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DatasetError(f"cannot plot {what}: missing column(s) {missing}")


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    log.info("wrote %s", path)
    return path


def plot_rd_curves(
    metrics: pd.DataFrame,
    path: Path,
    *,
    quality: str = "vmaf",
    group_column: str = "ladder",
    rate_column: str = "actual_bitrate_kbps",
    title: str | None = None,
) -> Path:
    """Rate-distortion curves, one line per ladder or codec.

    This is the hero figure: it is the whole "why does the same film look different
    on different services" question in one picture. Each line is a delivery recipe,
    and the horizontal distance between two lines at the same height is the bitrate
    one saves over the other for identical quality.
    """
    _require(metrics, [quality, rate_column, group_column], "rate-distortion curves")
    fig, axes = plt.subplots(figsize=FIGSIZE)

    for name, group in metrics.groupby(group_column):
        ordered = group.sort_values(rate_column)
        axes.plot(
            ordered[rate_column],
            ordered[quality],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=str(name),
        )

    axes.set_xscale("log")
    axes.set_xlabel("bitrate (kbps, measured from the encoded file)")
    axes.set_ylabel(QUALITY_LABELS.get(quality, quality))
    axes.set_title(title or f"Rate-distortion: {QUALITY_LABELS.get(quality, quality)} vs bitrate")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(title=group_column, fontsize=8)
    return _save(fig, path)


def plot_predicted_vs_mos(
    oof: pd.DataFrame,
    path: Path,
    *,
    prediction_column: str,
    label_column: str = "mos",
    group_column: str = "content",
    title: str | None = None,
) -> Path:
    """Out-of-fold predictions against human MOS, coloured by source content.

    Colouring by content is not decoration. If one content's cluster sits well off
    the diagonal, the model has a systematic blind spot for that kind of material,
    which a single global correlation number hides completely.
    """
    _require(oof, [prediction_column, label_column], "predicted vs MOS")
    fig, axes = plt.subplots(figsize=(6.4, 6.0))

    if group_column in oof.columns:
        for name, group in oof.groupby(group_column):
            axes.scatter(
                group[label_column], group[prediction_column], s=28, alpha=0.8, label=str(name)
            )
        if oof[group_column].nunique() <= 12:
            axes.legend(title=group_column, fontsize=7, loc="lower right")
    else:
        axes.scatter(oof[label_column], oof[prediction_column], s=28, alpha=0.8)

    low = float(min(oof[label_column].min(), oof[prediction_column].min()))
    high = float(max(oof[label_column].max(), oof[prediction_column].max()))
    margin = 0.05 * (high - low or 1.0)
    axes.plot([low, high], [low, high], "k--", linewidth=1, label="perfect agreement")
    axes.set_xlim(low - margin, high + margin)
    axes.set_ylim(low - margin, high + margin)
    axes.set_xlabel("subjective MOS")
    axes.set_ylabel(f"predicted ({prediction_column.removeprefix('pred_')})")
    axes.set_title(title or "Out-of-fold prediction vs human opinion")
    axes.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_metric_vs_banding(
    merged: pd.DataFrame,
    path: Path,
    *,
    quality: str = "vmaf",
    banding_column: str = "banding_max",
    vmaf_floor: float = 80.0,
    banding_threshold: float | None = None,
    selected: int | None = None,
    title: str | None = None,
) -> Path:
    """The blind-spot figure: quality score against measured banding.

    If the pooled metrics saw banding, this would be a downward-sloping cloud: more
    banding, lower score. The clips that matter are the ones in the top-right
    quadrant, which the metric calls good while the detector says they are visibly
    contoured. The shaded region marks that quadrant explicitly so the claim is not
    left to the reader's eye.

    ``banding_threshold`` should be the same value the blind-spot *table* used, so the
    figure and the table cannot disagree about which clips are being pointed at. It
    defaults to the population's upper quartile, which is that function's own default.

    ``selected`` is the table's own row count. Pass it: the table also requires a PSNR
    floor, which a two-axis figure cannot show, so counting the shaded points here gives a
    slightly larger number than the table lists. Quoting two different counts for the same
    claim is the kind of small inconsistency a reader is right to distrust.
    """
    _require(merged, [quality, banding_column], "metric vs banding")
    fig, axes = plt.subplots(figsize=FIGSIZE)

    axes.scatter(merged[banding_column], merged[quality], s=30, alpha=0.75)
    threshold = (
        float(banding_threshold)
        if banding_threshold is not None
        else float(merged[banding_column].quantile(0.75))
    )

    # Shade the actual rule - both conditions - rather than a vertical band. A reader
    # should be able to see the selected set, not reconstruct it from two guide lines.
    x_high = float(merged[banding_column].max()) * 1.05
    y_high = float(merged[quality].max()) * 1.01
    axes.add_patch(
        Rectangle(
            (threshold, vmaf_floor),
            x_high - threshold,
            y_high - vmaf_floor,
            alpha=0.10,
            color="crimson",
            zorder=0,
        )
    )
    axes.axhline(vmaf_floor, color="crimson", linestyle="--", linewidth=1)
    axes.axvline(threshold, color="crimson", linestyle="--", linewidth=1)
    shaded = int(((merged[banding_column] >= threshold) & (merged[quality] >= vmaf_floor)).sum())
    label = (
        f" the blind-spot table selects {selected} of {len(merged)} encodes"
        if selected is not None
        else f" in this quadrant: {shaded} of {len(merged)} encodes"
    )
    axes.text(
        threshold + (x_high - threshold) * 0.02,
        y_high,
        label,
        color="crimson",
        fontsize=8,
        va="top",
    )

    # State the relationship instead of leaving it to the eye. If the pooled metric
    # tracked introduced banding, this would be strongly negative.
    from scipy.stats import spearmanr

    rho = float(spearmanr(merged[banding_column], merged[quality]).statistic)
    axes.set_xlabel(f"banding the encode added ({banding_column}, higher is worse)")
    axes.set_ylabel(QUALITY_LABELS.get(quality, quality))
    axes.set_title(
        title
        or "Where the objective score and the visible artifact disagree "
        f"(Spearman {rho:+.2f}, n={len(merged)})"
    )
    axes.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_convex_hull(
    points: pd.DataFrame,
    path: Path,
    *,
    hull: pd.DataFrame | None = None,
    rate_column: str = "actual_bitrate_kbps",
    quality: str = "vmaf",
    height_column: str = "height",
    title: str | None = None,
) -> Path:
    """Per-title ladder selection: every probe encode, with the hull picked out.

    One line per resolution. The point of the picture is that the best resolution
    *changes* with bitrate: 360p wins when there are few bits to spend, and is
    beaten by 720p once there are enough. The hull is the envelope of those wins,
    and it is why a per-title ladder is not a fixed table.
    """
    _require(points, [rate_column, quality, height_column], "convex hull")
    fig, axes = plt.subplots(figsize=FIGSIZE)

    for height, group in points.groupby(height_column):
        ordered = group.sort_values(rate_column)
        axes.plot(
            ordered[rate_column],
            ordered[quality],
            marker="o",
            markersize=4,
            alpha=0.55,
            linewidth=1.2,
            label=f"{height}p",
        )

    if hull is not None and not hull.empty:
        ordered = hull.sort_values(rate_column)
        axes.plot(
            ordered[rate_column],
            ordered[quality],
            color="black",
            linewidth=2.4,
            marker="s",
            markersize=7,
            markerfacecolor="none",
            label="convex hull (selected)",
        )

    axes.set_xscale("log")
    axes.set_xlabel("bitrate (kbps)")
    axes.set_ylabel(QUALITY_LABELS.get(quality, quality))
    axes.set_title(title or "Per-title ladder: probe grid and the chosen hull")
    axes.grid(True, which="both", alpha=0.3)
    axes.legend(fontsize=8)
    return _save(fig, path)


def plot_banding_gallery(
    images: list[tuple[Path, str]],
    path: Path,
    *,
    max_images: int = 6,
    title: str | None = None,
) -> Path:
    """A grid of saved frames with their captions.

    Each caption carries the metric scores for that clip, because the figure only
    makes its point if the numbers sit next to the picture they contradict.
    """
    import cv2

    selected = images[:max_images]
    if not selected:
        raise DatasetError("no evidence frames to build a gallery from")

    columns = min(3, len(selected))
    rows = int(np.ceil(len(selected) / columns))
    fig, axes_grid = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.2 * rows))
    axes_list = np.atleast_1d(axes_grid).ravel()

    for axis, (image_path, caption) in zip(axes_list, selected, strict=False):
        image = cv2.imread(str(image_path))
        if image is None:
            axis.text(0.5, 0.5, f"could not read\n{image_path.name}", ha="center", va="center")
        else:
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axis.set_title(caption, fontsize=8)
        axis.axis("off")
    for axis in axes_list[len(selected) :]:
        axis.axis("off")

    fig.suptitle(title or "Banding gallery: worst-scoring frames", fontsize=11)
    return _save(fig, path)


def stretch_for_display(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map the reference's 1st-99th percentile onto 0-255, and apply it to both images.

    Banding lives in low-contrast regions, which is exactly where a screen shows almost
    nothing: the synthetic gradient here spans luma 13 to 43 out of 255, so an unstretched
    crop of it is a black rectangle. Stretching makes the contours visible.

    Two properties keep this honest. The mapping is **linear**, so it cannot create an
    edge that was not there, only make an existing one visible. And it is derived from
    the *reference* and applied identically to both panels, so the two are still directly
    comparable - a stretch computed per panel would flatter or exaggerate one of them.
    """
    low, high = np.percentile(reference.astype(np.float32), (1, 99))
    span = max(float(high - low), 1.0)
    scaled = (image.astype(np.float32) - float(low)) * (255.0 / span)
    stretched: np.ndarray = np.clip(scaled, 0, 255).astype(np.uint8)
    return stretched


def plot_banding_comparison(
    reference_path: Path,
    distorted_path: Path,
    path: Path,
    *,
    caption: str = "",
    reference_caption: str = "master (lossless)",
) -> Path:
    """The project's headline figure: the same frame, master beside encode.

    This is the only form the "metrics missed it" claim can honestly take. A banded
    encode on its own proves nothing, because the master may have been banded too - and
    on stylised animation it often is. Side by side, with the metric scores in the
    caption, the reader can see what the encoder added and what the numbers said about it.
    """
    import cv2

    reference = cv2.imread(str(reference_path))
    distorted = cv2.imread(str(distorted_path))
    if reference is None or distorted is None:
        raise DatasetError(
            f"cannot build the comparison figure: missing {reference_path.name} "
            f"or {distorted_path.name}"
        )
    if distorted.shape != reference.shape:
        # A lower rung was encoded smaller; upscale it the way a player would, which is
        # also how it was measured.
        distorted = cv2.resize(
            distorted, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC
        )

    from ..artifacts.detectors import contour_mask

    rows, columns = worst_region(reference, distorted)
    reference_crop = reference[rows, columns]
    distorted_crop = distorted[rows, columns]

    # The stretch is computed from the reference *crop*, not the whole frame: a bright
    # snowfield or a dark sky occupies a narrow slice of the full-frame range, and a
    # global stretch leaves it almost flat. Still one mapping, still applied to both.
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for axis, image, title in (
        (axes[0], reference_crop, reference_caption),
        (axes[1], distorted_crop, caption or "encode"),
    ):
        axis.imshow(
            cv2.cvtColor(stretch_for_display(image, reference_crop), cv2.COLOR_BGR2RGB),
            interpolation="nearest",
        )
        axis.set_title(title, fontsize=9)
        axis.axis("off")

    # Third panel: what the detector actually flagged. The claim should not depend on the
    # reader's monitor or on trusting a single number.
    added = contour_mask(distorted_crop) & ~contour_mask(reference_crop)
    grey = cv2.cvtColor(
        cv2.cvtColor(stretch_for_display(distorted_crop, reference_crop), cv2.COLOR_BGR2GRAY),
        cv2.COLOR_GRAY2RGB,
    )
    highlighted = grey.copy()
    highlighted[added] = (220, 30, 30)
    axes[2].imshow(highlighted, interpolation="nearest")
    axes[2].set_title(
        f"contours the encode added ({int(added.sum())} px of {added.size})", fontsize=9
    )
    axes[2].axis("off")

    fig.suptitle(
        "Same frame, same crop, one linear stretch applied to both panels. The crop is "
        "centred automatically\non the region where the encode added the most contouring; "
        "the third panel marks those pixels.",
        fontsize=10,
    )
    return _save(fig, path)


def worst_region(
    reference: np.ndarray, distorted: np.ndarray, *, fraction: float = 0.35
) -> tuple[slice, slice]:
    """Crop window centred where the encode added the most contouring.

    A full 720p frame in a figure hides exactly the low-contrast detail the figure is
    about, so a crop is necessary — and choosing one by eye is the kind of judgement call
    that invites "you picked the worst bit". This picks it mechanically: take the
    detector's own contour mask on each frame, subtract, blur to get a local density of
    *added* contours, and centre on the maximum. It is where the detector fired, not
    where it looked worst to me.
    """
    import cv2

    from ..artifacts.detectors import contour_mask

    added = contour_mask(distorted).astype(np.float32) - contour_mask(reference).astype(np.float32)
    height, width = added.shape
    window_h = max(16, int(height * fraction))
    window_w = max(16, int(width * fraction))
    # Odd-sized box filter, so the density at a pixel is the count in a window centred on it.
    density = cv2.blur(added, (window_w | 1, window_h | 1))
    centre_y, centre_x = np.unravel_index(int(np.argmax(density)), density.shape)

    top = int(np.clip(centre_y - window_h // 2, 0, max(height - window_h, 0)))
    left = int(np.clip(centre_x - window_w // 2, 0, max(width - window_w, 0)))
    return slice(top, top + window_h), slice(left, left + window_w)


def plot_per_frame_quality(
    per_frame: pd.DataFrame,
    path: Path,
    *,
    quality: str = "vmaf",
    label: str = "",
) -> Path:
    """Per-frame quality over time, with the pooled values marked.

    Worth having because pooling hides shape: two clips with the same mean VMAF can
    be steady-but-mediocre or excellent-with-a-terrible-stretch, and only one of
    those is acceptable to watch. The harmonic mean sits below the arithmetic mean
    by design, which this makes visible.
    """
    _require(per_frame, [quality], "per-frame quality")
    from ..metrics.vqm import pool_metric

    pooled = pool_metric(per_frame[quality])
    fig, axes = plt.subplots(figsize=FIGSIZE)
    axes.plot(
        per_frame.get("frame", pd.Series(range(len(per_frame)))), per_frame[quality], linewidth=1.2
    )
    axes.axhline(
        pooled["mean"],
        color="tab:green",
        linestyle="--",
        linewidth=1,
        label=f"mean {pooled['mean']:.2f}",
    )
    axes.axhline(
        pooled["harmonic_mean"],
        color="tab:orange",
        linestyle="--",
        linewidth=1,
        label=f"harmonic mean {pooled['harmonic_mean']:.2f}",
    )
    axes.axhline(
        pooled["min"], color="crimson", linestyle=":", linewidth=1, label=f"min {pooled['min']:.2f}"
    )
    axes.set_xlabel("frame")
    axes.set_ylabel(QUALITY_LABELS.get(quality, quality))
    axes.set_title(
        f"Per-frame {QUALITY_LABELS.get(quality, quality)}{f' - {label}' if label else ''}"
    )
    axes.grid(True, alpha=0.3)
    axes.legend(fontsize=8)
    return _save(fig, path)


def save_table(frame: pd.DataFrame, path: Path, *, markdown_too: bool = True) -> Path:
    """Write a table as CSV, and as markdown next to it for pasting into the README."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    log.info("wrote %s", path)
    if markdown_too:
        markdown_path = path.with_suffix(".md")
        markdown_path.write_text(to_markdown(frame), encoding="utf-8")
        log.info("wrote %s", markdown_path)
    return path


def to_markdown(frame: pd.DataFrame) -> str:
    """Render a small table as a markdown pipe table.

    Hand-rolled rather than using ``DataFrame.to_markdown``, which needs the
    optional ``tabulate`` package. Six lines is cheaper than a dependency.
    """
    header = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return "\n".join(lines) + "\n"


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".") if abs(value) < 1e6 else f"{value:.4g}"
    return "" if value is None else str(value)


def plot_correlation_bars(
    folds: pd.DataFrame,
    path: Path,
    *,
    metric: str = "srocc",
    title: str | None = None,
    caption: str | None = None,
) -> Path:
    """Per-predictor correlation with an error bar for the spread across folds.

    The error bars are the point of this figure. A bare table of correlations
    invites reading a 0.02 difference as a result; drawing one standard deviation
    across folds next to each bar shows when two predictors have simply not been
    told apart by a dataset this size. Baselines are drawn in a different shade so
    the "did fusion earn its complexity" comparison is visible at a glance.
    """
    mean_column, std_column = f"{metric}_mean", f"{metric}_std"
    _require(folds, [mean_column, std_column, "name"], "correlation bars")

    ordered = folds.sort_values(mean_column, ascending=True)
    names = [str(name) for name in ordered["name"]]
    is_baseline = [name.endswith(" alone") for name in names]
    colours = ["#c0796a" if baseline else "#3d6d99" for baseline in is_baseline]

    fig, axes = plt.subplots(figsize=(8.0, 0.55 * len(names) + 2.2))
    axes.barh(
        names,
        ordered[mean_column],
        xerr=ordered[std_column],
        color=colours,
        capsize=4,
        error_kw={"ecolor": "#444444", "elinewidth": 1.2},
    )
    for y, (value, spread) in enumerate(
        zip(ordered[mean_column], ordered[std_column], strict=True)
    ):
        # Clamped inside the axes: a predictor with a large spread would otherwise
        # push its own label off the figure, hiding the number that explains why
        # the bar is not to be trusted.
        axes.text(min(value + spread + 0.015, 0.98), y, f"{value:.3f}", va="center", fontsize=8)

    axes.set_xlim(0, 1.05)
    axes.set_xlabel(f"{metric.upper()} (mean across folds, bars show 1 s.d.)")
    axes.set_title(title or f"Agreement with subjective score: {metric.upper()}")
    axes.grid(True, axis="x", alpha=0.3)
    handles = [
        Rectangle((0, 0), 1, 1, color="#3d6d99"),
        Rectangle((0, 0), 1, 1, color="#c0796a"),
    ]
    axes.legend(handles, ["fused model", "single metric"], loc="lower right", fontsize=8)
    if caption:
        fig.text(0.01, 0.01, caption, fontsize=7, color="#555555", wrap=True)
    return _save(fig, path)


def correlation_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Tidy the evaluation rows into the table the README shows."""
    table = pd.DataFrame(results)
    if table.empty:
        return table
    columns = [c for c in ("name", "n", "plcc", "srocc", "krocc", "rmse") if c in table.columns]
    return table[columns].sort_values("srocc", ascending=False).reset_index(drop=True)
