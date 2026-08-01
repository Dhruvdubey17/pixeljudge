"""BD-Rate and BD-quality: turning two rate-distortion curves into one number.

The question "is codec B better than codec A" has no single-point answer, because
the two encoders are never asked for exactly the same bitrate. Bjontegaard's
answer: compare the *areas* between the curves.

BD-Rate reads "on average across the overlapping quality range, codec B needs
X% fewer bits than codec A for the same quality". Negative is an improvement.
BD-quality is the same comparison rotated: "how many more VMAF points (or dB) at
matched bitrate".

Implementation notes that matter for correctness:

* Work in ``log10(bitrate)``. Rate-distortion behaviour is roughly logarithmic, so
  a straight line in log space is a good local model and the interpolation error
  is far smaller than in linear space.
* Interpolate with **PCHIP** rather than the single cubic polynomial of the
  original 2001 formulation. A cubic through four points overshoots badly when the
  points are not evenly spaced, and can invent a curve that dips below both
  neighbours. PCHIP is piecewise, shape-preserving and monotone on monotone data.
* Only integrate over the **overlapping** quality (or bitrate) range. Extrapolating
  past the measured points is where BD-Rate results become fiction.
* Prefer **VMAF or PSNR** as the quality axis. BD numbers computed on SSIM are
  known to be unstable, because SSIM saturates near 1 and the top of the curve
  flattens into numerical noise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from ..errors import MetricsError
from ..logging_conf import get_logger

log = get_logger(__name__)

# Four points is the arithmetic minimum for a stable piecewise fit; five is what
# the literature actually recommends, so fewer than five earns a warning.
MIN_POINTS = 4
RECOMMENDED_POINTS = 5


@dataclass(frozen=True)
class RdCurve:
    """One codec's measured rate-distortion curve."""

    name: str
    bitrates_kbps: tuple[float, ...]
    quality: tuple[float, ...]
    metric: str = "vmaf"

    def __post_init__(self) -> None:
        if len(self.bitrates_kbps) != len(self.quality):
            raise MetricsError(
                f"curve {self.name!r}: {len(self.bitrates_kbps)} bitrates but "
                f"{len(self.quality)} quality values"
            )
        if len(self.bitrates_kbps) < MIN_POINTS:
            raise MetricsError(
                f"curve {self.name!r} has {len(self.bitrates_kbps)} points; BD metrics "
                f"need at least {MIN_POINTS} (>= {RECOMMENDED_POINTS} recommended)"
            )
        if any(rate <= 0 for rate in self.bitrates_kbps):
            raise MetricsError(f"curve {self.name!r} has a non-positive bitrate")
        if any(not math.isfinite(value) for value in self.quality):
            raise MetricsError(f"curve {self.name!r} has a non-finite quality value")
        if len(self.bitrates_kbps) < RECOMMENDED_POINTS:
            log.warning(
                "curve %r has only %d points; BD-Rate is noisy below %d",
                self.name,
                len(self.bitrates_kbps),
                RECOMMENDED_POINTS,
            )
        self._warn_if_not_monotonic()

    def _warn_if_not_monotonic(self) -> None:
        """Flag a curve where more bitrate bought less quality.

        Sorting by quality later hides this: the dip is silently reordered and the
        interpolation still succeeds, so without an explicit check a broken sweep
        produces a confident BD-Rate. A dip usually means the encoder changed a
        decision (a different GOP structure, or a rate cap kicking in), and the
        sweep is worth re-running before the number is quoted.
        """
        pairs = sorted(zip(self.bitrates_kbps, self.quality, strict=True))
        dips = [
            (low[0], high[0])
            for low, high in zip(pairs, pairs[1:], strict=False)
            if high[1] < low[1]
        ]
        if dips:
            log.warning(
                "curve %r is non-monotonic: quality drops as bitrate rises between %s. "
                "BD numbers are still computed, but check the sweep.",
                self.name,
                ", ".join(f"{a:.0f}->{b:.0f} kbps" for a, b in dips),
            )

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        name: str,
        *,
        rate_column: str = "actual_bitrate_kbps",
        quality_column: str = "vmaf",
    ) -> RdCurve:
        """Build a curve from a measured metrics table."""
        missing = {rate_column, quality_column} - set(frame.columns)
        if missing:
            raise MetricsError(f"curve {name!r}: metrics table is missing {sorted(missing)}")
        ordered = frame.sort_values(rate_column)
        return cls(
            name=name,
            bitrates_kbps=tuple(float(v) for v in ordered[rate_column]),
            quality=tuple(float(v) for v in ordered[quality_column]),
            metric=quality_column,
        )

    @property
    def log_rate(self) -> np.ndarray:
        return np.log10(np.asarray(self.bitrates_kbps, dtype=float))

    @property
    def quality_array(self) -> np.ndarray:
        return np.asarray(self.quality, dtype=float)


def _monotone_pairs(x: np.ndarray, y: np.ndarray, label: str) -> tuple[np.ndarray, np.ndarray]:
    """Sort by ``x`` and keep only strictly increasing ``x``.

    PCHIP needs strictly increasing sample points. Real measurements sometimes
    repeat or invert (two CRF values that produced the same bitrate, or a VMAF
    that dipped as bitrate rose because the encoder made a different decision).
    Dropping the offending point is better than crashing, but it is worth a log
    line because a badly non-monotonic curve means the sweep needs re-running.
    """
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    keep = np.ones(x.size, dtype=bool)
    last = -np.inf
    for index, value in enumerate(x):
        if value <= last:
            keep[index] = False
        else:
            last = value
    dropped = int((~keep).sum())
    if dropped:
        log.warning("%s: dropped %d non-monotonic point(s) before interpolation", label, dropped)
    if keep.sum() < 2:
        raise MetricsError(f"{label}: fewer than two usable points after deduplication")
    return x[keep], y[keep]


def _overlap(a: np.ndarray, b: np.ndarray, label: str) -> tuple[float, float]:
    low = max(float(a.min()), float(b.min()))
    high = min(float(a.max()), float(b.max()))
    if high <= low:
        raise MetricsError(
            f"{label}: the two curves do not overlap "
            f"([{a.min():.3f}, {a.max():.3f}] vs [{b.min():.3f}, {b.max():.3f}]). "
            "Re-run one sweep with quality targets that reach into the other's range."
        )
    return low, high


def _average_over(x: np.ndarray, y: np.ndarray, low: float, high: float) -> float:
    """Mean of the PCHIP fit of ``y(x)`` over ``[low, high]``."""
    interpolator = PchipInterpolator(x, y, extrapolate=False)
    return float(interpolator.integrate(low, high) / (high - low))


def bd_rate(anchor: RdCurve, test: RdCurve) -> float:
    """Average bitrate difference of ``test`` versus ``anchor`` at equal quality, in %.

    Negative means ``test`` needs fewer bits, i.e. it is the more efficient codec.
    """
    _check_comparable(anchor, test)
    # Quality is the independent variable here: we ask "what did each codec have
    # to pay for this quality level".
    ax, ay = _monotone_pairs(anchor.quality_array, anchor.log_rate, f"{anchor.name} (BD-Rate)")
    bx, by = _monotone_pairs(test.quality_array, test.log_rate, f"{test.name} (BD-Rate)")
    low, high = _overlap(ax, bx, f"BD-Rate {test.name} vs {anchor.name} on {anchor.metric}")

    anchor_log_rate = _average_over(ax, ay, low, high)
    test_log_rate = _average_over(bx, by, low, high)
    # Back out of log space: a mean difference of d in log10 is a factor 10**d.
    return float((10.0 ** (test_log_rate - anchor_log_rate) - 1.0) * 100.0)


def bd_quality(anchor: RdCurve, test: RdCurve) -> float:
    """Average quality difference of ``test`` versus ``anchor`` at equal bitrate.

    Positive means ``test`` looks better for the same bits. Units are the metric's
    own (VMAF points, or dB for PSNR).
    """
    _check_comparable(anchor, test)
    ax, ay = _monotone_pairs(anchor.log_rate, anchor.quality_array, f"{anchor.name} (BD-quality)")
    bx, by = _monotone_pairs(test.log_rate, test.quality_array, f"{test.name} (BD-quality)")
    low, high = _overlap(ax, bx, f"BD-quality {test.name} vs {anchor.name} on {anchor.metric}")
    return float(_average_over(bx, by, low, high) - _average_over(ax, ay, low, high))


def _check_comparable(anchor: RdCurve, test: RdCurve) -> None:
    if anchor.metric != test.metric:
        raise MetricsError(
            f"cannot compare curves measured with different metrics: "
            f"{anchor.name} uses {anchor.metric}, {test.name} uses {test.metric}"
        )
    if anchor.metric.startswith("float_ssim") or anchor.metric.startswith("float_ms_ssim"):
        log.warning(
            "BD metrics on %s are unstable because SSIM saturates near 1; " "prefer vmaf or psnr_y",
            anchor.metric,
        )


def bd_table(curves: dict[str, RdCurve], anchor: str) -> pd.DataFrame:
    """BD-Rate and BD-quality for every curve against one anchor codec.

    A pair that cannot legitimately be compared is **skipped with a warning** rather
    than aborting the table. The primitives above stay strict on purpose - a
    non-overlapping BD number is fiction and should raise - but a report covering four
    codecs and four clips should tell you the fifteen comparisons it could make and
    name the one it could not. This is not hypothetical: on near-flat synthetic
    content, x264 spends 10-31 kbit/s while SVT-AV1's floor puts it at 60-200, so
    their bitrate ranges are disjoint and BD-quality has nothing to integrate over.
    """
    if anchor not in curves:
        raise MetricsError(f"anchor curve {anchor!r} not among {sorted(curves)}")
    reference = curves[anchor]
    rows: list[dict[str, object]] = []
    for name, curve in curves.items():
        if name == anchor:
            continue
        row: dict[str, object] = {
            "codec": name,
            "anchor": anchor,
            "metric": curve.metric,
            "n_points": len(curve.bitrates_kbps),
        }
        for column, compute in (("bd_rate_pct", bd_rate), ("bd_quality", bd_quality)):
            try:
                row[column] = round(compute(reference, curve), 3)
            except MetricsError as exc:
                log.warning("skipping %s for %s vs %s: %s", column, name, anchor, exc)
                # NaN rather than None, so the column is float64 even when *every*
                # comparison for this source failed. A column of all-None infers as
                # object, and concatenating that with a float64 column from another
                # source makes pandas fall back to guessing the result dtype - which
                # is what the "concatenation with empty or all-NA entries" warning is
                # about. The values are identical either way; only the dtype differs.
                # This is not hypothetical: on the synthetic gradient clip all three
                # BD-quality comparisons are skipped for non-overlap.
                row[column] = float("nan")
        rows.append(row)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    # Comparisons that produced nothing sort last rather than vanishing, so a reader
    # sees that they were attempted.
    return table.sort_values("bd_rate_pct", na_position="last").reset_index(drop=True)
