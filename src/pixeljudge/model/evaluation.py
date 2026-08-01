"""Evaluating a quality predictor the way the video-quality literature does.

The convention matters here, because the obvious approach gives the wrong answer.
If you correlate raw VMAF (0-100) against MOS (1-5) with Pearson, you measure two
things at once: how well VMAF *ranks* clips, and how badly its scale differs from
the MOS scale. The second part is not a modelling error, it is a units mismatch.

So VQEG's procedure, which this module implements:

1. Fit a monotonic 5-parameter logistic that maps predictions onto the MOS scale.
   This absorbs the harmless nonlinearity (metrics saturate at both ends,
   opinion scores do not).
2. Then measure:
   * **PLCC** (Pearson, after fitting) - accuracy: how close are the values.
   * **SROCC** (Spearman) - monotonicity: does higher predicted mean higher MOS.
     Rank-based, so the logistic mapping cannot change it. Computed on raw
     predictions for exactly that reason.
   * **KROCC** (Kendall) - a stricter ordinal agreement, based on counting
     concordant and discordant pairs.
   * **RMSE** (after fitting) - the typical size of the error, in MOS units.

Higher is better for the three correlations, lower for RMSE.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from scipy.stats import kendalltau, pearsonr, spearmanr

from ..errors import DatasetError
from ..logging_conf import get_logger

log = get_logger(__name__)

# Parameter vector for the logistic below, in order.
LOGISTIC_PARAM_NAMES = ("b1", "b2", "b3", "b4", "b5")


def logistic5(x: np.ndarray, b1: float, b2: float, b3: float, b4: float, b5: float) -> np.ndarray:
    """The VQEG 5-parameter logistic.

    ``b1 * (0.5 - 1/(1 + exp(b2 * (x - b3)))) + b4 * x + b5``

    The first term is a sigmoid: it handles the saturation at both ends of a
    metric's range. The ``b4 * x`` term adds a straight-line component so the fit
    can stay accurate through the middle of the range, where a pure sigmoid would
    be too steep or too flat. ``b5`` is the offset onto the MOS scale.

    ``b2`` carries the direction: a negative ``b2`` fits a metric where lower means
    better, so the same function serves PSNR, VMAF and anything else.
    """
    x = np.asarray(x, dtype=float)
    # Clip the exponent before exp() to avoid an overflow warning during fitting;
    # exp(700) is already beyond float64's range, so the result is unchanged.
    exponent = np.clip(b2 * (x - b3), -700.0, 700.0)
    return b1 * (0.5 - 1.0 / (1.0 + np.exp(exponent))) + b4 * x + b5


@dataclass(frozen=True)
class LogisticFit:
    """A fitted mapping from prediction space onto the MOS scale."""

    params: tuple[float, float, float, float, float]
    converged: bool

    def apply(self, x: np.ndarray | pd.Series) -> np.ndarray:
        return logistic5(np.asarray(x, dtype=float), *self.params)

    def as_dict(self) -> dict[str, float | bool]:
        mapping: dict[str, float | bool] = dict(zip(LOGISTIC_PARAM_NAMES, self.params, strict=True))
        mapping["converged"] = self.converged
        return mapping


@dataclass(frozen=True)
class EvaluationResult:
    """The four numbers, plus enough context to know what they describe."""

    name: str
    n: int
    plcc: float
    srocc: float
    krocc: float
    rmse: float
    logistic_converged: bool

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def fit_logistic(predictions: np.ndarray | pd.Series, mos: np.ndarray | pd.Series) -> LogisticFit:
    """Fit the 5-parameter logistic from predictions to MOS.

    **The fit is done on standardised predictions.** This is not cosmetic. The
    metrics being mapped live on wildly different scales - SSIM spans about 0.05,
    PSNR about 4, VMAF about 18 - and ``curve_fit`` is a local optimiser working on
    all five parameters at once. On raw units the problem is badly conditioned:
    ``b2`` has to come out around ``1/spread`` while ``b3`` sits near the mean, so
    the two differ by orders of magnitude and the solver either stalls or wanders
    until it exhausts ``maxfev``. Centring and scaling ``x`` first puts every
    parameter on a comparable footing.

    The fitted parameters are then converted back to raw-``x`` units algebraically,
    so a :class:`LogisticFit` still means exactly what it says - five parameters
    applied directly to unscaled predictions - and nothing downstream needs to know
    the standardisation happened.

    Initial values are chosen from the data rather than hardcoded:

    * ``b1`` starts at the MOS range (the sigmoid's height),
    * ``b2`` at a unit slope, signed by the observed correlation so the curve starts
      pointing the right way,
    * ``b3`` at the (standardised) mean prediction, i.e. zero,
    * ``b4`` at zero and ``b5`` at the mean MOS.

    A failed fit is reported, not hidden: we fall back to a least-squares straight
    line so the caller still gets numbers, and ``converged=False`` says the
    nonlinear fit did not take. Silently returning a bad mapping would corrupt
    PLCC and RMSE for every model in the comparison.
    """
    x = np.asarray(predictions, dtype=float)
    y = np.asarray(mos, dtype=float)
    if x.size != y.size:
        raise DatasetError(f"prediction/label length mismatch: {x.size} vs {y.size}")
    if x.size < len(LOGISTIC_PARAM_NAMES):
        raise DatasetError(
            f"need at least {len(LOGISTIC_PARAM_NAMES)} points to fit a 5-parameter "
            f"logistic, got {x.size}"
        )

    centre = float(x.mean())
    scale = float(x.std()) or 1.0
    z = (x - centre) / scale

    direction = 1.0
    if x.size > 2 and np.std(x) > 0 and np.std(y) > 0:
        direction = math.copysign(1.0, float(np.corrcoef(x, y)[0, 1]) or 1.0)
    guess = [float(y.max() - y.min()) or 1.0, direction, 0.0, 0.0, float(y.mean())]

    import warnings

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", OptimizeWarning)
            params, _ = curve_fit(logistic5, z, y, p0=guess, maxfev=20000)
    except (RuntimeError, OptimizeWarning, ValueError) as exc:
        log.warning("logistic fit did not converge (%s); falling back to a linear map", exc)
        slope, intercept = np.polyfit(x, y, 1) if np.std(x) > 0 else (0.0, float(y.mean()))
        return LogisticFit(params=(0.0, 1.0, 0.0, float(slope), float(intercept)), converged=False)

    return LogisticFit(params=_unstandardise(params, centre, scale), converged=True)


def _unstandardise(
    params: np.ndarray, centre: float, scale: float
) -> tuple[float, float, float, float, float]:
    """Rewrite parameters fitted on ``(x - centre) / scale`` to act on raw ``x``.

    Substituting ``z = (x - centre) / scale`` into the logistic and collecting terms
    gives an exact re-parameterisation, so the mapping is unchanged - only its
    coordinates are:

    * the sigmoid's slope divides by the scale, ``b2 -> b2 / scale``
    * its centre moves back into raw units, ``b3 -> centre + b3 * scale``
    * the linear term divides by the scale, ``b4 -> b4 / scale``
    * and the intercept absorbs the shift, ``b5 -> b5 - b4 * centre / scale``

    ``b1`` is a height in MOS units and is untouched.
    """
    b1, b2, b3, b4, b5 = (float(p) for p in params)
    return (b1, b2 / scale, centre + b3 * scale, b4 / scale, b5 - b4 * centre / scale)


def evaluate(
    predictions: np.ndarray | pd.Series,
    mos: np.ndarray | pd.Series,
    *,
    name: str = "model",
    fit: LogisticFit | None = None,
) -> EvaluationResult:
    """PLCC/SROCC/KROCC/RMSE for one predictor.

    Pass ``fit`` to reuse a mapping fitted on training data. Fitting the logistic
    on the same rows you then score is mildly optimistic; for the model comparison
    we fit per fold on the training split, which is why this is a parameter and
    not always computed here.
    """
    x = np.asarray(predictions, dtype=float)
    y = np.asarray(mos, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        raise DatasetError(f"{name}: fewer than three usable prediction/label pairs")
    if finite.sum() != x.size:
        log.warning("%s: dropped %d non-finite pair(s)", name, int(x.size - finite.sum()))
    x, y = x[finite], y[finite]

    mapping = fit or fit_logistic(x, y)
    mapped = mapping.apply(x)

    # Pearson needs variance in both arrays; a constant predictor has none.
    plcc = float(pearsonr(mapped, y).statistic) if np.std(mapped) > 0 else float("nan")
    srocc = float(spearmanr(x, y).statistic) if np.std(x) > 0 else float("nan")
    krocc = float(kendalltau(x, y).statistic) if np.std(x) > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((mapped - y) ** 2)))

    return EvaluationResult(
        name=name,
        n=int(x.size),
        plcc=round(plcc, 4),
        srocc=round(srocc, 4),
        krocc=round(krocc, 4),
        rmse=round(rmse, 4),
        logistic_converged=mapping.converged,
    )


def evaluate_single_metric_baselines(
    frame: pd.DataFrame,
    metric_columns: list[str],
    label_column: str = "mos",
) -> pd.DataFrame:
    """Score each metric on its own, each with its own logistic fit.

    This is the comparison that keeps the project honest. A fused model that cannot
    beat VMAF alone has not earned its complexity, and the only way to know is to
    give every single metric the same treatment (own fit, same rows, same numbers).
    """
    missing = [column for column in [*metric_columns, label_column] if column not in frame.columns]
    if missing:
        raise DatasetError(f"feature table is missing columns {missing}")

    rows = [
        evaluate(frame[column], frame[label_column], name=f"{column} alone").as_row()
        for column in metric_columns
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FoldSummary:
    """One predictor's scores aggregated over folds, as mean and spread.

    Reported alongside the pooled numbers because a single correlation over all
    held-out clips hides how much it moved from fold to fold. Two models whose
    SROCC differs by less than the standard deviation across folds have not been
    told apart by this dataset, and saying so is more useful than ranking them.
    """

    name: str
    n_folds: int
    plcc_mean: float
    plcc_std: float
    srocc_mean: float
    srocc_std: float
    krocc_mean: float
    krocc_std: float
    rmse_mean: float
    rmse_std: float

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


def summarise_folds(
    folds: Sequence[tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]],
    *,
    name: str = "model",
    fit_per_fold: bool = False,
) -> FoldSummary:
    """Score each fold separately, then average.

    ``fit_per_fold`` fits the logistic inside each fold, which is what a
    single-metric baseline needs (its raw units are not MOS). A model that already
    predicts in label units is scored as-is.

    Folds too small to score are skipped rather than allowed to abort the run: with
    the release's 80/20 splits a fold can come out tiny after scoping, and losing
    one trial is better than losing the comparison.
    """
    rows: list[EvaluationResult] = []
    skipped = 0
    for predictions, truth in folds:
        try:
            fit = None if fit_per_fold else LogisticFit((0.0, 1.0, 0.0, 1.0, 0.0), converged=True)
            rows.append(evaluate(predictions, truth, name=name, fit=fit))
        except DatasetError:
            skipped += 1
    if not rows:
        raise DatasetError(f"{name}: no fold had enough usable points to score")
    if skipped:
        log.warning("%s: skipped %d fold(s) with too few usable points", name, skipped)

    def stats(attribute: str) -> tuple[float, float]:
        values = np.array([getattr(row, attribute) for row in rows], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return float("nan"), float("nan")
        return round(float(values.mean()), 4), round(float(values.std(ddof=0)), 4)

    plcc, plcc_std = stats("plcc")
    srocc, srocc_std = stats("srocc")
    krocc, krocc_std = stats("krocc")
    rmse, rmse_std = stats("rmse")
    return FoldSummary(
        name=name,
        n_folds=len(rows),
        plcc_mean=plcc,
        plcc_std=plcc_std,
        srocc_mean=srocc,
        srocc_std=srocc_std,
        krocc_mean=krocc,
        krocc_std=krocc_std,
        rmse_mean=rmse,
        rmse_std=rmse_std,
    )


def fold_table(summaries: Sequence[FoldSummary]) -> pd.DataFrame:
    """Fold summaries as a table, best mean SROCC first."""
    table = pd.DataFrame([summary.as_row() for summary in summaries])
    if table.empty:
        return table
    return table.sort_values("srocc_mean", ascending=False).reset_index(drop=True)


def comparison_table(results: Sequence[EvaluationResult]) -> pd.DataFrame:
    """Rank predictors by SROCC, the metric least sensitive to scale."""
    table = pd.DataFrame([result.as_row() for result in results])
    return table.sort_values("srocc", ascending=False).reset_index(drop=True)
