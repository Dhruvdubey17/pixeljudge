"""Training the metric-fusion regressor, with content-grouped cross-validation.

The model itself is unremarkable on purpose: scale the features, fit a regressor,
grid-search its hyperparameters. What makes the result trustworthy is the
evaluation protocol around it.

**Nested, content-grouped cross-validation.** The outer loop is a
:class:`~sklearn.model_selection.GroupKFold` over source content, and it produces
one out-of-fold prediction per clip. Inside each outer training split, another
GroupKFold drives the hyperparameter search. Two consequences:

* No content is ever in training and test at the same time, so the model cannot
  score well by recognising a clip instead of judging its quality.
* Hyperparameters are never chosen using the rows they are then scored on.

**The baselines get the friendlier deal.** Each single-metric baseline is given a
logistic mapping fitted on the fold's training data, which is extra flexibility.
The fused model's predictions are used as-is, because it was trained to output MOS
directly. If the model still wins, it wins against a favoured opponent. If it
does not, that is the honest result and it goes in the README.

SVR is the headline candidate for a specific reason: VMAF itself fuses its
elementary features with an SVM, so an RBF SVR here is the same idea applied one
level up. RandomForest and Ridge are kept as a nonlinear alternative and a
transparent floor.

**Two ways to split.** By default the outer loop is GroupKFold, which is
exhaustive: every clip is held out exactly once. A caller can instead pass explicit
``splits`` - used for the LIVE-Netflix release's 1000 pre-generated 80/20 content
splits, so results can be lined up against the published ones. Those splits sample
rather than partition, so a clip appears in many test sets and there is no single
out-of-fold vector; :func:`train_and_evaluate` detects which case it is from the
splits themselves rather than taking a mode flag on trust.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ..errors import DatasetError
from ..logging_conf import get_logger
from .dataset import GROUP_COLUMN, LABEL_COLUMN, content_groups, describe, require_columns
from .evaluation import (
    EvaluationResult,
    FoldSummary,
    LogisticFit,
    comparison_table,
    evaluate,
    fit_logistic,
    fold_table,
    summarise_folds,
)

log = get_logger(__name__)

DEFAULT_N_SPLITS = 5


def identity_fit() -> LogisticFit:
    """A mapping that leaves predictions alone (``b4=1``, everything else zero)."""
    return LogisticFit(params=(0.0, 1.0, 0.0, 1.0, 0.0), converged=True)


def candidate_models(random_seed: int = 1234) -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    """Each candidate as a (pipeline, grid) pair.

    The scaler lives *inside* the pipeline so it is fitted on training folds only.
    Scaling the whole table before splitting is the classic quiet leak: the test
    rows' mean and variance end up baked into the training transform.
    """
    return {
        "svr_rbf": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [1.0, 10.0, 100.0],
                "model__gamma": ["scale", 0.1, 0.01],
                "model__epsilon": [0.05, 0.1, 0.5],
            },
        ),
        "random_forest": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", RandomForestRegressor(random_state=random_seed, n_jobs=-1)),
                ]
            ),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 2],
            },
        ),
        "ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
    }


@dataclass
class TrainingReport:
    """Everything the report and the model card need from one training run."""

    features: list[str]
    label: str
    n_rows: int
    n_contents: int
    n_splits: int
    results: list[EvaluationResult]
    best_model: str
    best_params: dict[str, Any]
    oof: pd.DataFrame
    dataset_summary: dict[str, Any] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    fold_summaries: list[FoldSummary] = field(default_factory=list)
    # False when the splits sample rather than partition (the release's 80/20
    # trials), in which case the pooled numbers are absent and the fold means are
    # the result.
    pooled_is_valid: bool = True

    @property
    def table(self) -> pd.DataFrame:
        return comparison_table(self.results)

    @property
    def fold_table(self) -> pd.DataFrame:
        return fold_table(self.fold_summaries)

    def result_for(self, name: str) -> EvaluationResult | None:
        return next((result for result in self.results if result.name == name), None)

    def beats_vmaf_baseline(self) -> bool | None:
        """Whether the best model out-ranks VMAF alone on SROCC.

        ``None`` when there was no VMAF column to compare against, so a missing
        baseline is never silently read as a win.
        """
        model = self.result_for(self.best_model)
        baseline = self.result_for("vmaf alone")
        if model is None or baseline is None:
            return None
        return bool(model.srocc >= baseline.srocc)

    def as_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "label": self.label,
            "n_rows": self.n_rows,
            "n_contents": self.n_contents,
            "n_splits": self.n_splits,
            "best_model": self.best_model,
            "best_params": self.best_params,
            "dataset": self.dataset_summary,
            "feature_importances": self.feature_importances,
            "results": [result.as_row() for result in self.results],
            "fold_summaries": [summary.as_row() for summary in self.fold_summaries],
            "pooled_is_valid": self.pooled_is_valid,
            "beats_vmaf_alone": self.beats_vmaf_baseline(),
        }


def _check_groups(groups: pd.Series, n_splits: int) -> int:
    """GroupKFold cannot make more folds than there are groups."""
    n_groups = int(groups.nunique())
    if n_groups < 2:
        raise DatasetError(
            f"content-grouped cross-validation needs at least 2 distinct contents, found "
            f"{n_groups}. With one content there is no way to test on unseen material."
        )
    if n_groups < n_splits:
        log.warning(
            "only %d contents available; reducing cross-validation from %d folds to %d",
            n_groups,
            n_splits,
            n_groups,
        )
        return n_groups
    return n_splits


Split = tuple[np.ndarray, np.ndarray]


def grouped_splits(groups: pd.Series, n_splits: int) -> list[Split]:
    """GroupKFold splits as an explicit list, so every path takes the same shape."""
    indices = np.arange(len(groups))
    values = groups.to_numpy()
    return [
        (train, test) for train, test in GroupKFold(n_splits=n_splits).split(indices, None, values)
    ]


def masks_to_splits(masks: Sequence[np.ndarray]) -> list[Split]:
    """Boolean train masks (as the LIVE-Netflix release ships them) to index pairs."""
    return [(np.flatnonzero(mask), np.flatnonzero(~mask)) for mask in masks]


def _is_partition(splits: Sequence[Split], n_rows: int) -> bool:
    """Whether the test sets tile the data exactly once.

    Decides whether a single out-of-fold vector exists. GroupKFold gives one;
    sampled 80/20 trials do not, because a clip lands in many test sets and
    overwriting its prediction each time would report only the last trial. Checked
    from the splits rather than assumed from a mode name.
    """
    tested = np.concatenate([test for _, test in splits]) if splits else np.array([], dtype=int)
    return tested.size == n_rows and np.array_equal(np.unique(tested), np.arange(n_rows))


def cross_val_predict_grouped(
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    groups: pd.Series,
    name: str,
    pipeline: Pipeline,
    grid: dict[str, list[Any]],
    n_splits: int,
    splits: Sequence[Split] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]], list[tuple[np.ndarray, np.ndarray]]]:
    """Predictions from nested CV, per fold and (where meaningful) pooled.

    Returns the pooled out-of-fold vector, the hyperparameters each fold chose, and
    the ``(prediction, truth)`` pair for every fold. The per-fold pairs are what
    produce the mean +/- std the report quotes; the pooled vector is left as NaN
    where the splits do not cover a row exactly once.
    """
    x = frame[features].to_numpy(dtype=float)
    y = frame[label].to_numpy(dtype=float)
    group_values = groups.to_numpy()
    folds = list(splits) if splits is not None else grouped_splits(groups, n_splits)

    predictions = np.full(y.shape, np.nan)
    chosen: list[dict[str, Any]] = []
    per_fold: list[tuple[np.ndarray, np.ndarray]] = []

    for fold, (train_index, test_index) in enumerate(folds):
        train_groups = group_values[train_index]
        inner_splits = min(n_splits, len(np.unique(train_groups)))
        if inner_splits < 2:
            raise DatasetError(
                "not enough distinct contents inside a training fold to tune "
                "hyperparameters; use fewer outer folds"
            )
        search = GridSearchCV(
            pipeline,
            grid,
            cv=GroupKFold(n_splits=inner_splits),
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            refit=True,
        )
        search.fit(x[train_index], y[train_index], groups=train_groups)
        fold_predictions = search.predict(x[test_index])
        predictions[test_index] = fold_predictions
        per_fold.append((fold_predictions, y[test_index]))
        chosen.append(dict(search.best_params_))
        log.debug("%s fold %d: best params %s", name, fold, search.best_params_)

    return predictions, chosen, per_fold


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    features: list[str],
    label: str = LABEL_COLUMN,
    group_column: str = GROUP_COLUMN,
    n_splits: int = DEFAULT_N_SPLITS,
    random_seed: int = 1234,
    baseline_metrics: list[str] | None = None,
    splits: Sequence[Split] | None = None,
) -> TrainingReport:
    """Fit every candidate under nested grouped CV and compare with the baselines.

    ``splits`` overrides the default GroupKFold with explicit index pairs - used
    for the LIVE-Netflix release's pre-generated trials. When those splits do not
    cover each row exactly once, the headline numbers become the across-fold means
    and ``pooled_is_valid`` is set False.
    """
    require_columns(frame, [*features, label, group_column], source="feature table")
    frame = frame.dropna(subset=[*features, label]).reset_index(drop=True)
    if frame.empty:
        raise DatasetError("feature table has no rows left after dropping missing values")

    groups = content_groups(frame, group_column)
    n_splits = _check_groups(groups, n_splits)
    folds = list(splits) if splits is not None else grouped_splits(groups, n_splits)
    pooled_is_valid = _is_partition(folds, len(frame))
    summary = describe(frame, label_column=label)
    log.info(
        "training on %d clips over %d contents, %d features, %d %s",
        summary["rows"],
        summary["contents"],
        len(features),
        len(folds),
        "folds" if pooled_is_valid else "sampled trials (reporting across-trial means)",
    )

    results: list[EvaluationResult] = []
    fold_summaries: list[FoldSummary] = []
    oof = frame[[group_column, label]].copy()
    best_name, best_srocc, best_params = "", -np.inf, {}

    for name, (pipeline, grid) in candidate_models(random_seed).items():
        predictions, chosen, per_fold = cross_val_predict_grouped(
            frame, features, label, groups, name, pipeline, grid, n_splits, folds
        )
        if pooled_is_valid:
            oof[f"pred_{name}"] = predictions
        # The model already predicts in label units, so no fitted mapping is applied.
        summary_row = summarise_folds(per_fold, name=name)
        fold_summaries.append(summary_row)
        result = (
            evaluate(predictions, frame[label], name=name, fit=identity_fit())
            if pooled_is_valid
            else _from_folds(summary_row, per_fold)
        )
        results.append(result)
        log.info(
            "%-14s PLCC %.4f  SROCC %.4f (+/-%.4f)  KROCC %.4f  RMSE %.4f",
            name,
            result.plcc,
            result.srocc,
            summary_row.srocc_std,
            result.krocc,
            result.rmse,
        )
        if result.srocc > best_srocc:
            best_name, best_srocc = name, result.srocc
            best_params = _most_common_params(chosen)

    baseline_metrics = baseline_metrics if baseline_metrics is not None else list(features)
    for metric in baseline_metrics:
        if metric not in frame.columns:
            continue
        predictions, per_fold = _baseline_oof(frame, metric, label, folds)
        if pooled_is_valid:
            oof[f"pred_{metric}_alone"] = predictions
        name = f"{metric} alone"
        summary_row = summarise_folds(per_fold, name=name)
        fold_summaries.append(summary_row)
        result = (
            evaluate(predictions, frame[label], name=name, fit=identity_fit())
            if pooled_is_valid
            else _from_folds(summary_row, per_fold)
        )
        results.append(result)
        log.info(
            "%-14s PLCC %.4f  SROCC %.4f (+/-%.4f)  KROCC %.4f  RMSE %.4f  (baseline)",
            metric,
            result.plcc,
            result.srocc,
            summary_row.srocc_std,
            result.krocc,
            result.rmse,
        )

    importances = _fit_importances(frame, features, label, random_seed)
    return TrainingReport(
        features=list(features),
        label=label,
        n_rows=int(len(frame)),
        n_contents=int(groups.nunique()),
        n_splits=len(folds),
        results=results,
        best_model=best_name,
        best_params=best_params,
        oof=oof,
        dataset_summary=summary,
        feature_importances=importances,
        fold_summaries=fold_summaries,
        pooled_is_valid=pooled_is_valid,
    )


def _from_folds(
    summary: FoldSummary, per_fold: list[tuple[np.ndarray, np.ndarray]]
) -> EvaluationResult:
    """Present across-fold means in the same shape as a pooled result.

    Used when the splits sample rather than partition, so that the comparison
    table, the choice of best model and the report all keep working off one type
    instead of branching. ``n`` is the total number of scored predictions, which is
    larger than the number of clips because trials overlap.
    """
    return EvaluationResult(
        name=summary.name,
        n=int(sum(len(truth) for _, truth in per_fold)),
        plcc=summary.plcc_mean,
        srocc=summary.srocc_mean,
        krocc=summary.krocc_mean,
        rmse=summary.rmse_mean,
        logistic_converged=True,
    )


def _baseline_oof(
    frame: pd.DataFrame,
    metric: str,
    label: str,
    folds: Sequence[Split],
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Out-of-fold predictions for a single metric, mapped onto the label scale.

    The logistic is fitted on each fold's *training* rows and applied to the held-out
    rows, the same discipline the model gets. Fitting it on all the data would
    quietly hand the baseline information about the test set.
    """
    x = frame[metric].to_numpy(dtype=float)
    y = frame[label].to_numpy(dtype=float)
    predictions = np.full(y.shape, np.nan)
    per_fold: list[tuple[np.ndarray, np.ndarray]] = []

    for train_index, test_index in folds:
        mapping = fit_logistic(x[train_index], y[train_index])
        mapped = mapping.apply(x[test_index])
        predictions[test_index] = mapped
        per_fold.append((mapped, y[test_index]))
    return predictions, per_fold


def _most_common_params(chosen: list[dict[str, Any]]) -> dict[str, Any]:
    """The hyperparameters most folds agreed on.

    Reported rather than "the best fold's", because a single fold's winner on a
    small dataset is mostly noise.
    """
    if not chosen:
        return {}
    counts: dict[str, dict[str, int]] = {}
    for params in chosen:
        for key, value in params.items():
            counts.setdefault(key, {}).setdefault(str(value), 0)
            counts[key][str(value)] += 1
    return {key: max(values, key=lambda k: values[k]) for key, values in counts.items()}


def _fit_importances(
    frame: pd.DataFrame, features: list[str], label: str, random_seed: int
) -> dict[str, float]:
    """Random-forest feature importances, fitted on everything.

    These are descriptive only: they say which features the forest leaned on, not
    how well anything generalises. That is what the cross-validated numbers are for.
    """
    forest = RandomForestRegressor(n_estimators=400, random_state=random_seed, n_jobs=-1)
    forest.fit(frame[features].to_numpy(dtype=float), frame[label].to_numpy(dtype=float))
    return {
        feature: round(float(value), 4)
        for feature, value in sorted(
            zip(features, forest.feature_importances_, strict=True),
            key=lambda pair: pair[1],
            reverse=True,
        )
    }


def fit_final_model(
    frame: pd.DataFrame,
    features: list[str],
    label: str,
    model_name: str,
    params: dict[str, Any],
    random_seed: int = 1234,
) -> Pipeline:
    """Refit the winning candidate on every row, for use as a saved artifact.

    Its honest accuracy is the cross-validated number, not anything measured on
    these rows.
    """
    pipeline, _ = candidate_models(random_seed)[model_name]
    typed = {key: _coerce_param(value) for key, value in params.items()}
    pipeline.set_params(**typed)
    pipeline.fit(frame[features].to_numpy(dtype=float), frame[label].to_numpy(dtype=float))
    return pipeline


def _coerce_param(value: Any) -> Any:
    """Turn the stringified winners from ``_most_common_params`` back into values."""
    if not isinstance(value, str):
        return value
    if value in {"None", "none"}:
        return None
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def save_report(report: TrainingReport, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    log.info("wrote training report %s", path)
    return path


def save_model(pipeline: Pipeline, path: Path) -> Path:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    log.info("wrote model %s", path)
    return path
