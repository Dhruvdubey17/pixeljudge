"""Training tests, including a direct test of the anti-leakage property.

``test_grouped_cv_refuses_to_reward_content_memorisation`` is the one that matters.
It builds a table where the features are pure noise and the label is decided
entirely by which content a row belongs to. A random row split would let a model
learn "content 3 scores 4.5" and post an excellent correlation. With content-grouped
folds the test content is never seen in training, so there is nothing to memorise
and the score has to collapse. That is the difference between a plausible number
and a real one.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pixeljudge.errors import DatasetError
from pixeljudge.model.dataset import load_feature_table
from pixeljudge.model.train import (
    _coerce_param,
    _most_common_params,
    candidate_models,
    fit_final_model,
    identity_fit,
    save_model,
    save_report,
    train_and_evaluate,
)

FIXTURES = Path(__file__).parent / "fixtures"
FEATURES = ["vmaf", "psnr_y", "float_ssim", "float_ms_ssim"]


@pytest.fixture(scope="module")
def features_table() -> pd.DataFrame:
    return load_feature_table(FIXTURES / "features_sample.csv")


@pytest.fixture(scope="module")
def report(features_table: pd.DataFrame):  # type: ignore[no-untyped-def]
    # Three folds keeps the fixture run quick; the pipeline is identical at five.
    return train_and_evaluate(features_table, features=FEATURES, n_splits=3)


def test_every_candidate_and_baseline_is_reported(report) -> None:  # type: ignore[no-untyped-def]
    names = {result.name for result in report.results}
    assert {"svr_rbf", "random_forest", "ridge"} <= names
    assert {f"{metric} alone" for metric in FEATURES} <= names


def test_out_of_fold_predictions_cover_every_row(report) -> None:  # type: ignore[no-untyped-def]
    # Nothing may be left unpredicted: a NaN here means a row was never in a test
    # fold, which would silently shrink the evaluation set.
    prediction_columns = [c for c in report.oof.columns if c.startswith("pred_")]
    assert prediction_columns
    assert report.oof[prediction_columns].notna().all().all()


def test_model_is_at_least_as_good_as_the_weakest_baseline(report) -> None:  # type: ignore[no-untyped-def]
    best = report.result_for(report.best_model)
    worst_baseline = min(
        (r for r in report.results if r.name.endswith(" alone")), key=lambda r: r.srocc
    )
    assert best is not None
    assert best.srocc > worst_baseline.srocc


def test_vmaf_is_the_strongest_single_metric(report) -> None:  # type: ignore[no-untyped-def]
    # The expected, honest ordering: VMAF beats PSNR and SSIM on their own.
    baselines = {r.name: r.srocc for r in report.results if r.name.endswith(" alone")}
    assert baselines["vmaf alone"] > baselines["psnr_y alone"]
    assert baselines["vmaf alone"] > baselines["float_ssim alone"]


def test_beats_vmaf_baseline_is_a_tristate(report) -> None:  # type: ignore[no-untyped-def]
    assert report.beats_vmaf_baseline() in {True, False}
    stripped = report
    stripped.results = [r for r in report.results if r.name != "vmaf alone"]
    # With no VMAF baseline present the answer is "unknown", never a silent win.
    assert stripped.beats_vmaf_baseline() is None


def test_report_serialises_to_json(report, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path = save_report(report, tmp_path / "report.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["n_contents"] == 8
    assert payload["features"] == FEATURES
    assert payload["results"]


def test_grouped_cv_refuses_to_reward_content_memorisation() -> None:
    rng = np.random.default_rng(0)
    n_contents, per_content = 6, 8
    rows: list[dict[str, object]] = []
    for content in range(n_contents):
        # The label depends only on the content, never on the features.
        label = 1.0 + 4.0 * content / (n_contents - 1)
        for _ in range(per_content):
            rows.append(
                {
                    "content": f"c{content}",
                    "vmaf": float(rng.normal(70, 10)),
                    "psnr_y": float(rng.normal(40, 3)),
                    "mos": label,
                }
            )
    frame = pd.DataFrame(rows)

    grouped = train_and_evaluate(frame, features=["vmaf", "psnr_y"], n_splits=3)
    best = grouped.result_for(grouped.best_model)
    assert best is not None
    # Nothing generalisable exists in these features, so a correctly grouped CV
    # cannot produce a strong correlation.
    assert abs(best.srocc) < 0.5, f"grouped CV leaked: SROCC {best.srocc}"


def test_single_content_is_refused() -> None:
    frame = pd.DataFrame(
        {
            "content": ["only"] * 10,
            "vmaf": np.linspace(40, 95, 10),
            "psnr_y": np.linspace(30, 45, 10),
            "mos": np.linspace(1, 5, 10),
        }
    )
    with pytest.raises(DatasetError, match="at least 2 distinct contents"):
        train_and_evaluate(frame, features=["vmaf", "psnr_y"], n_splits=3)


def test_fold_count_is_reduced_to_the_number_of_contents(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rng = np.random.default_rng(1)
    frame = pd.DataFrame(
        {
            "content": ["a"] * 8 + ["b"] * 8 + ["c"] * 8,
            "vmaf": rng.uniform(30, 98, 24),
            "psnr_y": rng.uniform(28, 48, 24),
            "mos": rng.uniform(1, 5, 24),
        }
    )
    with caplog.at_level("WARNING"):
        report = train_and_evaluate(frame, features=["vmaf", "psnr_y"], n_splits=5)
    assert report.n_splits == 3
    assert "reducing cross-validation" in caplog.text


def test_missing_feature_column_is_refused(features_table: pd.DataFrame) -> None:
    with pytest.raises(DatasetError, match="missing required column"):
        train_and_evaluate(features_table, features=["vmaf", "does_not_exist"])


def test_all_rows_missing_is_refused() -> None:
    frame = pd.DataFrame({"content": ["a", "b"], "vmaf": [np.nan, np.nan], "mos": [3.0, 4.0]})
    with pytest.raises(DatasetError, match="no rows left"):
        train_and_evaluate(frame, features=["vmaf"], n_splits=2)


def test_identity_fit_leaves_predictions_alone() -> None:
    values = np.array([1.0, 2.5, 4.0])
    assert np.allclose(identity_fit().apply(values), values)


def test_most_common_params_picks_the_modal_choice() -> None:
    chosen: list[dict[str, object]] = [
        {"model__C": 10.0, "model__gamma": "scale"},
        {"model__C": 10.0, "model__gamma": 0.1},
        {"model__C": 1.0, "model__gamma": "scale"},
    ]
    assert _most_common_params(chosen) == {"model__C": "10.0", "model__gamma": "scale"}


def test_most_common_params_on_empty_input() -> None:
    assert _most_common_params([]) == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("None", None), ("4", 4), ("0.1", 0.1), ("scale", "scale"), (7, 7)],
)
def test_coerce_param_round_trips_grid_values(raw: object, expected: object) -> None:
    assert _coerce_param(raw) == expected


def test_candidate_models_scale_inside_the_pipeline() -> None:
    # The scaler must be a pipeline step, not applied to the table beforehand,
    # or the test fold's statistics leak into the training transform.
    for name, (pipeline, grid) in candidate_models().items():
        assert pipeline.steps[0][0] == "scale", name
        assert grid, name


def test_final_model_refits_and_predicts(features_table: pd.DataFrame) -> None:
    pipeline = fit_final_model(features_table, FEATURES, "mos", "ridge", {"model__alpha": "1.0"})
    predictions = pipeline.predict(features_table[FEATURES].to_numpy(dtype=float))
    assert predictions.shape == (len(features_table),)
    assert np.isfinite(predictions).all()


def test_save_model_writes_a_loadable_artifact(
    features_table: pd.DataFrame, tmp_path: Path
) -> None:
    import joblib

    pipeline = fit_final_model(features_table, FEATURES, "mos", "ridge", {})
    path = save_model(pipeline, tmp_path / "model.joblib")
    reloaded = joblib.load(path)
    assert np.allclose(
        reloaded.predict(features_table[FEATURES].to_numpy(dtype=float)),
        pipeline.predict(features_table[FEATURES].to_numpy(dtype=float)),
    )
