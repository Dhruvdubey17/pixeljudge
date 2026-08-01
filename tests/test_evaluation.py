"""Tests for the VQEG evaluation maths.

The interesting cases are the ones that justify the procedure existing at all: a
metric that ranks perfectly but lives on the wrong scale, and a metric that runs
in the opposite direction. Both should come out with a near-perfect PLCC *after*
the logistic fit, and that is the whole argument for fitting it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from pixeljudge.errors import DatasetError
from pixeljudge.model.evaluation import (
    LogisticFit,
    comparison_table,
    evaluate,
    evaluate_single_metric_baselines,
    fit_logistic,
    logistic5,
)

TRUE_PARAMS = (4.0, 0.12, 65.0, 0.004, 2.6)


def synthetic_metric_and_mos(n: int = 40, noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A metric on a 0-100 scale and MOS on 1-5, related by a known logistic."""
    rng = np.random.default_rng(7)
    x = np.linspace(10, 100, n)
    y = logistic5(x, *TRUE_PARAMS)
    if noise:
        y = y + rng.normal(0, noise, n)
    return x, y


def test_logistic_is_monotonic_for_positive_slope() -> None:
    x = np.linspace(0, 100, 200)
    y = logistic5(x, *TRUE_PARAMS)
    assert np.all(np.diff(y) > 0)


def test_logistic_reverses_for_negative_slope() -> None:
    x = np.linspace(0, 100, 200)
    y = logistic5(x, 4.0, -0.12, 65.0, -0.004, 2.6)
    assert np.all(np.diff(y) < 0)


def test_logistic_survives_extreme_inputs() -> None:
    # The exponent is clipped, so a huge argument must not overflow into inf/nan.
    y = logistic5(np.array([-1e6, 0.0, 1e6]), *TRUE_PARAMS)
    assert np.isfinite(y).all()


def test_fit_recovers_a_known_mapping() -> None:
    x, y = synthetic_metric_and_mos()
    fit = fit_logistic(x, y)
    assert fit.converged
    assert np.allclose(fit.apply(x), y, atol=1e-3)


def test_fit_handles_a_negatively_correlated_metric() -> None:
    # Pretend lower is better (like a distortion score rather than a quality one).
    x, y = synthetic_metric_and_mos()
    fit = fit_logistic(-x, y)
    assert fit.converged
    assert np.allclose(fit.apply(-x), y, atol=1e-2)


def test_logistic_fit_beats_raw_correlation_on_a_saturating_relation() -> None:
    # This is the reason the fit exists: raw Pearson between a saturating metric
    # and MOS understates agreement, because it is measuring the curve as error.
    x, y = synthetic_metric_and_mos(noise=0.08)
    raw = abs(pearsonr(x, y).statistic)
    result = evaluate(x, y, name="metric")
    assert result.plcc > raw


def test_srocc_is_unchanged_by_a_monotonic_rescaling() -> None:
    # Rank correlation cannot see a monotonic transform, which is exactly why it is
    # reported on raw predictions.
    x, y = synthetic_metric_and_mos(noise=0.05)
    plain = evaluate(x, y, name="plain")
    rescaled = evaluate(np.exp(x / 50.0), y, name="rescaled")
    assert plain.srocc == pytest.approx(rescaled.srocc)


def test_perfect_predictor_scores_one() -> None:
    x, y = synthetic_metric_and_mos()
    result = evaluate(x, y, name="perfect")
    assert result.plcc == pytest.approx(1.0, abs=1e-3)
    assert result.srocc == pytest.approx(1.0)
    assert result.krocc == pytest.approx(1.0)
    assert result.rmse < 1e-2


def test_noise_scores_near_zero_rank_correlation() -> None:
    rng = np.random.default_rng(3)
    y = rng.normal(3.0, 0.8, 60)
    x = rng.normal(50.0, 15.0, 60)
    result = evaluate(x, y, name="noise")
    assert abs(result.srocc) < 0.35


def test_rmse_is_in_label_units() -> None:
    x, y = synthetic_metric_and_mos()
    # Shift the labels by a constant the mapping cannot absorb: b5 can, so instead
    # add noise of known size and check RMSE lands in the right neighbourhood.
    rng = np.random.default_rng(11)
    noisy = y + rng.normal(0, 0.25, y.size)
    result = evaluate(x, noisy, name="noisy")
    assert 0.1 < result.rmse < 0.45


def test_evaluate_drops_non_finite_pairs(caplog: pytest.LogCaptureFixture) -> None:
    x, y = synthetic_metric_and_mos(n=20)
    x = x.copy()
    x[3] = np.nan
    with caplog.at_level("WARNING"):
        result = evaluate(x, y, name="with-nan")
    assert result.n == 19
    assert "dropped 1 non-finite" in caplog.text


def test_evaluate_needs_three_points() -> None:
    with pytest.raises(DatasetError, match="fewer than three"):
        evaluate(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_fit_rejects_mismatched_lengths() -> None:
    with pytest.raises(DatasetError, match="length mismatch"):
        fit_logistic(np.arange(5.0), np.arange(4.0))


def test_fit_rejects_too_few_points() -> None:
    with pytest.raises(DatasetError, match="at least 5 points"):
        fit_logistic(np.arange(4.0), np.arange(4.0))


def test_constant_predictor_does_not_crash() -> None:
    # A predictor with no variance has undefined correlation; report NaN, not an
    # exception, so one broken column cannot abort a whole comparison table.
    x = np.full(20, 42.0)
    y = np.linspace(1, 5, 20)
    result = evaluate(x, y, name="constant")
    assert np.isnan(result.srocc)
    assert not result.logistic_converged


def test_supplied_fit_is_reused() -> None:
    x, y = synthetic_metric_and_mos(n=20)
    identity = LogisticFit(params=(0.0, 1.0, 0.0, 1.0, 0.0), converged=True)
    result = evaluate(y, y, name="identity", fit=identity)
    assert result.rmse == pytest.approx(0.0)
    assert result.plcc == pytest.approx(1.0)


def test_single_metric_baselines_cover_every_column() -> None:
    x, y = synthetic_metric_and_mos(n=30, noise=0.1)
    frame = pd.DataFrame({"vmaf": x, "psnr_y": x * 0.4 + 20, "mos": y})
    table = evaluate_single_metric_baselines(frame, ["vmaf", "psnr_y"])
    assert list(table["name"]) == ["vmaf alone", "psnr_y alone"]
    assert (table["plcc"] > 0.9).all()


def test_single_metric_baselines_reject_missing_columns() -> None:
    frame = pd.DataFrame({"vmaf": [1.0, 2.0, 3.0], "mos": [1.0, 2.0, 3.0]})
    with pytest.raises(DatasetError, match="missing columns"):
        evaluate_single_metric_baselines(frame, ["vmaf", "nope"])


def test_comparison_table_is_sorted_by_srocc() -> None:
    x, y = synthetic_metric_and_mos(n=30, noise=0.1)
    rng = np.random.default_rng(5)
    good = evaluate(x, y, name="good")
    bad = evaluate(rng.normal(size=30), y, name="bad")
    table = comparison_table([bad, good])
    assert list(table["name"]) == ["good", "bad"]


def test_logistic_fit_survives_a_tiny_x_scale() -> None:
    """A metric spanning 0.05 must fit as well as one spanning 50.

    The fit runs on standardised predictions for exactly this reason: on raw units
    the solver has to find a slope near ``1/spread`` and a centre near the mean,
    which differ by orders of magnitude when the spread is small, and it exhausts
    its iteration budget instead of converging. A non-converged baseline is not a
    cosmetic problem - it silently understates the metric the fused model has to
    beat.
    """
    rng = np.random.default_rng(0)
    # An SSIM-like feature: values crowded into 0.78-0.99.
    x = np.linspace(0.78, 0.99, 60)
    y = 3.0 * x**2 - 1.5 + rng.normal(0, 0.02, x.size)

    fit = fit_logistic(x, y)
    assert fit.converged
    assert pearsonr(fit.apply(x), y).statistic > 0.95


def test_logistic_fit_is_invariant_to_the_units_of_x() -> None:
    """Rescaling the predictions must not change the mapped values.

    This is what the un-standardisation step has to guarantee: the parameters come
    back in raw-``x`` coordinates, so fitting the same relationship expressed in
    different units lands in the same place.
    """
    x = np.linspace(20.0, 60.0, 40)
    y = 0.05 * x - 1.0

    plain = fit_logistic(x, y)
    scaled = fit_logistic(x * 1000.0, y)
    assert np.allclose(plain.apply(x), scaled.apply(x * 1000.0), atol=1e-6)
