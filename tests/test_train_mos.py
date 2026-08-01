"""Cross-validation discipline for the subjective regression, on a synthetic table.

These tests are about the protocol, not the numbers. The failure they exist to
catch is the quiet one: a split that lets a content appear in training and test at
once still produces a plausible correlation, just an inflated one, and nothing
crashes. So the fixture is built so that leakage would be *visible* - the label is
driven by a per-content offset that a model can only exploit if it has seen that
content.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pixeljudge.errors import DatasetError
from pixeljudge.model.evaluation import summarise_folds
from pixeljudge.model.train import (
    _is_partition,
    grouped_splits,
    masks_to_splits,
    train_and_evaluate,
)

FEATURES = ["vmaf_mean", "vmaf_hmean", "psnr_mean", "ssim_mean"]
LABEL = "subj_score"


def make_table(n_contents: int = 8, n_conditions: int = 4, seed: int = 0) -> pd.DataFrame:
    """A feature table shaped like the real one: contents x playout conditions.

    The label depends on quality plus a per-content offset. The offset is the trap:
    a model that has seen a content in training can fit its offset and look good on
    that content's other conditions, which is exactly what grouped CV must prevent.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for content in range(n_contents):
        offset = rng.normal(0, 0.4)
        for condition in range(n_conditions):
            quality = 20.0 + 20.0 * condition + rng.normal(0, 2.0)
            rows.append(
                {
                    "video_id": f"content_{content}_seq_{condition}",
                    "content": f"content_{content:02d}",
                    "content_index": content + 1,
                    "condition_index": condition,
                    "vmaf_mean": quality,
                    "vmaf_hmean": quality - 2.0,
                    "psnr_mean": 25.0 + 0.12 * quality + rng.normal(0, 0.5),
                    "ssim_mean": 0.90 + 0.001 * quality + rng.normal(0, 0.002),
                    LABEL: 0.02 * quality - 1.0 + offset + rng.normal(0, 0.05),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def table() -> pd.DataFrame:
    return make_table()


def test_grouped_folds_never_share_a_content(table: pd.DataFrame) -> None:
    for train, test in grouped_splits(table["content"], 4):
        assert not set(table["content"].iloc[train]) & set(table["content"].iloc[test])


def test_grouped_folds_partition_the_rows(table: pd.DataFrame) -> None:
    splits = grouped_splits(table["content"], 4)
    assert _is_partition(splits, len(table))


def test_sampled_trials_are_not_a_partition(table: pd.DataFrame) -> None:
    """Two overlapping 75/25 trials must not be mistaken for out-of-fold coverage.

    If they were, each clip's prediction would be overwritten by whichever trial
    ran last and the report would silently describe one trial instead of all of
    them.
    """
    contents = table["content"].to_numpy()
    masks = [~np.isin(contents, ["content_00", "content_01"]), ~np.isin(contents, ["content_02"])]
    splits = masks_to_splits(masks)
    assert not _is_partition(splits, len(table))


def test_report_covers_models_and_baselines(table: pd.DataFrame) -> None:
    report = train_and_evaluate(
        table,
        features=FEATURES,
        label=LABEL,
        n_splits=4,
        baseline_metrics=["vmaf_mean", "psnr_mean", "ssim_mean"],
    )
    names = {result.name for result in report.results}
    assert {"svr_rbf", "random_forest", "ridge"} <= names
    assert {"vmaf_mean alone", "psnr_mean alone", "ssim_mean alone"} <= names
    assert report.pooled_is_valid
    assert report.oof[[c for c in report.oof.columns if c.startswith("pred_")]].notna().all().all()
    # Every predictor also gets a fold-level summary, which is what the report
    # quotes as mean +/- std.
    assert {summary.name for summary in report.fold_summaries} == names
    assert all(summary.n_folds == 4 for summary in report.fold_summaries)


def test_grouped_cv_is_not_inflated_by_content_memorisation() -> None:
    """Grouped CV must score no better than a split that lets contents leak.

    Built the way leakage actually pays in this domain. Each content gets a
    signature the features expose - a distinct PSNR base level, which is真 of real
    material: the SSIM of a detailed scene and a flat one are not comparable - and
    an offset on the label that only that signature predicts. A model that has seen
    a content in training can read the signature and recover the offset for its
    other conditions. Grouping by content removes that shortcut.

    Compared against a run where the content labels are shuffled, so the same rows
    are split without respecting content. If grouped CV ever scored *higher*, the
    grouping would not be doing anything.
    """
    rng = np.random.default_rng(7)
    rows = []
    for content in range(6):
        # The signature: a per-content offset visible in the features...
        signature = 4.0 * content
        # ...and a label shift that only the signature explains.
        shift = 1.2 * ((content % 3) - 1)
        for condition in range(6):
            quality = 20.0 + 12.0 * condition + rng.normal(0, 1.5)
            rows.append(
                {
                    "content": f"content_{content:02d}",
                    "vmaf_mean": quality,
                    "vmaf_hmean": quality - 2.0,
                    "psnr_mean": 25.0 + signature + 0.1 * quality + rng.normal(0, 0.3),
                    "ssim_mean": 0.90 + 0.0005 * signature + 0.001 * quality,
                    LABEL: 0.02 * quality - 1.0 + shift + rng.normal(0, 0.05),
                }
            )
    frame = pd.DataFrame(rows)

    grouped = train_and_evaluate(frame, features=FEATURES, label=LABEL, n_splits=3)

    leaky_frame = frame.copy()
    leaky_frame["content"] = rng.permutation(leaky_frame["content"].to_numpy())
    leaky = train_and_evaluate(leaky_frame, features=FEATURES, label=LABEL, n_splits=3)

    grouped_best = max(r.srocc for r in grouped.results if not r.name.endswith(" alone"))
    leaky_best = max(r.srocc for r in leaky.results if not r.name.endswith(" alone"))
    assert grouped_best < leaky_best


def test_released_splits_report_across_trial_means(table: pd.DataFrame) -> None:
    """With sampled trials there is no out-of-fold vector, so folds are the result."""
    contents = sorted(table["content"].unique())
    masks = [
        ~table["content"].isin(contents[0:2]).to_numpy(),
        ~table["content"].isin(contents[2:4]).to_numpy(),
        ~table["content"].isin(contents[4:6]).to_numpy(),
    ]
    report = train_and_evaluate(
        table,
        features=FEATURES,
        label=LABEL,
        n_splits=3,
        baseline_metrics=["vmaf_mean"],
        splits=masks_to_splits(masks),
    )
    assert not report.pooled_is_valid
    # No pooled OOF is written, because a clip is tested by more than one trial.
    assert not [c for c in report.oof.columns if c.startswith("pred_")]
    assert report.n_splits == 3
    # The headline numbers are the across-trial means.
    svr = report.result_for("svr_rbf")
    summary = next(s for s in report.fold_summaries if s.name == "svr_rbf")
    assert svr is not None
    assert svr.srocc == pytest.approx(summary.srocc_mean)


def test_summarise_folds_reports_spread() -> None:
    truth = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    perfect = (truth.copy(), truth)
    reversed_fold = (truth[::-1].copy(), truth)
    summary = summarise_folds([perfect, reversed_fold], name="mixed")

    assert summary.n_folds == 2
    assert summary.srocc_mean == pytest.approx(0.0)
    # One fold ranks perfectly, one perfectly backwards: the mean hides that and
    # the standard deviation is the only thing that shows it.
    assert summary.srocc_std == pytest.approx(1.0)


def test_summarise_folds_skips_unscorable_folds() -> None:
    usable = (np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 3.0, 4.0]))
    too_small = (np.array([1.0]), np.array([1.0]))
    summary = summarise_folds([usable, too_small], name="sparse")
    assert summary.n_folds == 1


def test_summarise_folds_needs_one_usable_fold() -> None:
    with pytest.raises(DatasetError, match="no fold"):
        summarise_folds([(np.array([1.0]), np.array([1.0]))], name="empty")


def test_baseline_logistic_is_fitted_per_fold_not_globally(table: pd.DataFrame) -> None:
    """A baseline must not see held-out rows through its logistic mapping.

    Fitting the mapping on all the data first is the subtle version of leakage: the
    metric is unchanged, but the curve placing it on the label scale was chosen
    using the answers. Detected here by checking the baseline's pooled prediction
    for a fold differs from what a globally-fitted mapping would give.
    """
    from pixeljudge.model.evaluation import fit_logistic
    from pixeljudge.model.train import _baseline_oof

    folds = grouped_splits(table["content"], 4)
    predictions, per_fold = _baseline_oof(table, "vmaf_mean", LABEL, folds)
    global_map = fit_logistic(table["vmaf_mean"], table[LABEL]).apply(table["vmaf_mean"])

    assert len(per_fold) == 4
    assert not np.allclose(predictions, global_map)


def test_compare_feature_tables_flags_a_scale_mismatch() -> None:
    """The Stage 3 agreement check must separate rank agreement from offset.

    Two VMAF implementations that differ by a constant still rank clips
    identically, and that would not change a single conclusion in the report. One
    that ranks them differently is a real defect. Spearman and the difference
    columns are reported separately so the two cases are told apart rather than
    collapsed into one "how close is it" number.
    """
    from pixeljudge.model.features import compare_feature_tables

    theirs = pd.DataFrame(
        {"video_id": [f"v{i}" for i in range(6)], "vmaf_mean": [10.0, 20, 30, 40, 50, 60]}
    )
    offset = theirs.assign(vmaf_mean=theirs["vmaf_mean"] + 2.0)
    scrambled = theirs.assign(vmaf_mean=[30.0, 10, 60, 20, 50, 40])

    agreed = compare_feature_tables(theirs, offset).iloc[0]
    assert agreed["spearman"] == pytest.approx(1.0)
    assert agreed["mean_difference"] == pytest.approx(2.0)

    disagreed = compare_feature_tables(theirs, scrambled).iloc[0]
    assert disagreed["spearman"] < 0.95


def test_compare_feature_tables_refuses_a_table_it_cannot_join() -> None:
    from pixeljudge.model.features import compare_feature_tables

    theirs = pd.DataFrame({"video_id": ["a", "b", "c"], "vmaf_mean": [1.0, 2.0, 3.0]})
    elsewhere = pd.DataFrame({"video_id": ["x", "y", "z"], "vmaf_mean": [1.0, 2.0, 3.0]})
    with pytest.raises(DatasetError, match="no clip in common"):
        compare_feature_tables(theirs, elsewhere)
