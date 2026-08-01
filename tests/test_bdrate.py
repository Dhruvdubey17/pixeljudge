"""BD-Rate tests built on curve pairs whose answer is known in closed form.

The trick that makes these exact: if the challenger needs exactly half the
bitrate of the anchor at *every* quality level, then the mean difference in
log10-bitrate is log10(0.5) everywhere, so BD-Rate must be exactly -50%. Any
interpolation or unit-conversion mistake breaks that identity immediately.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

from pixeljudge.errors import MetricsError
from pixeljudge.metrics.bdrate import RdCurve, bd_quality, bd_rate, bd_table

ANCHOR_RATES = (500.0, 1000.0, 2000.0, 4000.0, 8000.0)
ANCHOR_QUALITY = (70.0, 80.0, 87.0, 92.0, 95.0)


def anchor_curve(name: str = "h264") -> RdCurve:
    return RdCurve(name=name, bitrates_kbps=ANCHOR_RATES, quality=ANCHOR_QUALITY)


def scaled_curve(factor: float, name: str = "hevc") -> RdCurve:
    """Same quality at ``factor`` times the bitrate."""
    return RdCurve(
        name=name,
        bitrates_kbps=tuple(rate * factor for rate in ANCHOR_RATES),
        quality=ANCHOR_QUALITY,
    )


def test_identical_curves_have_zero_bd_rate() -> None:
    assert bd_rate(anchor_curve(), anchor_curve("copy")) == pytest.approx(0.0, abs=1e-9)


def test_half_bitrate_curve_is_minus_fifty_percent() -> None:
    assert bd_rate(anchor_curve(), scaled_curve(0.5)) == pytest.approx(-50.0, abs=1e-6)


def test_double_bitrate_curve_is_plus_hundred_percent() -> None:
    assert bd_rate(anchor_curve(), scaled_curve(2.0)) == pytest.approx(100.0, abs=1e-6)


def test_bd_rate_is_asymmetric_in_the_expected_way() -> None:
    # -50% one way is +100% the other: halving the bits is the same as doubling.
    forward = bd_rate(anchor_curve(), scaled_curve(0.5))
    backward = bd_rate(scaled_curve(0.5), anchor_curve())
    assert forward == pytest.approx(-50.0, abs=1e-6)
    assert backward == pytest.approx(100.0, abs=1e-6)


def test_bd_quality_of_a_constant_offset_curve_is_that_offset() -> None:
    better = RdCurve(
        name="better",
        bitrates_kbps=ANCHOR_RATES,
        quality=tuple(q + 3.0 for q in ANCHOR_QUALITY),
    )
    assert bd_quality(anchor_curve(), better) == pytest.approx(3.0, abs=1e-9)


def test_bd_quality_zero_for_identical_curves() -> None:
    assert bd_quality(anchor_curve(), anchor_curve("copy")) == pytest.approx(0.0, abs=1e-9)


def test_non_overlapping_curves_raise() -> None:
    # A sweep that never reaches into the anchor's quality range cannot be
    # compared; extrapolating would invent the answer.
    disjoint = RdCurve(
        name="way_better",
        bitrates_kbps=ANCHOR_RATES,
        quality=(96.5, 97.0, 97.5, 98.0, 99.0),
    )
    with pytest.raises(MetricsError, match="do not overlap"):
        bd_rate(anchor_curve(), disjoint)


def test_too_few_points_raise() -> None:
    with pytest.raises(MetricsError, match="at least 4"):
        RdCurve(name="short", bitrates_kbps=(1000.0, 2000.0), quality=(80.0, 90.0))


def test_four_points_are_allowed_but_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("WARNING"):
        RdCurve(
            name="four",
            bitrates_kbps=(500.0, 1000.0, 2000.0, 4000.0),
            quality=(70.0, 80.0, 87.0, 92.0),
        )
    assert "only 4 points" in caplog.text


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(MetricsError, match="bitrates but"):
        RdCurve(name="bad", bitrates_kbps=(1.0, 2.0, 3.0, 4.0), quality=(1.0, 2.0))


def test_non_positive_bitrate_raises() -> None:
    with pytest.raises(MetricsError, match="non-positive bitrate"):
        RdCurve(name="bad", bitrates_kbps=(0.0, 1.0, 2.0, 3.0), quality=(1.0, 2.0, 3.0, 4.0))


def test_non_finite_quality_raises() -> None:
    with pytest.raises(MetricsError, match="non-finite"):
        RdCurve(
            name="bad",
            bitrates_kbps=(1.0, 2.0, 3.0, 4.0),
            quality=(1.0, 2.0, float("nan"), 4.0),
        )


def test_quality_dip_is_reported_but_still_computed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # A real sweep occasionally dips: more bitrate, slightly lower VMAF. Sorting by
    # quality would hide it, so the curve itself complains at construction time.
    with caplog.at_level("WARNING"):
        wobbly = RdCurve(
            name="wobbly",
            bitrates_kbps=ANCHOR_RATES,
            quality=(70.0, 80.0, 79.5, 92.0, 95.0),
        )
    assert "non-monotonic" in caplog.text
    assert isinstance(bd_rate(anchor_curve(), wobbly), float)


def test_duplicate_quality_points_are_deduplicated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Two CRF values that landed on the same VMAF: PCHIP needs strictly
    # increasing sample points, so one of them has to go.
    tied = RdCurve(
        name="tied",
        bitrates_kbps=ANCHOR_RATES,
        quality=(70.0, 80.0, 80.0, 92.0, 95.0),
    )
    with caplog.at_level("WARNING"):
        value = bd_rate(anchor_curve(), tied)
    assert "dropped 1 non-monotonic point" in caplog.text
    assert isinstance(value, float)


def test_metric_mismatch_raises() -> None:
    psnr_curve = RdCurve(
        name="psnr", bitrates_kbps=ANCHOR_RATES, quality=ANCHOR_QUALITY, metric="psnr_y"
    )
    with pytest.raises(MetricsError, match="different metrics"):
        bd_rate(anchor_curve(), psnr_curve)


def test_ssim_curves_warn_about_instability(caplog: pytest.LogCaptureFixture) -> None:
    ssim_a = RdCurve(
        name="a",
        bitrates_kbps=ANCHOR_RATES,
        quality=(0.90, 0.95, 0.97, 0.98, 0.99),
        metric="float_ssim",
    )
    ssim_b = RdCurve(
        name="b",
        bitrates_kbps=tuple(r * 0.5 for r in ANCHOR_RATES),
        quality=(0.90, 0.95, 0.97, 0.98, 0.99),
        metric="float_ssim",
    )
    with caplog.at_level("WARNING"):
        bd_rate(ssim_a, ssim_b)
    assert "unstable" in caplog.text


def test_from_frame_reads_a_metrics_table() -> None:
    frame = pd.DataFrame(
        {
            "actual_bitrate_kbps": [4000.0, 500.0, 1000.0, 2000.0, 8000.0],
            "vmaf": [92.0, 70.0, 80.0, 87.0, 95.0],
        }
    )
    curve = RdCurve.from_frame(frame, "h264")
    # from_frame must sort by bitrate, not trust the row order.
    assert curve.bitrates_kbps == ANCHOR_RATES
    assert curve.quality == ANCHOR_QUALITY


def test_from_frame_missing_column_raises() -> None:
    frame = pd.DataFrame({"vmaf": [1.0, 2.0, 3.0, 4.0]})
    with pytest.raises(MetricsError, match="missing"):
        RdCurve.from_frame(frame, "h264")


def test_bd_table_ranks_codecs_and_excludes_the_anchor() -> None:
    curves = {
        "h264": anchor_curve(),
        "hevc": scaled_curve(0.5, "hevc"),
        "av1": scaled_curve(0.35, "av1"),
    }
    table = bd_table(curves, anchor="h264")
    assert list(table["codec"]) == ["av1", "hevc"]  # sorted by best saving first
    assert table.loc[table["codec"] == "hevc", "bd_rate_pct"].iloc[0] == pytest.approx(-50.0)
    assert (table["bd_quality"] > 0).all()


def test_bd_table_unknown_anchor_raises() -> None:
    with pytest.raises(MetricsError, match="not among"):
        bd_table({"h264": anchor_curve()}, anchor="vp9")


def test_bd_table_skips_an_incomparable_pair_instead_of_aborting(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One un-comparable pair must not take the whole table down with it.

    Real case: on near-flat content x264 spends 10-31 kbit/s while SVT-AV1's floor puts
    it at 60-200, so their bitrate ranges are disjoint and BD-quality cannot be
    integrated. The fifteen comparisons that *do* work should still be reported.
    """
    curves = {
        "h264": anchor_curve(),
        "hevc": scaled_curve(0.5, "hevc"),
        # Same quality range (so BD-Rate works) but a bitrate range far above the
        # anchor's, so BD-quality has no overlap to integrate over.
        "av1": RdCurve(
            name="av1",
            bitrates_kbps=tuple(rate * 200 for rate in ANCHOR_RATES),
            quality=ANCHOR_QUALITY,
        ),
    }
    with caplog.at_level("WARNING"):
        table = bd_table(curves, anchor="h264")
    assert set(table["codec"]) == {"hevc", "av1"}
    assert table.loc[table["codec"] == "hevc", "bd_rate_pct"].iloc[0] == pytest.approx(-50.0)
    # The impossible cell is empty and the reason was logged.
    assert pd.isna(table.loc[table["codec"] == "av1", "bd_quality"].iloc[0])
    assert "skipping bd_quality" in caplog.text


def test_all_skipped_column_stays_numeric_so_tables_concatenate_cleanly(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A column where *every* comparison was skipped must still be float64.

    The report builds one BD table per source clip and concatenates them. On the
    synthetic gradient clip every BD-quality comparison is skipped for non-overlap,
    so that whole column is missing. Filling it with ``None`` would infer an
    ``object`` column, and concatenating object with float64 makes pandas guess at
    the result dtype - the "concatenation with empty or all-NA entries" deprecation.
    The values are the same either way; only the dtype differs, so this is purely
    about not relying on behaviour pandas has announced it will change.

    Note what is *not* the fix: dropping such frames before the concat. That would
    discard the gradient clip's perfectly good BD-Rate numbers along with its empty
    BD-quality column.
    """
    # Every challenger sits far above the anchor's bitrate range, so BD-quality has
    # nothing to integrate over for any of them.
    curves = {
        "h264": anchor_curve(),
        "hevc": RdCurve(
            name="hevc",
            bitrates_kbps=tuple(rate * 200 for rate in ANCHOR_RATES),
            quality=ANCHOR_QUALITY,
        ),
        "av1": RdCurve(
            name="av1",
            bitrates_kbps=tuple(rate * 300 for rate in ANCHOR_RATES),
            quality=ANCHOR_QUALITY,
        ),
    }
    with caplog.at_level("WARNING"):
        table = bd_table(curves, anchor="h264")

    assert table["bd_quality"].isna().all()
    assert table["bd_quality"].dtype == "float64"

    # The real shape of the bug: concatenating this table with a fully-populated one.
    populated = bd_table({"h264": anchor_curve(), "hevc": scaled_curve(0.5, "hevc")}, "h264")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        combined = pd.concat([populated, table], ignore_index=True)

    assert len(combined) == 3
    assert combined["bd_quality"].dtype == "float64"
    # BD-Rate survived for every row even though BD-quality did not.
    assert combined["bd_rate_pct"].notna().all()
