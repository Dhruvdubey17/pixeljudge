"""Per-title ladder maths.

The hull is where per-title encoding actually happens, and it is pure geometry, so
it can be tested against cases whose answer is obvious by construction: a point
that is both dearer and worse than another must not survive.
"""

from __future__ import annotations

import pytest

from pixeljudge.errors import ConfigError
from pixeljudge.ladder.builder import (
    RdPoint,
    ladder_to_yaml_dict,
    per_title_ladder,
    probe_grid,
    select_rungs,
    trim_to_target_quality,
    upper_convex_hull,
)


def point(height: int, bitrate: float, vmaf: float, crf: int | None = None) -> RdPoint:
    return RdPoint(height=height, bitrate_kbps=bitrate, vmaf=vmaf, crf=crf)


def test_hull_keeps_a_rising_frontier() -> None:
    points = [point(720, 500, 70), point(720, 1000, 80), point(720, 2000, 88)]
    assert [p.bitrate_kbps for p in upper_convex_hull(points)] == [500, 1000, 2000]


def test_hull_drops_a_dominated_point() -> None:
    # 1000 kbps at VMAF 60 is beaten outright by 500 kbps at VMAF 70: cheaper and
    # better, so the expensive one can never be worth shipping.
    points = [point(720, 500, 70), point(720, 1000, 60), point(720, 2000, 88)]
    hull = upper_convex_hull(points)
    assert 1000 not in [p.bitrate_kbps for p in hull]


def test_hull_prefers_the_better_of_two_points_at_the_same_bitrate() -> None:
    # This is the resolution decision: at one bitrate, one resolution wins.
    points = [
        point(360, 1000, 82),
        point(720, 1000, 75),
        point(720, 4000, 95),
        point(360, 400, 70),
    ]
    hull = upper_convex_hull(points)
    at_1000 = [p for p in hull if p.bitrate_kbps == 1000]
    assert len(at_1000) == 1
    assert at_1000[0].height == 360


def test_hull_drops_interior_points_below_the_envelope() -> None:
    # A point sitting under the straight line between its neighbours is not on the
    # frontier: interpolating between the neighbours already beats it.
    points = [point(720, 500, 60), point(720, 1000, 65), point(720, 2000, 90)]
    hull = upper_convex_hull(points)
    assert [p.bitrate_kbps for p in hull] == [500, 2000]


def test_hull_of_a_single_point_or_nothing() -> None:
    assert upper_convex_hull([]) == []
    assert len(upper_convex_hull([point(720, 1000, 80)])) == 1


def test_zero_bitrate_point_is_refused() -> None:
    # log10(0) has no answer, and a zero-byte encode is a failed encode.
    with pytest.raises(ConfigError, match="non-positive bitrate"):
        upper_convex_hull([point(720, 0, 80), point(720, 1000, 85)])


def test_trim_stops_at_the_first_point_that_reaches_the_target() -> None:
    hull = [point(360, 500, 70), point(720, 2000, 90), point(720, 4000, 94.5), point(720, 8000, 97)]
    trimmed = trim_to_target_quality(hull, target_vmaf=94.0)
    # Everything past the target is bitrate spent on quality nobody can see.
    assert [p.bitrate_kbps for p in trimmed] == [500, 2000, 4000]


def test_trim_keeps_everything_and_warns_when_the_target_is_unreachable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    hull = [point(360, 500, 60), point(720, 2000, 75)]
    with caplog.at_level("WARNING"):
        trimmed = trim_to_target_quality(hull, target_vmaf=94.0)
    assert len(trimmed) == 2
    assert "no probe point reached VMAF" in caplog.text


def test_select_rungs_spaces_them_in_log_bitrate() -> None:
    hull = [point(360, 10 ** (i / 4), 60 + i) for i in range(9)]
    chosen = select_rungs(hull, 3)
    assert len(chosen) == 3
    assert chosen[0].bitrate_kbps == pytest.approx(1.0)
    assert chosen[-1].bitrate_kbps == pytest.approx(10**2)
    # The middle rung should sit near the geometric mean, not the arithmetic one.
    assert chosen[1].bitrate_kbps == pytest.approx(10.0)


def test_select_rungs_returns_everything_when_asked_for_more_than_exists() -> None:
    hull = [point(360, 500, 70), point(720, 2000, 90)]
    assert len(select_rungs(hull, 5)) == 2


def test_select_rungs_refuses_zero() -> None:
    with pytest.raises(ConfigError, match="at least one rung"):
        select_rungs([point(360, 500, 70)], 0)


def test_per_title_ladder_produces_bitrate_rungs_from_crf_probes() -> None:
    points = [
        point(360, 400, 72, crf=32),
        point(360, 900, 84, crf=26),
        point(720, 1800, 91, crf=26),
        point(720, 3600, 95, crf=20),
        point(720, 7200, 97, crf=14),
    ]
    ladder = per_title_ladder(points, codec="h264", n_rungs=3, target_vmaf=94.0)
    assert ladder.codec == "h264"
    # A packager ships bitrates, so CRF probes become bitrate targets.
    assert all(rung.bitrate_kbps is not None for rung in ladder.rungs)
    assert all(rung.crf is None for rung in ladder.rungs)
    # Nothing above the quality target survives.
    assert max(rung.bitrate_kbps or 0 for rung in ladder.rungs) <= 3600


def test_per_title_ladder_needs_points() -> None:
    with pytest.raises(ConfigError, match="at least one measured RD point"):
        per_title_ladder([], codec="h264")


def test_probe_grid_covers_every_resolution_and_quality_pair() -> None:
    grid = probe_grid("h264", heights=(360, 720), crfs=(20, 30, 40))
    assert len(grid.rungs) == 6
    assert {rung.height for rung in grid.rungs} == {360, 720}


def test_probe_grid_skips_resolutions_above_the_master() -> None:
    # Upscaling invents no detail, so such a point can never be usefully on the hull.
    grid = probe_grid("h264", heights=(360, 720, 1080), source_height=720)
    assert {rung.height for rung in grid.rungs} == {360, 720}


def test_probe_grid_refuses_when_nothing_fits() -> None:
    with pytest.raises(ConfigError, match="no probe height fits"):
        probe_grid("h264", heights=(1080,), source_height=720)


def test_ladder_yaml_dict_omits_empty_fields() -> None:
    ladder = per_title_ladder(
        [
            point(360, 400, 72),
            point(360, 900, 84),
            point(720, 1800, 91),
            point(720, 3600, 96),
        ],
        codec="hevc",
        n_rungs=2,
    )
    payload = ladder_to_yaml_dict(ladder)
    assert payload["codec"] == "hevc"
    for rung in payload["rungs"]:
        assert "crf" not in rung  # bitrate-mode rungs carry no CRF
        assert rung["bitrate_kbps"] > 0
