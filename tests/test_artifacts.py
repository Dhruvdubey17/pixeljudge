"""Artifact detector tests, all on synthetic images whose truth is known.

The key test is ``test_banded_gradient_scores_higher_than_smooth``: same content,
same value range, the only difference is how many intensity levels survive. That
is exactly what quantisation does, so if the detector cannot separate those two
images it cannot separate a banded encode from a clean one either.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pixeljudge.artifacts.detectors import (
    MIN_VISIBLE_STEP,
    banded_gradient,
    banding_score,
    blocking_score,
    blur_score,
    checkerboard,
    score_frame,
    sharpness,
    smooth_gradient,
    to_luma,
)
from pixeljudge.artifacts.scan import (
    find_metric_blind_spots,
    sample_frame_indices,
    save_evidence_frame,
)


def test_banded_gradient_scores_higher_than_smooth() -> None:
    smooth = banding_score(smooth_gradient())[0]
    banded = banding_score(banded_gradient(levels=6))[0]
    assert banded > smooth
    assert smooth == pytest.approx(0.0)


def test_eight_bit_quantised_ramp_is_not_called_banding() -> None:
    # A real 8-bit master steps by one code value here and there. If that counted,
    # every pristine gradient would look banded and the detector would be useless.
    eight_bit = np.round(smooth_gradient()).astype(np.float32)
    assert banding_score(eight_bit)[0] == pytest.approx(0.0)


def test_banding_score_grows_as_levels_are_removed() -> None:
    scores = [banding_score(banded_gradient(levels=n))[0] for n in (12, 6, 4)]
    assert scores == sorted(scores), scores  # fewer levels must score worse
    assert scores[0] > 0


def test_banding_ignores_steps_below_visibility() -> None:
    # 24 levels over a 32-code-value ramp is a step of ~1.4, below the threshold.
    assert np.ptp(banded_gradient(levels=24)) > 0
    assert banding_score(banded_gradient(levels=24))[0] == pytest.approx(0.0)
    assert MIN_VISIBLE_STEP > 1.4


def test_banding_reports_flat_fraction() -> None:
    _, flat_smooth = banding_score(smooth_gradient())
    _, flat_texture = banding_score(checkerboard())
    assert flat_smooth > 0.9  # a ramp is nearly all smooth
    assert flat_texture < 0.05  # a checkerboard has no smooth area at all


def test_texture_is_not_reported_as_banding() -> None:
    rng = np.random.default_rng(0)
    noise = rng.normal(128, 20, (180, 320)).astype(np.float32)
    assert banding_score(noise)[0] == pytest.approx(0.0)


def test_a_mostly_textured_frame_does_not_outscore_a_banded_one() -> None:
    """The bug this guards against inverted the detector on real content.

    Normalising by *smooth area* meant a frame that is 3% smooth had a tiny
    denominator, so a handful of qualifying pixels produced an enormous score.
    Photographic clips then outranked actual skies by two orders of magnitude. The
    score is normalised by total pixels instead, so a small banded patch inside a
    textured frame stays a small contribution.
    """
    rng = np.random.default_rng(2)
    frame = rng.normal(128, 25, (180, 320)).astype(np.float32)
    patch = banded_gradient(width=60, height=40, levels=6)
    frame[10:50, 10:70] = patch

    mostly_texture = banding_score(frame)[0]
    fully_banded = banding_score(banded_gradient(levels=6))[0]
    assert mostly_texture > 0  # the patch is found
    assert mostly_texture < fully_banded  # but it does not dominate
    assert banding_score(frame)[1] < 0.2  # and the frame is reported as un-smooth


def test_checkerboard_scores_high_on_blocking() -> None:
    # Every discontinuity sits exactly on the 8-pixel grid, which is the strongest
    # possible blocking signal.
    assert blocking_score(checkerboard()) > 100
    assert blocking_score(smooth_gradient()) < 2


def test_blocking_of_random_noise_is_about_one() -> None:
    # Noise has no reason to prefer multiples of eight, so on-grid and off-grid
    # differences should match: a ratio near 1 means "no blocking".
    rng = np.random.default_rng(1)
    noise = rng.normal(128, 30, (180, 320)).astype(np.float32)
    assert 0.9 < blocking_score(noise) < 1.1


def test_blocking_detects_an_offset_grid_less_strongly() -> None:
    # A checkerboard whose period is 7 pixels is not aligned to the coding grid,
    # so it must not be reported as blocking.
    aligned = blocking_score(checkerboard(block=8))
    misaligned = blocking_score(checkerboard(block=7))
    assert aligned > misaligned


def test_blocking_on_a_tiny_frame_is_zero() -> None:
    assert blocking_score(np.zeros((4, 4), dtype=np.float32)) == 0.0


def test_blur_score_rises_as_sharpness_falls() -> None:
    import cv2

    sharp = checkerboard()
    soft = cv2.GaussianBlur(sharp, (9, 9), 3)
    assert sharpness(sharp) > sharpness(soft)
    assert blur_score(soft) > blur_score(sharp)


def test_to_luma_handles_colour_and_grayscale() -> None:
    colour = np.zeros((10, 10, 3), dtype=np.uint8)
    colour[:, :, 2] = 255  # pure red in BGR
    luma = to_luma(colour)
    assert luma.shape == (10, 10)
    assert 50 < luma.mean() < 100  # red contributes ~30% of luma
    assert to_luma(np.zeros((10, 10), dtype=np.uint8)).shape == (10, 10)


def test_score_frame_returns_every_detector() -> None:
    scores = score_frame(banded_gradient(levels=6))
    assert scores.banding > 0
    assert scores.blur > 0
    assert set(scores.as_dict()) == {"blur", "blocking", "banding", "flat_fraction"}


def test_empty_frame_is_handled() -> None:
    assert banding_score(np.zeros((0, 0), dtype=np.float32)) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# Sampling and reporting logic (no decoding involved)
# ---------------------------------------------------------------------------


def test_sample_frame_indices_are_interior_and_spread() -> None:
    indices = sample_frame_indices(100, 4)
    assert indices == [12, 37, 62, 87]
    # Never the first frame (an unrepresentatively clean keyframe) or the last.
    assert 0 not in indices
    assert 99 not in indices


def test_sample_frame_indices_are_deterministic() -> None:
    assert sample_frame_indices(250, 12) == sample_frame_indices(250, 12)


def test_sample_frame_indices_handles_short_and_empty_clips() -> None:
    assert sample_frame_indices(3, 12) == [0, 1, 2]
    assert sample_frame_indices(0, 12) == []
    assert sample_frame_indices(100, 0) == []


def test_save_evidence_frame_writes_a_png(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "frame.png"
    save_evidence_frame(banded_gradient(levels=6).astype(np.uint8), destination)
    assert destination.exists()
    assert destination.stat().st_size > 0


def test_find_metric_blind_spots_uses_introduced_banding_not_absolute() -> None:
    """The check that keeps the project's headline claim honest.

    Both clips score 60 for absolute banding. One got there because its master already
    had contours (stylised animation); the other's master was clean and the encoder put
    them there. Only the second is a metric blind spot, and an absolute score cannot
    tell them apart. This is not hypothetical: the Sintel master scores 52 on its own.
    """
    import pandas as pd

    metrics = pd.DataFrame(
        {
            "distorted": ["stylised.mp4", "wrecked.mp4"],
            "vmaf": [99.7, 95.0],
            "psnr_y": [50.0, 42.0],
        }
    )
    artifacts = pd.DataFrame(
        {
            "path": ["out/stylised.mp4", "out/wrecked.mp4"],
            "banding_max": [60.0, 60.0],
            "reference_banding_mean": [58.0, 2.0],
            "banding_delta_max": [2.0, 55.0],
            "banding_delta_worst_frame": [40, 40],
        }
    )
    blind = find_metric_blind_spots(metrics, artifacts)
    assert list(blind["distorted"]) == ["wrecked.mp4"]


def test_find_metric_blind_spots_refuses_without_a_reference_scan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Without deltas the question cannot be answered, so return nothing and say why
    # rather than falling back to absolute scores and producing false positives.
    import pandas as pd

    metrics = pd.DataFrame({"distorted": ["a.mp4"], "vmaf": [99.0], "psnr_y": [48.0]})
    artifacts = pd.DataFrame({"path": ["out/a.mp4"], "banding_max": [90.0]})
    with caplog.at_level("WARNING"):
        blind = find_metric_blind_spots(metrics, artifacts)
    assert blind.empty
    assert "cannot separate introduced banding" in caplog.text


def test_find_metric_blind_spots_ignores_clips_the_metrics_already_condemn() -> None:
    import pandas as pd

    metrics = pd.DataFrame({"distorted": ["bad.mp4"], "vmaf": [41.0], "psnr_y": [28.0]})
    artifacts = pd.DataFrame(
        {"path": ["out/bad.mp4"], "banding_max": [90.0], "banding_delta_max": [80.0]}
    )
    # The metrics are not blind here: they already say the clip is bad.
    assert find_metric_blind_spots(metrics, artifacts).empty


def test_find_metric_blind_spots_can_use_a_population_quantile() -> None:
    import pandas as pd

    names = [f"clip{i}.mp4" for i in range(8)]
    metrics = pd.DataFrame({"distorted": names, "vmaf": [95.0] * 8, "psnr_y": [45.0] * 8})
    artifacts = pd.DataFrame(
        {
            "path": [f"out/{n}" for n in names],
            "banding_delta_max": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 4.0],
        }
    )
    # Every delta is below the absolute floor, but one is far above the rest.
    assert find_metric_blind_spots(metrics, artifacts).empty
    relative = find_metric_blind_spots(metrics, artifacts, banding_floor=None, banding_quantile=0.9)
    assert list(relative["distorted"]) == ["clip7.mp4"]


def test_find_metric_blind_spots_on_empty_input() -> None:
    import pandas as pd

    assert find_metric_blind_spots(pd.DataFrame(), pd.DataFrame()).empty


def test_contour_mask_agrees_with_the_score() -> None:
    # The mask is what the figure highlights, so it must be the same decision the number
    # is made of: no flagged pixels exactly when the score is zero.
    from pixeljudge.artifacts.detectors import contour_mask

    assert not contour_mask(smooth_gradient()).any()
    assert banding_score(smooth_gradient())[0] == pytest.approx(0.0)
    assert contour_mask(banded_gradient(levels=6)).any()
    assert banding_score(banded_gradient(levels=6))[0] > 0


def test_contour_mask_ignores_texture_and_empty_frames() -> None:
    from pixeljudge.artifacts.detectors import contour_mask

    rng = np.random.default_rng(4)
    assert not contour_mask(rng.normal(128, 25, (180, 320)).astype(np.float32)).any()
    assert contour_mask(np.zeros((0, 0), dtype=np.float32)).size == 0
