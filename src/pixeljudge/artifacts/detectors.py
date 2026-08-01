"""Artifact detectors: looking for the damage the pooled metrics do not see.

PSNR, SSIM and VMAF all answer "how far is this frame from the original", and
they answer it well on average. None of them is *designed* to answer "does this
frame have visible banding in the sky". Banding moves very few pixels by very
little, so squared-error-based measures barely register it while a human sees
rings immediately. Netflix hit the same wall and built a dedicated index (CAMBI)
for it.

These three detectors are deliberately simple, no-reference proxies. They are not
CAMBI and they are not perceptual models. They are cheap, explainable signals
that flag frames worth looking at, which is what makes the "metrics said fine,
the eye says otherwise" comparison possible at all.

Each detector returns a *higher is worse* score:

* ``blur_score``     - inverse sharpness from the variance of the Laplacian
* ``blocking_score`` - extra discontinuity on the 8-pixel coding grid
* ``banding_score``  - contrast-weighted count of visible steps in quiet neighbourhoods

All three read the luma plane only. Chroma artifacts exist (colour bleeding), but
luma is where the eye's sensitivity and the codec's bit budget both concentrate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Transform blocks in H.264/HEVC/VP9/AV1 all include an 8x8 grid, so 8 is the
# period a blocking artifact is most likely to sit on.
BLOCK_SIZE = 8

# Below a quarter of a code value nothing is visible; used as a denominator floor
# so ratios stay finite on synthetic or heavily flattened content.
OFF_GRID_FLOOR = 0.25
FLAT_EPSILON = 1e-6

# A contour has to step by about this many code values before a viewer notices it.
# Smaller steps are what an ordinary 8-bit gradient looks like anyway: a 1-level
# staircase is not banding, it is 8-bit. This threshold is the whole reason the
# detector can tell a quantised gradient from a smooth one.
MIN_VISIBLE_STEP = 2.0
# Above this, the step is an edge in the content rather than a contour.
MAX_CONTOUR_STEP = 12.0

# Mean gradient magnitude allowed in the 9x9 window around a contour. This is the test
# that separates a contour from texture: around a real contour the window is mostly
# plateau, so the average stays low even though the centre pixel is a step. Measured
# values for context: a banded gradient sits at 0.2, a 145 kbit/s photographic frame at
# 6, random noise at 22, a hard checkerboard at 90.
MAX_LOCAL_GRADIENT = 2.5


@dataclass(frozen=True)
class ArtifactScores:
    """All detector outputs for one frame. Higher is worse, for every field."""

    blur: float
    blocking: float
    banding: float
    flat_fraction: float  # how much of the frame was smooth enough to band

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def to_luma(frame: np.ndarray) -> np.ndarray:
    """Return the luma plane as float32.

    OpenCV hands us BGR; the conversion weights match Rec.601 luma, which is what
    the 'Y' in yuv420p means.
    """
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame.astype(np.float32)


def blur_score(frame: np.ndarray) -> float:
    """Blur as the inverse of Laplacian variance.

    The Laplacian is a second-derivative filter: it is large wherever intensity
    changes sharply. A sharp image therefore has a wide spread of Laplacian
    responses, and a blurred one has almost none. Variance captures that spread
    in a single number, and it is the standard cheap sharpness measure.

    We invert it (and scale so numbers land in a readable range) because every
    detector here follows "higher is worse".
    """
    luma = to_luma(frame)
    variance = float(cv2.Laplacian(luma, cv2.CV_32F).var())
    return float(1000.0 / (variance + 1.0))


def sharpness(frame: np.ndarray) -> float:
    """Raw Laplacian variance, exposed because it is easier to reason about."""
    return float(cv2.Laplacian(to_luma(frame), cv2.CV_32F).var())


def blocking_score(frame: np.ndarray, block: int = BLOCK_SIZE) -> float:
    """Excess discontinuity on the coding grid, relative to everywhere else.

    The idea: measure the average absolute difference between neighbouring pixel
    columns *on* the block boundaries (x = 8, 16, 24 ...) and compare it with the
    same quantity on the columns in between. Natural content has no reason to
    prefer multiples of eight, so a ratio above 1 is evidence that the codec's
    block structure has become visible. Rows are treated the same way and the two
    directions are averaged.

    Returning a ratio rather than an absolute difference is what makes the score
    comparable across clips: a busy frame has large gradients everywhere, and only
    the *relative* excess on the grid is an artifact.
    """
    luma = to_luma(frame)
    if luma.shape[0] < 2 * block or luma.shape[1] < 2 * block:
        return 0.0

    vertical = np.abs(np.diff(luma, axis=1))  # differences between columns
    horizontal = np.abs(np.diff(luma, axis=0))  # differences between rows

    # diff index i holds |x[i+1] - x[i]|, so a boundary at column b appears at i = b-1.
    v_index = np.arange(vertical.shape[1])
    h_index = np.arange(horizontal.shape[0])
    v_on = (v_index + 1) % block == 0
    h_on = (h_index + 1) % block == 0

    ratios: list[float] = []
    for axis, (diff, on_grid) in enumerate(((horizontal, h_on), (vertical, v_on))):
        if not on_grid.any() or on_grid.all():
            continue
        on_mean = float(diff.take(np.flatnonzero(on_grid), axis=axis).mean())
        off_mean = float(diff.take(np.flatnonzero(~on_grid), axis=axis).mean())
        if on_mean <= FLAT_EPSILON and off_mean <= FLAT_EPSILON:
            # A genuinely flat frame has no discontinuity anywhere to compare.
            ratios.append(1.0)
            continue
        # Floor the denominator instead of special-casing it. Content that is
        # perfectly smooth between block boundaries (a checkerboard, or a heavily
        # quantised flat area) would otherwise divide by zero, and returning the
        # "no artifact" value of 1.0 there would hide the strongest possible
        # blocking signal. A quarter of a code value is well below visibility, so
        # the resulting ratio is large but finite and still means what it says.
        ratios.append(on_mean / max(off_mean, OFF_GRID_FLOOR))
    if not ratios:
        return 0.0
    return float(np.mean(ratios))


def banding_score(
    frame: np.ndarray,
    *,
    flat_threshold: float = 2.0,
    step_range: tuple[float, float] = (MIN_VISIBLE_STEP, MAX_CONTOUR_STEP),
    neighbourhood: int = 9,
) -> tuple[float, float]:
    """Detect contouring in smooth areas. Returns ``(score, flat_fraction)``.

    Banding is a specific shape of damage: inside a region that *should* be a smooth
    ramp, the values collapse onto a few discrete levels, so the gradient becomes zero
    across wide plateaus separated by small, sharp steps. A pixel is counted as a
    contour when all three of these hold:

    1. **The step is big enough to see.** Below ``MIN_VISIBLE_STEP`` we are looking at
       what *any* 8-bit gradient does: a ramp spanning 32 code values across 320 pixels
       has to change by one somewhere. Counting those would score a pristine gradient
       as badly as a wrecked one. Only steps the eye can resolve count, which is the
       contrast-awareness that puts the "C" in CAMBI.
    2. **The step is not an edge.** Above ``MAX_CONTOUR_STEP`` it is content.
    3. **The neighbourhood is quiet.** The *mean* gradient magnitude in a 9x9 window
       must be below ``MAX_LOCAL_GRADIENT``. This is what separates a contour from
       texture: around a contour the window is mostly plateau, so the mean gradient
       stays low even though the pixel itself is a step. In texture, every pixel is a
       step and the mean is high.

    The score is ``100 * sum(step_magnitude^2) / total_pixels``. Two decisions in that
    formula, both of which were arrived at the hard way:

    * **Squaring** the step size. The total variation across a ramp is fixed, so
      weighting linearly gives a 6-level staircase and a 12-level one identical scores
      even though the coarser one is far more obvious. Squaring makes one large jump
      count for more than several small ones, which is the direction visibility goes.
    * Normalising by **total pixels**, not by smooth area. Dividing by the smooth area
      inverts the detector on real content: a frame that is 3% smooth has a tiny
      denominator, so a handful of qualifying pixels produce an enormous score, and
      heavily textured clips outrank actual skies. The measured consequence was Big
      Buck Bunny scoring ~1000 while a visibly banded gradient scored 5.

    ``flat_fraction`` is returned alongside as a diagnostic: it is the share of the
    frame whose local standard deviation is below ``flat_threshold``, i.e. how much
    smooth area there was to band in the first place. A frame of pure texture cannot
    band, and knowing that is different from scoring it.
    """
    luma = to_luma(frame)
    if luma.size == 0:
        return 0.0, 0.0

    # Local standard deviation via box filters (E[x^2] - E[x]^2), for the diagnostic.
    kernel = (neighbourhood, neighbourhood)
    local_mean = cv2.blur(luma, kernel)
    local_mean_sq = cv2.blur(luma * luma, kernel)
    local_var = np.clip(local_mean_sq - local_mean * local_mean, 0.0, None)
    flat_fraction = float((np.sqrt(local_var) < flat_threshold).mean())

    # Gradient magnitude from a 3x3 Sobel, which is less noise-sensitive than a raw
    # pixel difference on the near-flat data we care about. Sobel's taps sum to 4 per
    # direction, so divide to get back to code values.
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2) / 4.0
    local_gradient = cv2.blur(magnitude, kernel)

    low, high = step_range
    mask = (magnitude >= low) & (magnitude <= high) & (local_gradient < MAX_LOCAL_GRADIENT)
    weighted = float((magnitude[mask] ** 2).sum())
    return float(100.0 * weighted / float(magnitude.size)), flat_fraction


def contour_mask(frame: np.ndarray, *, neighbourhood: int = 9) -> np.ndarray:
    """Which pixels :func:`banding_score` counted as contours.

    Exposed so a figure can show *where* the detector fired instead of asking a reader
    to take a single number on trust. Same three conditions, same constants.
    """
    luma = to_luma(frame)
    if luma.size == 0:
        return np.zeros_like(luma, dtype=bool)
    grad_x = cv2.Sobel(luma, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(luma, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(grad_x**2 + grad_y**2) / 4.0
    local_gradient = cv2.blur(magnitude, (neighbourhood, neighbourhood))
    mask: np.ndarray = (
        (magnitude >= MIN_VISIBLE_STEP)
        & (magnitude <= MAX_CONTOUR_STEP)
        & (local_gradient < MAX_LOCAL_GRADIENT)
    )
    return mask


def score_frame(frame: np.ndarray) -> ArtifactScores:
    """Run every detector on one frame."""
    banding, flat_fraction = banding_score(frame)
    return ArtifactScores(
        blur=blur_score(frame),
        blocking=blocking_score(frame),
        banding=banding,
        flat_fraction=flat_fraction,
    )


# ---------------------------------------------------------------------------
# Synthetic images. These exist so the detectors can be tested against content
# whose ground truth is known by construction, and so the docs can show what each
# detector is actually looking at.
# ---------------------------------------------------------------------------


def smooth_gradient(
    width: int = 320, height: int = 180, low: int = 16, high: int = 48
) -> np.ndarray:
    """A continuous horizontal ramp, computed in float and left un-quantised."""
    ramp = np.linspace(low, high, width, dtype=np.float32)
    return np.tile(ramp, (height, 1))


def banded_gradient(
    width: int = 320,
    height: int = 180,
    low: int = 16,
    high: int = 48,
    levels: int = 6,
) -> np.ndarray:
    """The same ramp forced onto ``levels`` discrete steps.

    This is what quantisation does to a gradient, and it is the reference case for
    the banding detector: identical content, identical range, only the number of
    available levels differs.
    """
    smooth = smooth_gradient(width, height, low, high)
    span = max(high - low, 1)
    quantised = np.round((smooth - low) / span * (levels - 1)) / (levels - 1)
    return (quantised * span + low).astype(np.float32)


def checkerboard(width: int = 320, height: int = 180, block: int = BLOCK_SIZE) -> np.ndarray:
    """Hard 8-pixel checkerboard: maximum discontinuity exactly on the grid."""
    ys, xs = np.mgrid[0:height, 0:width]
    pattern = ((xs // block) + (ys // block)) % 2
    return np.where(pattern == 0, 16.0, 235.0).astype(np.float32)
