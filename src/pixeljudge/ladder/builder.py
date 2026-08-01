"""Ladders: the fixed table, and the per-title convex hull.

A fixed ladder is one bitrate table applied to every title. It is simple and it
is wasteful: a talking-head interview and a firework display do not need the same
bitrate to look the same. Per-title encoding fixes that by measuring the title.

The per-title recipe implemented here is the published Netflix approach, reduced
to its essentials:

1. Encode a *grid* of (resolution, quality) probe points.
2. Measure VMAF and the real bitrate of each point.
3. Plot quality against bitrate. Many points are dominated: some other point is
   cheaper *and* better. Keep only the upper convex hull, which is the frontier
   of "best quality available at this bitrate".
4. Stop climbing once quality reaches the point of diminishing returns (a top
   rung around VMAF 93-95; past that viewers cannot tell and the bits are wasted).
5. Space the final rungs out so a player has meaningful steps to switch between.

The hull maths is pure and unit-tested; nothing here touches ffmpeg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..config import CodecName, LadderConfig, Rung
from ..errors import ConfigError
from ..logging_conf import get_logger

log = get_logger(__name__)

# Default probe grid. Kept small on purpose: the grid is encoded *and* measured,
# so a 4x5 grid is already 20 encodes plus 20 VMAF passes per title.
DEFAULT_PROBE_HEIGHTS = (360, 540, 720, 1080)
DEFAULT_PROBE_CRFS = (20, 26, 32, 38, 44)

# Above this VMAF, extra bitrate buys nothing a viewer can see.
DEFAULT_TARGET_VMAF = 94.0


@dataclass(frozen=True)
class RdPoint:
    """One measured (bitrate, quality) point, tagged with how it was produced."""

    height: int
    bitrate_kbps: float
    vmaf: float
    crf: int | None = None
    width: int | None = None
    label: str | None = None

    @property
    def log_bitrate(self) -> float:
        # Rate-distortion behaviour is roughly logarithmic in bitrate, so the hull
        # is computed in log space. Guard against a zero-length/empty encode.
        if self.bitrate_kbps <= 0:
            raise ConfigError(f"non-positive bitrate in RD point: {self}")
        return math.log10(self.bitrate_kbps)


def probe_grid(
    codec: CodecName,
    *,
    heights: tuple[int, ...] = DEFAULT_PROBE_HEIGHTS,
    crfs: tuple[int, ...] = DEFAULT_PROBE_CRFS,
    source_height: int | None = None,
    name: str = "pertitle_probe",
) -> LadderConfig:
    """The grid of test encodes a per-title ladder is chosen from.

    Rungs above the master's own height are dropped: upscaling invents no detail,
    so such a point can never be on the useful part of the hull.
    """
    usable = [h for h in heights if source_height is None or h <= source_height]
    if not usable:
        raise ConfigError(f"no probe height fits a {source_height}p source; heights were {heights}")
    rungs = [Rung(height=h, crf=crf) for h in usable for crf in crfs]
    return LadderConfig(
        name=name,
        codec=codec,
        description="Probe grid for per-title convex-hull ladder selection",
        rungs=rungs,
    )


def upper_convex_hull(points: list[RdPoint]) -> list[RdPoint]:
    """Keep only the points on the upper convex hull of quality vs log-bitrate.

    "Upper hull" is the geometric way of saying "the best deal available": a point
    is dropped if some combination of cheaper and better points dominates it.
    Implemented as a monotone chain, which is O(n log n) and easy to read.

    Ties in bitrate are resolved by keeping the higher-quality point, so a
    resolution that wastes bits at the same rate never survives.
    """
    if not points:
        return []

    # Sort by cost, then by quality descending so the best of a bitrate tie leads.
    ordered = sorted(points, key=lambda p: (p.log_bitrate, -p.vmaf))
    deduped: list[RdPoint] = []
    for point in ordered:
        if deduped and math.isclose(point.log_bitrate, deduped[-1].log_bitrate, rel_tol=1e-12):
            continue  # same bitrate, lower quality: dominated
        deduped.append(point)

    hull: list[RdPoint] = []
    for point in deduped:
        # Drop the previous point while it sits below the straight line from the
        # one before it to the new one: that means it is not on the frontier.
        while len(hull) >= 2 and _cross(hull[-2], hull[-1], point) >= 0:
            hull.pop()
        hull.append(point)

    # A frontier must not go downhill: a point that costs more and scores less
    # than its predecessor is never worth shipping.
    monotone: list[RdPoint] = []
    for point in hull:
        if monotone and point.vmaf <= monotone[-1].vmaf:
            continue
        monotone.append(point)
    return monotone


def _cross(a: RdPoint, b: RdPoint, c: RdPoint) -> float:
    """Z component of (b-a) x (c-a) in (log-bitrate, VMAF) space.

    Positive means a->b->c turns left, i.e. b lies below the a-c line and is not
    on the upper hull.
    """
    return (b.log_bitrate - a.log_bitrate) * (c.vmaf - a.vmaf) - (b.vmaf - a.vmaf) * (
        c.log_bitrate - a.log_bitrate
    )


def trim_to_target_quality(
    hull: list[RdPoint], target_vmaf: float = DEFAULT_TARGET_VMAF
) -> list[RdPoint]:
    """Stop the ladder at the first point that reaches ``target_vmaf``.

    Everything above it is bits spent on quality the viewer cannot see. If no
    point reaches the target we keep the whole hull: the title needs more bitrate
    than the grid offered, and silently returning a short ladder would hide that.
    """
    for index, point in enumerate(hull):
        if point.vmaf >= target_vmaf:
            return hull[: index + 1]
    if hull:
        log.warning(
            "no probe point reached VMAF %.1f (best was %.2f); the probe grid may need "
            "a higher-quality corner",
            target_vmaf,
            hull[-1].vmaf,
        )
    return hull


def select_rungs(hull: list[RdPoint], n_rungs: int) -> list[RdPoint]:
    """Thin the hull down to ``n_rungs``, spaced evenly in log-bitrate.

    Even spacing in log space is what an ABR player wants: each step up roughly
    doubles the bandwidth demand, so the switching decisions are meaningful.
    """
    if n_rungs < 1:
        raise ConfigError("a ladder needs at least one rung")
    if len(hull) <= n_rungs:
        return list(hull)

    low, high = hull[0].log_bitrate, hull[-1].log_bitrate
    targets = [low + (high - low) * i / (n_rungs - 1) for i in range(n_rungs)]
    chosen: list[RdPoint] = []
    for target in targets:
        nearest = min(hull, key=lambda p: abs(p.log_bitrate - target))
        if nearest not in chosen:
            chosen.append(nearest)
    return sorted(chosen, key=lambda p: p.log_bitrate)


def per_title_ladder(
    points: list[RdPoint],
    *,
    codec: CodecName,
    name: str = "per_title",
    n_rungs: int = 5,
    target_vmaf: float = DEFAULT_TARGET_VMAF,
) -> LadderConfig:
    """Full per-title selection: hull, quality cap, spacing, then a ladder config.

    The output rungs are expressed as *bitrate* targets even though the probes
    were CRF encodes, because that is what a packager ships. The bitrate used is
    the one the probe encode actually achieved.
    """
    if not points:
        raise ConfigError("per-title ladder needs at least one measured RD point")
    hull = trim_to_target_quality(upper_convex_hull(points), target_vmaf)
    selected = select_rungs(hull, n_rungs)
    log.info(
        "per-title ladder: %d probe points -> %d on the hull -> %d rungs",
        len(points),
        len(hull),
        len(selected),
    )
    rungs = [
        Rung(
            height=point.height,
            width=point.width,
            bitrate_kbps=max(1, int(round(point.bitrate_kbps))),
            label=f"{point.height}p_{int(round(point.bitrate_kbps))}k",
        )
        for point in selected
    ]
    return LadderConfig(
        name=name,
        codec=codec,
        description=(
            f"Per-title convex-hull ladder, target VMAF {target_vmaf:g}, "
            f"selected from {len(points)} probe encodes"
        ),
        rungs=rungs,
    )


def ladder_to_yaml_dict(ladder: LadderConfig) -> dict[str, Any]:
    """Plain dict for dumping a generated ladder next to the handwritten ones."""
    return {
        "name": ladder.name,
        "codec": ladder.codec,
        "description": ladder.description,
        "rungs": [
            {
                key: value
                for key, value in (
                    ("width", rung.width),
                    ("height", rung.height),
                    ("bitrate_kbps", rung.bitrate_kbps),
                    ("crf", rung.crf),
                    ("label", rung.label),
                )
                if value is not None
            }
            for rung in ladder.rungs
        ],
    }
