#!/usr/bin/env python
"""Generate tiny synthetic master clips with ffmpeg's own sources.

Why synthesise instead of downloading: the test suite and the demo should run on
a laptop with no network and no multi-gigabyte media checkout. These clips are a
few dozen kilobytes each and are generated deterministically, so a test that says
"the banding detector fires here" means the same thing on every machine.

Three clips, each chosen to stress a different part of the pipeline:

* ``gradient``     - a slow, low-contrast gradient. Smooth gradients are exactly
                     where quantisation produces banding, and where PSNR/SSIM/VMAF
                     are least likely to notice it.
* ``checkerboard`` - an 8-pixel checkerboard that drifts sideways. High-frequency
                     detail aligned to the coding grid, so it blocks badly at low
                     bitrate and gives the blocking detector something real.
* ``motion``       - ffmpeg's testsrc2 pattern: colour, edges, and movement, as a
                     general-purpose clip.

The masters are encoded with x264 at ``-qp 0`` (lossless for the luma/chroma we
feed it) so that later encodes are measured against a clean reference rather than
against something already compressed.

Usage:
    uv run python scripts/make_fixtures.py                     # tests/fixtures, 320x180
    uv run python scripts/make_fixtures.py --out data/raw --width 1280 --height 720
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script straight from a checkout without installing.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pixeljudge.errors import PixelJudgeError  # noqa: E402
from pixeljudge.io.ffmpeg import probe, run_ffmpeg  # noqa: E402
from pixeljudge.logging_conf import get_logger, setup_logging  # noqa: E402

log = get_logger("make_fixtures")

# Lossless master encode: no compression artifacts of our own in the reference.
MASTER_ARGS = ["-c:v", "libx264", "-qp", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-an"]


def gradient_source(width: int, height: int, fps: int, duration: float) -> str:
    """A slowly drifting dark gradient: the classic banding trap.

    Low contrast matters. A gradient spanning only a few code values has to be
    represented by a handful of levels after quantisation, and those levels show
    up as visible steps.
    """
    return (
        f"gradients=size={width}x{height}:rate={fps}:duration={duration}"
        ":c0=0x0d0d16:c1=0x2b2b3c:nb_colors=2:speed=0.02,format=yuv420p"
    )


def checkerboard_source(width: int, height: int, fps: int, duration: float) -> str:
    """8-pixel checkerboard drifting two pixels per frame.

    The 8-pixel period lines up with the transform block grid, which is what makes
    it a fair blocking test: any extra discontinuity on those boundaries came from
    the codec, not from the content.
    """
    expression = "if(mod(floor((X+2*N)/8)+floor(Y/8),2),235,16)"
    return (
        f"nullsrc=size={width}x{height}:rate={fps}:duration={duration}"
        f",format=yuv420p,geq=lum='{expression}':cb=128:cr=128"
    )


def motion_source(width: int, height: int, fps: int, duration: float) -> str:
    """ffmpeg's built-in test pattern: edges, colour and movement."""
    return f"testsrc2=size={width}x{height}:rate={fps}:duration={duration},format=yuv420p"


SOURCES = {
    "gradient": gradient_source,
    "checkerboard": checkerboard_source,
    "motion": motion_source,
}


def generate(
    name: str,
    out_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    duration: float,
    overwrite: bool,
) -> Path:
    destination = out_dir / f"{name}.mp4"
    if destination.exists() and not overwrite:
        log.info("skip (exists): %s", destination)
        return destination
    graph = SOURCES[name](width, height, fps, duration)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(["-f", "lavfi", "-i", graph, *MASTER_ARGS, str(destination)])
    info = probe(destination)
    log.info(
        "%s: %s @ %.2f fps, %d frames, %.1f kB",
        destination.name,
        info.resolution,
        info.fps,
        info.n_frames or -1,
        info.size_bytes / 1000,
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path("tests/fixtures"), help="output directory")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--duration", type=float, default=2.0, help="seconds")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only",
        choices=sorted(SOURCES),
        action="append",
        help="generate only this clip (repeatable)",
    )
    args = parser.parse_args(argv)
    setup_logging()

    names = args.only or sorted(SOURCES)
    try:
        for name in names:
            generate(
                name,
                args.out,
                width=args.width,
                height=args.height,
                fps=args.fps,
                duration=args.duration,
                overwrite=args.overwrite,
            )
    except PixelJudgeError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
