"""Thin, well-behaved wrappers around the ffmpeg and ffprobe binaries.

We shell out rather than use a Python binding because libvmaf is a *compile-time*
feature of ffmpeg: the filter either exists in the installed build or it does
not, and no Python package can change that. Keeping every subprocess call in one
module means the rest of the codebase never has to think about quoting, exit
codes, or where the binary lives.

Two conventions worth knowing:

* ffmpeg writes its progress and its errors to stderr, and a failure message is
  almost always in the *last* few lines. On failure we surface that tail instead
  of the whole log.
* Binary discovery honours ``PIXELJUDGE_FFMPEG`` / ``PIXELJUDGE_FFPROBE`` so a
  custom build (say, one you compiled with libvmaf yourself) can be used without
  touching PATH.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..errors import FfmpegError, MissingDependencyError
from ..logging_conf import get_logger

log = get_logger(__name__)

INSTALL_HINT = (
    "Install an ffmpeg built with libvmaf:\n"
    "  macOS:   brew install ffmpeg\n"
    "  Ubuntu:  apt-get install ffmpeg   (or use a static build / the project Dockerfile)\n"
    "  Windows: use a gyan.dev 'full' build, or run the Dockerfile\n"
    "Verify with:  ffmpeg -hide_banner -filters | grep libvmaf\n"
    "A custom build can be pointed at with PIXELJUDGE_FFMPEG=/path/to/ffmpeg."
)


@dataclass(frozen=True)
class VideoInfo:
    """The handful of stream properties the pipeline actually needs."""

    path: Path
    width: int
    height: int
    fps: float
    codec: str
    pix_fmt: str
    duration_s: float
    n_frames: int | None
    size_bytes: int

    @property
    def bitrate_kbps(self) -> float:
        """Actual delivered bitrate, measured from file size rather than trusted
        from the container header (headers lie, especially after remuxing)."""
        if self.duration_s <= 0:
            return 0.0
        return self.size_bytes * 8 / self.duration_s / 1000.0

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def _find_binary(name: str, env_var: str) -> str:
    override = os.getenv(env_var)
    if override:
        if not (Path(override).is_file() and os.access(override, os.X_OK)):
            raise MissingDependencyError(f"{env_var}={override} is not an executable file")
        return override
    found = shutil.which(name)
    if found is None:
        raise MissingDependencyError(f"{name} was not found on PATH.\n{INSTALL_HINT}")
    return found


def ffmpeg_binary() -> str:
    return _find_binary("ffmpeg", "PIXELJUDGE_FFMPEG")


def ffprobe_binary() -> str:
    return _find_binary("ffprobe", "PIXELJUDGE_FFPROBE")


def ffmpeg_available() -> bool:
    """True when both binaries can be located. Used to skip integration tests."""
    try:
        ffmpeg_binary()
        ffprobe_binary()
    except MissingDependencyError:
        return False
    return True


def run(
    args: list[str],
    *,
    timeout: float | None = None,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a command, raising :class:`FfmpegError` on a non-zero exit.

    ``args[0]`` is the binary. We never use ``shell=True``: paths with spaces are
    common in media work and a shell would only add a quoting bug.
    """
    log.debug("run: %s", " ".join(args))
    try:
        proc = subprocess.run(  # noqa: S603 - argument list, no shell
            args,
            check=False,
            text=True,
            timeout=timeout,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        raise MissingDependencyError(f"{args[0]} could not be executed.\n{INSTALL_HINT}") from exc
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"{Path(args[0]).name} timed out after {timeout}s") from exc

    if proc.returncode != 0:
        tail = _stderr_tail(proc.stderr)
        raise FfmpegError(
            f"{Path(args[0]).name} exited with code {proc.returncode}.\n"
            f"command: {' '.join(args)}\n{tail}"
        )
    return proc


def _stderr_tail(stderr: str | None, lines: int = 12) -> str:
    if not stderr:
        return "(no stderr captured)"
    tail = [line for line in stderr.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(tail)


def run_ffmpeg(
    args: list[str], *, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run ffmpeg with the flags we always want.

    ``-nostdin`` matters: without it ffmpeg competes for the terminal and can
    swallow keystrokes when several encodes run in a row. ``-y`` overwrites, so
    re-running a phase is idempotent.
    """
    return run([ffmpeg_binary(), "-hide_banner", "-nostdin", "-y", *args], timeout=timeout)


@lru_cache(maxsize=1)
def ffmpeg_version() -> str:
    """Version string, e.g. ``8.1.2`` (or the raw first line if unparseable)."""
    first_line = run([ffmpeg_binary(), "-hide_banner", "-version"]).stdout.splitlines()[0]
    match = re.search(r"ffmpeg version (\S+)", first_line)
    return match.group(1) if match else first_line.strip()


@lru_cache(maxsize=1)
def ffprobe_version() -> str:
    first_line = run([ffprobe_binary(), "-hide_banner", "-version"]).stdout.splitlines()[0]
    match = re.search(r"ffprobe version (\S+)", first_line)
    return match.group(1) if match else first_line.strip()


@lru_cache(maxsize=1)
def has_libvmaf() -> bool:
    """Whether the installed ffmpeg exposes the libvmaf filter.

    Checked by listing filters rather than by trying a measurement, so the answer
    costs milliseconds and can be printed by ``doctor``.
    """
    try:
        listing = run([ffmpeg_binary(), "-hide_banner", "-filters"]).stdout
    except (FfmpegError, MissingDependencyError):
        return False
    return any(line.split()[1:2] == ["libvmaf"] for line in listing.splitlines() if line.strip())


def require_libvmaf() -> None:
    """Fail early with an actionable message when libvmaf is missing."""
    if not has_libvmaf():
        raise MissingDependencyError(
            "this ffmpeg build has no libvmaf filter, so VMAF cannot be computed.\n" + INSTALL_HINT
        )


@lru_cache(maxsize=1)
def available_encoders() -> frozenset[str]:
    """Encoder names the installed ffmpeg can use (``libx264``, ``libsvtav1``, ...)."""
    try:
        listing = run([ffmpeg_binary(), "-hide_banner", "-encoders"]).stdout
    except (FfmpegError, MissingDependencyError):
        return frozenset()
    names: set[str] = set()
    for line in listing.splitlines():
        parts = line.split()
        # Encoder lines look like: " V....D libx264   H.264 ... "
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


@lru_cache(maxsize=1)
def libvmaf_models() -> tuple[str, ...]:
    """VMAF model versions this build recognises.

    libvmaf ships its models compiled in, and there is no flag that lists them,
    so we probe by name: ask ffmpeg to build a filtergraph with each model and
    see whether it complains. Cheap (no frames are decoded) and honest.
    """
    from ..config import KNOWN_VMAF_MODELS

    if not has_libvmaf():
        return ()
    found: list[str] = []
    for name in KNOWN_VMAF_MODELS:
        args = [
            ffmpeg_binary(),
            "-hide_banner",
            "-nostdin",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=1:duration=0.1",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=1:duration=0.1",
            "-lavfi",
            f"libvmaf=model=version={name}",
            "-f",
            "null",
            "-",
        ]
        try:
            run(args, timeout=120)
        except (FfmpegError, MissingDependencyError):
            continue
        found.append(name)
    return tuple(found)


def probe(path: Path | str) -> VideoInfo:
    """Read stream properties of the first video stream.

    Raises :class:`FfmpegError` for a missing file, a file with no video stream,
    or ffprobe output we cannot make sense of.
    """
    path = Path(path)
    if not path.exists():
        raise FfmpegError(f"cannot probe missing file: {path}")

    args = [
        ffprobe_binary(),
        "-hide_banner",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(path),
    ]
    raw = run(args).stdout
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FfmpegError(f"ffprobe returned unparseable JSON for {path}") from exc

    streams = payload.get("streams") or []
    if not streams:
        raise FfmpegError(f"{path} contains no video stream")
    stream = streams[0]
    fmt = payload.get("format") or {}

    try:
        width = int(stream["width"])
        height = int(stream["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FfmpegError(f"ffprobe gave no usable dimensions for {path}") from exc

    duration = _first_float(stream.get("duration"), fmt.get("duration"))
    size_bytes = int(fmt.get("size") or path.stat().st_size)
    fps = _parse_rational(stream.get("avg_frame_rate")) or _parse_rational(
        stream.get("r_frame_rate")
    )

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps or 0.0,
        codec=str(stream.get("codec_name", "unknown")),
        pix_fmt=str(stream.get("pix_fmt", "unknown")),
        duration_s=duration or 0.0,
        n_frames=_parse_frame_count(stream.get("nb_frames"), duration, fps),
        size_bytes=size_bytes,
    )


def _first_float(*candidates: object) -> float | None:
    for candidate in candidates:
        try:
            value = float(str(candidate))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _parse_rational(value: object) -> float | None:
    """ffprobe reports frame rates as ``"30000/1001"``. ``0/0`` means unknown."""
    if not isinstance(value, str) or "/" not in value:
        return _first_float(value)
    num, _, den = value.partition("/")
    try:
        numerator, denominator = float(num), float(den)
    except ValueError:
        return None
    if denominator == 0:
        return None
    return numerator / denominator


def _parse_frame_count(raw: object, duration: float | None, fps: float | None) -> int | None:
    """nb_frames is often absent or "N/A"; fall back to duration * fps."""
    try:
        count = int(str(raw))
        if count > 0:
            return count
    except (TypeError, ValueError):
        pass
    if duration and fps:
        return int(round(duration * fps))
    return None


def environment_report() -> dict[str, str]:
    """Human-readable environment summary, used by ``pixeljudge doctor`` and
    recorded alongside every metric result so numbers stay traceable."""
    report: dict[str, str] = {}
    try:
        report["ffmpeg"] = ffmpeg_binary()
        report["ffmpeg_version"] = ffmpeg_version()
    except (MissingDependencyError, FfmpegError) as exc:
        report["ffmpeg"] = f"MISSING ({exc})"
        return report
    try:
        report["ffprobe"] = ffprobe_binary()
        report["ffprobe_version"] = ffprobe_version()
    except (MissingDependencyError, FfmpegError) as exc:
        report["ffprobe"] = f"MISSING ({exc})"
    report["libvmaf"] = "yes" if has_libvmaf() else "no"
    encoders = available_encoders()
    for logical, encoder in (
        ("h264", "libx264"),
        ("hevc", "libx265"),
        ("vp9", "libvpx-vp9"),
        ("av1", "libsvtav1"),
    ):
        report[f"encoder_{logical}"] = encoder if encoder in encoders else "missing"
    return report
