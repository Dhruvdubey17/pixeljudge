"""Logical codec names to concrete ffmpeg encoder flags.

The rest of the code says "h264" or "av1" and never learns that those mean
``libx264`` and ``libsvtav1``. That indirection earns its keep in three places:
containers differ (VP9 goes in WebM), constant-quality mode is spelled
differently per encoder, and speed presets are on completely different scales.

The traps encoded here, so nobody has to rediscover them:

* libvpx-vp9 ignores ``-crf`` unless ``-b:v 0`` is also passed. Without it you
  silently get a bitrate-targeted encode and your "constant quality" sweep is a
  lie.
* SVT-AV1's ``-preset`` is a number (0 slowest ... 13 fastest), the opposite kind
  of scale from x264/x265's word presets.
* HEVC in MP4 needs ``-tag:v hvc1`` or QuickTime refuses to play the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import CodecName, EncodeConfig, Rung
from ..errors import ConfigError


@dataclass(frozen=True)
class CodecSpec:
    """What ffmpeg needs to know about one logical codec."""

    name: CodecName
    encoder: str
    container: str  # file extension, without the dot
    crf_flag: str  # the flag that means "constant quality" for this encoder
    max_crf: int
    notes: str = ""


CODECS: dict[CodecName, CodecSpec] = {
    "h264": CodecSpec(
        name="h264",
        encoder="libx264",
        container="mp4",
        crf_flag="-crf",
        max_crf=51,
        notes="Universal baseline; plays on everything.",
    ),
    "hevc": CodecSpec(
        name="hevc",
        encoder="libx265",
        container="mp4",
        crf_flag="-crf",
        max_crf=51,
        notes="Roughly 50% more efficient than H.264 at the cost of encode time.",
    ),
    "vp9": CodecSpec(
        name="vp9",
        encoder="libvpx-vp9",
        container="webm",
        crf_flag="-crf",
        max_crf=63,
        notes="Royalty-free; needs -b:v 0 for constant-quality mode.",
    ),
    "av1": CodecSpec(
        name="av1",
        encoder="libsvtav1",
        container="mp4",
        crf_flag="-crf",
        max_crf=63,
        notes="Most efficient of the four, by far the slowest to encode.",
    ),
}


def get_codec(name: str) -> CodecSpec:
    try:
        return CODECS[name]  # type: ignore[index]
    except KeyError as exc:
        raise ConfigError(
            f"unknown codec {name!r}; supported: {', '.join(sorted(CODECS))}"
        ) from exc


def scale_filter(rung: Rung, cfg: EncodeConfig) -> str:
    """Build the scale filter for one rung.

    A rung with no explicit width uses ``-2``, which asks ffmpeg to preserve the
    source aspect ratio and round to an even width (required by 4:2:0 chroma
    subsampling). ``setsar=1`` stops a non-square source pixel aspect from
    following the scaled output around.
    """
    width = rung.width if rung.width is not None else -2
    return f"scale={width}:{rung.height}:flags={cfg.scale_flags},setsar=1"


def rate_args(spec: CodecSpec, rung: Rung, cfg: EncodeConfig) -> list[str]:
    """Flags that set the quality/rate target for one rung."""
    if rung.crf is not None:
        if rung.crf > spec.max_crf:
            raise ConfigError(
                f"crf {rung.crf} is out of range for {spec.encoder} (max {spec.max_crf})"
            )
        args = [spec.crf_flag, str(rung.crf)]
        if spec.name == "vp9":
            # The flag that makes -crf mean anything at all for libvpx.
            args += ["-b:v", "0"]
        return args

    assert rung.bitrate_kbps is not None  # guaranteed by Rung's validator
    bitrate = rung.bitrate_kbps
    maxrate = int(bitrate * cfg.maxrate_multiplier)
    bufsize = int(bitrate * cfg.bufsize_multiplier)
    return [
        "-b:v",
        f"{bitrate}k",
        "-maxrate",
        f"{maxrate}k",
        "-bufsize",
        f"{bufsize}k",
    ]


def speed_args(spec: CodecSpec, cfg: EncodeConfig) -> list[str]:
    """Encoder speed/efficiency knobs, per encoder's own scale."""
    if spec.name in {"h264", "hevc"}:
        args = ["-preset", cfg.preset]
        if spec.name == "hevc":
            # x265 prints a banner and per-frame stats to stderr; quiet it so a
            # real error is visible in the tail we surface on failure.
            args += ["-x265-params", "log-level=error"]
        return args
    if spec.name == "vp9":
        return ["-cpu-used", str(cfg.vp9_cpu_used), "-row-mt", "1"]
    return ["-preset", str(cfg.av1_preset)]


def container_args(spec: CodecSpec) -> list[str]:
    if spec.name == "hevc":
        return ["-tag:v", "hvc1"]
    return []


def gop_args(cfg: EncodeConfig, fps: float) -> list[str]:
    """Keyframe interval in frames.

    Streaming needs a keyframe at every segment boundary so a player can switch
    rungs, so the GOP is expressed in seconds and converted here.
    """
    if fps <= 0:
        return []
    return ["-g", str(max(1, round(fps * cfg.gop_seconds)))]


def build_encode_args(
    spec: CodecSpec,
    rung: Rung,
    cfg: EncodeConfig,
    fps: float,
) -> list[str]:
    """Every flag after the input and before the output path."""
    return [
        "-vf",
        scale_filter(rung, cfg),
        "-c:v",
        spec.encoder,
        *rate_args(spec, rung, cfg),
        *speed_args(spec, cfg),
        *gop_args(cfg, fps),
        "-pix_fmt",
        cfg.pix_fmt,
        *container_args(spec),
        # Quality work only ever looks at the video; audio would just cost bits
        # and confuse the measured bitrate.
        "-an",
    ]
