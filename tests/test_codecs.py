"""Encoder flag construction. No ffmpeg is invoked: these assert the command line.

The value of testing a command line is that encoder flags fail *quietly*. An
encode with a missing ``-b:v 0`` still produces a perfectly playable file, at
entirely the wrong bitrate, and you find out three plots later.
"""

from __future__ import annotations

import pytest

from pixeljudge.config import EncodeConfig, Rung
from pixeljudge.encode.codecs import (
    CODECS,
    build_encode_args,
    container_args,
    get_codec,
    gop_args,
    rate_args,
    scale_filter,
    speed_args,
)
from pixeljudge.errors import ConfigError


def test_every_codec_maps_to_an_encoder_and_container() -> None:
    assert CODECS["h264"].encoder == "libx264"
    assert CODECS["hevc"].encoder == "libx265"
    assert CODECS["vp9"].encoder == "libvpx-vp9"
    assert CODECS["av1"].encoder == "libsvtav1"
    # VP9 goes in WebM; the others in MP4.
    assert CODECS["vp9"].container == "webm"
    assert {CODECS[name].container for name in ("h264", "hevc", "av1")} == {"mp4"}


def test_unknown_codec_names_the_supported_ones() -> None:
    with pytest.raises(ConfigError, match="supported: av1, h264, hevc, vp9"):
        get_codec("mpeg2")


def test_vp9_crf_mode_includes_the_bitrate_zero_flag() -> None:
    # Without "-b:v 0" libvpx ignores -crf and silently targets a bitrate, so a
    # constant-quality sweep would not be one.
    args = rate_args(CODECS["vp9"], Rung(height=720, crf=32), EncodeConfig())
    assert args[:2] == ["-crf", "32"]
    assert args[2:] == ["-b:v", "0"]


def test_other_codecs_do_not_get_the_vp9_workaround() -> None:
    for name in ("h264", "hevc", "av1"):
        assert rate_args(CODECS[name], Rung(height=720, crf=32), EncodeConfig()) == [
            "-crf",
            "32",
        ]


def test_bitrate_mode_applies_a_rate_cap() -> None:
    # Streaming rungs are capped VBR, not free-running: -maxrate/-bufsize mirror
    # what a packager enforces.
    args = rate_args(CODECS["h264"], Rung(height=720, bitrate_kbps=3000), EncodeConfig())
    assert args == ["-b:v", "3000k", "-maxrate", "6000k", "-bufsize", "6000k"]


def test_rate_cap_multipliers_are_configurable() -> None:
    cfg = EncodeConfig(maxrate_multiplier=1.5, bufsize_multiplier=3.0)
    args = rate_args(CODECS["h264"], Rung(height=720, bitrate_kbps=2000), cfg)
    assert "3000k" in args and "6000k" in args


def test_crf_out_of_range_for_the_codec_is_refused() -> None:
    # 63 is valid for VP9/AV1 but not for x264, whose ceiling is 51.
    with pytest.raises(ConfigError, match="out of range"):
        rate_args(CODECS["h264"], Rung(height=720, crf=60), EncodeConfig())
    assert rate_args(CODECS["vp9"], Rung(height=720, crf=60), EncodeConfig())[:2] == [
        "-crf",
        "60",
    ]


def test_scale_filter_uses_minus_two_when_width_is_unset() -> None:
    # -2 preserves aspect ratio and rounds to an even width, which 4:2:0 requires.
    assert scale_filter(Rung(height=720, crf=20), EncodeConfig()) == (
        "scale=-2:720:flags=bicubic,setsar=1"
    )


def test_scale_filter_honours_an_explicit_width_and_flags() -> None:
    cfg = EncodeConfig(scale_flags="lanczos")
    assert scale_filter(Rung(height=432, width=768, crf=20), cfg) == (
        "scale=768:432:flags=lanczos,setsar=1"
    )


def test_speed_args_use_each_encoders_own_scale() -> None:
    cfg = EncodeConfig(preset="slow", av1_preset=6, vp9_cpu_used=4)
    assert speed_args(CODECS["h264"], cfg) == ["-preset", "slow"]
    # SVT-AV1's preset is a number, and higher means faster: the opposite kind of
    # scale from x264's words.
    assert speed_args(CODECS["av1"], cfg) == ["-preset", "6"]
    assert speed_args(CODECS["vp9"], cfg) == ["-cpu-used", "4", "-row-mt", "1"]


def test_x265_logging_is_quietened() -> None:
    # x265 prints per-frame stats to stderr, which would bury a real error in the
    # tail we surface on failure.
    assert "log-level=error" in " ".join(speed_args(CODECS["hevc"], EncodeConfig()))


def test_hevc_in_mp4_gets_the_hvc1_tag() -> None:
    assert container_args(CODECS["hevc"]) == ["-tag:v", "hvc1"]
    assert container_args(CODECS["h264"]) == []


def test_gop_is_expressed_in_frames_from_seconds() -> None:
    # A player can only switch rungs at a keyframe, so the GOP is a duration.
    assert gop_args(EncodeConfig(gop_seconds=2.0), 25.0) == ["-g", "50"]
    assert gop_args(EncodeConfig(gop_seconds=2.0), 29.97) == ["-g", "60"]


def test_gop_is_omitted_when_the_frame_rate_is_unknown() -> None:
    assert gop_args(EncodeConfig(), 0.0) == []


def test_full_argument_list_drops_audio_and_sets_pixel_format() -> None:
    args = build_encode_args(
        CODECS["h264"], Rung(height=720, bitrate_kbps=3000), EncodeConfig(), 25.0
    )
    assert "-an" in args  # nothing downstream looks at audio
    assert args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert args[0] == "-vf"
