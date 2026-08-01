"""Exception types for the whole package.

Every failure the user can plausibly cause (a bad config, a missing binary, an
ffmpeg build without libvmaf, a log we cannot parse) is raised as one of these
instead of leaking a raw ``CalledProcessError`` or ``KeyError``. The CLI catches
``PixelJudgeError`` at the top level and prints a single clear line, so a typo
never turns into a wall of traceback.
"""

from __future__ import annotations


class PixelJudgeError(Exception):
    """Base class for every error this project raises on purpose."""


class ConfigError(PixelJudgeError):
    """A YAML/config file is missing, malformed, or fails validation."""


class FfmpegError(PixelJudgeError):
    """An ffmpeg/ffprobe invocation failed or its output was unusable."""


class MissingDependencyError(PixelJudgeError):
    """A required external tool or build feature is absent (e.g. libvmaf)."""


class MetricsError(PixelJudgeError):
    """A metrics log could not be parsed, or the numbers are unusable."""


class DatasetError(PixelJudgeError):
    """A subjective dataset is missing, incomplete, or fails its schema check."""
