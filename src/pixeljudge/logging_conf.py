"""Logging setup.

Every module asks for its logger with ``get_logger(__name__)`` and never calls
``print``. The reason is practical rather than stylistic: encodes and
measurements are long-running, so we want timestamps, levels and the ability to
turn the noise up or down from one place. ``rich`` gives us readable console
output; the underlying records are still plain ``logging`` records, so a file
handler or a CI-friendly plain formatter can be bolted on later.
"""

from __future__ import annotations

import logging
import os

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False

# One shared console keeps rich's progress bars and log lines from fighting over
# the same terminal.
console = Console(stderr=True)


def setup_logging(level: int | str | None = None) -> None:
    """Install the rich handler on the root logger. Safe to call repeatedly.

    Level resolution order: explicit argument, then ``PIXELJUDGE_LOG_LEVEL``,
    then INFO.
    """
    global _CONFIGURED
    if _CONFIGURED:
        if level is not None:
            logging.getLogger().setLevel(_coerce_level(level))
        return

    resolved = _coerce_level(
        level if level is not None else os.getenv("PIXELJUDGE_LOG_LEVEL", "INFO")
    )
    handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        markup=False,
        show_path=False,
        omit_repeated_times=False,
    )
    logging.basicConfig(
        level=resolved,
        format="%(message)s",
        datefmt="%H:%M:%S",
        handlers=[handler],
        force=True,
    )
    # matplotlib's font manager is chatty at DEBUG and drowns out our own logs.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True


def _coerce_level(level: int | str) -> int:
    if isinstance(level, int):
        return level
    named = logging.getLevelName(level.upper())
    return named if isinstance(named, int) else logging.INFO


def get_logger(name: str) -> logging.Logger:
    """Return a logger, configuring the root handler on first use."""
    setup_logging()
    return logging.getLogger(name)
