"""Logging helpers.

The project uses the standard :mod:`logging` module everywhere. Call
:func:`configure_logging` once at process start (the UI, API and scripts do
this); library modules simply call :func:`get_logger`.
"""

from __future__ import annotations

import logging
import sys

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger once with a consistent format."""
    global _CONFIGURED
    if _CONFIGURED:
        logging.getLogger().setLevel(level.upper())
        return
    logging.basicConfig(level=level.upper(), format=_FORMAT, stream=sys.stderr)
    # Third-party libraries are noisy at INFO.
    for noisy in ("httpx", "httpcore", "urllib3", "transformers", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module logger (``logging.getLogger`` with a project convention)."""
    return logging.getLogger(name)
