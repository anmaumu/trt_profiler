"""Logging helpers."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a named logger.

    Parameters
    ----------
    name
        Logger name.

    Returns
    -------
    logging.Logger
        Named logger instance.
    """

    return logging.getLogger(name)
