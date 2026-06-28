"""Small validation helpers for configuration dictionaries."""

from __future__ import annotations

from typing import Any


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    """Require keys in a configuration mapping.

    Parameters
    ----------
    config
        Configuration mapping to validate.
    keys
        Required key names.
    context
        Human-readable context used in error messages.

    Raises
    ------
    ValueError
        If one or more keys are missing.
    """

    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {missing}")
