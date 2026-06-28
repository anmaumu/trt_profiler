"""Path helper functions."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if it does not exist.

    Parameters
    ----------
    path
        Directory path.

    Returns
    -------
    Path
        Created or existing directory path.
    """

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved
