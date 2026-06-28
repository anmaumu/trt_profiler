"""YAML configuration loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Parameters
    ----------
    path
        YAML file path.

    Returns
    -------
    dict[str, Any]
        Loaded configuration mapping.

    Raises
    ------
    RuntimeError
        If PyYAML is not installed.
    ValueError
        If the loaded YAML root is not a mapping.
    """

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("pyyaml is required to load config files.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded
