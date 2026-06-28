from __future__ import annotations

from typing import Any


def require_keys(config: dict[str, Any], keys: list[str], context: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing required keys in {context}: {missing}")
