from __future__ import annotations

import importlib
from typing import Any


def load_class(class_path: str) -> type[Any]:
    """Load a class from a dotted import path."""
    if "." not in class_path:
        raise ValueError(f"Class path must include a module path: {class_path!r}")

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    loaded = getattr(module, class_name)
    if not isinstance(loaded, type):
        raise TypeError(f"Loaded object is not a class: {class_path}")
    return loaded
