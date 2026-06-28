from __future__ import annotations

import numpy as np


def as_float64(array: np.ndarray) -> np.ndarray:
    return np.asarray(array).astype(np.float64, copy=False)
