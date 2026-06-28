"""NumPy helper functions."""

from __future__ import annotations

import numpy as np


def as_float64(array: np.ndarray) -> np.ndarray:
    """View or convert an array as float64.

    Parameters
    ----------
    array
        Input array.

    Returns
    -------
    np.ndarray
        Float64 array. A copy is avoided when possible.
    """

    return np.asarray(array).astype(np.float64, copy=False)
