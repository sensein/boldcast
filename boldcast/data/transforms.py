"""Run-level numerical transforms applied before tokenization."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["standardize_run"]


def standardize_run(
    x: NDArray[np.floating[Any]], eps: float = 1e-8
) -> NDArray[np.float32]:
    """Per-column (per-grayordinate) Z-score standardization over time.

    Parameters
    ----------
    x : ndarray of shape ``(T, V)``
        BOLD timeseries for one run.
    eps : float
        Numerical floor added to each column's std before division. Columns
        whose std is exactly zero (constant timeseries) are returned as zeros.

    Returns
    -------
    standardized : ndarray of shape ``(T, V)`` float32
        Each column has approximately zero mean and unit variance over time;
        a column that was constant in input is exactly zero in output.
    """
    if x.ndim != 2:
        raise ValueError(f"standardize_run expects 2-D input (T, V), got shape {x.shape}")
    arr = np.asarray(x, dtype=np.float64)  # accumulate in f64, return f32
    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    constant = std < eps
    safe_std = np.where(constant, 1.0, std)
    out = (arr - mean) / safe_std
    out = np.where(constant, 0.0, out)
    return out.astype(np.float32)
