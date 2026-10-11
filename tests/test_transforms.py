"""Tests for ``boldcast/data/transforms.py``."""

from __future__ import annotations

import numpy as np
import pytest
from boldcast.data.transforms import standardize_run


def test_standardize_run_zero_mean_unit_std_per_grayordinate() -> None:
    """Each column should be exactly zero-mean and approximately unit-std."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((100, 50)).astype(np.float32) * 5.0 + 3.0
    out = standardize_run(x)
    assert out.shape == x.shape
    assert out.dtype == np.float32
    np.testing.assert_allclose(out.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(out.std(axis=0), 1.0, atol=1e-5)


def test_standardize_run_handles_zero_variance_column() -> None:
    """Constant columns should not produce NaN or inf."""
    x = np.zeros((10, 3), dtype=np.float32)
    x[:, 1] = 7.0  # constant column
    out = standardize_run(x)
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(out[:, 1], np.zeros(10, dtype=np.float32))


def test_standardize_run_rejects_non_2d() -> None:
    with pytest.raises(ValueError, match="2-D"):
        standardize_run(np.zeros((10,), dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        standardize_run(np.zeros((2, 3, 4), dtype=np.float32))
