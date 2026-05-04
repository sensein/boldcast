"""Tests for ``boldcast/tokenize/patcher.py``."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from boldcast.tokenize.patcher import Patcher


def _assignment_with_full_coverage(n_v: int, n_p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    assignment = rng.integers(low=0, high=n_p, size=n_v)
    # Guarantee no empty patch by forcing patch p onto vertex p when missing.
    for p in range(n_p):
        if (assignment == p).sum() == 0:
            assignment[p] = p
    return assignment.astype(np.int64)


def test_forward_shape_and_dtype() -> None:
    n_v, n_p, n_t = 100, 8, 5
    assignment = _assignment_with_full_coverage(n_v, n_p)
    patcher = Patcher(torch.from_numpy(assignment), n_patches=n_p)

    x = torch.randn(n_t, n_v, dtype=torch.float32)
    out = patcher.forward(x)
    assert out.shape == (n_t, n_p)
    assert out.dtype == torch.float32


def test_scatter_mean_correctness() -> None:
    n_v, n_p, n_t = 60, 4, 3
    assignment = _assignment_with_full_coverage(n_v, n_p)
    patcher = Patcher(torch.from_numpy(assignment), n_patches=n_p)

    x = torch.randn(n_t, n_v, dtype=torch.float32)
    out = patcher.forward(x).numpy()

    expected = np.stack(
        [x.numpy()[:, assignment == p].mean(axis=1) for p in range(n_p)],
        axis=1,
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_empty_patch_raises_at_init() -> None:
    n_v, n_p = 20, 4
    assignment = np.zeros(n_v, dtype=np.int64)  # everything in patch 0
    with pytest.raises(ValueError, match="empty patch"):
        Patcher(torch.from_numpy(assignment), n_patches=n_p)


def test_assignment_out_of_range_raises_at_init() -> None:
    assignment = np.array([0, 1, 5], dtype=np.int64)  # 5 >= n_patches=4
    with pytest.raises(ValueError, match="out of range"):
        Patcher(torch.from_numpy(assignment), n_patches=4)


def test_wrong_input_shape_raises() -> None:
    n_v, n_p = 20, 4
    assignment = _assignment_with_full_coverage(n_v, n_p)
    patcher = Patcher(torch.from_numpy(assignment), n_patches=n_p)
    with pytest.raises(ValueError, match="expected x of shape"):
        patcher.forward(torch.randn(5, 19))  # wrong V


def test_int32_assignment_accepted() -> None:
    """Coming from precompute_patches, the array is int32; Patcher must coerce."""
    n_v, n_p = 60, 4
    assignment = _assignment_with_full_coverage(n_v, n_p).astype(np.int32)
    patcher = Patcher(torch.from_numpy(assignment), n_patches=n_p)
    out = patcher.forward(torch.randn(2, n_v))
    assert out.shape == (2, n_p)
