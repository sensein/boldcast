"""Forward-pass tests for Day-3 model components.

CPU-runnable tests (KNNAttention, embed shapes) run under uv. Tests that
import mamba_ssm carry @pytest.mark.gpu and run only on a GPU node under
the micromamba env.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from boldcast.models.spatial import KNNAttention


def _identity_adjacency(n_patches: int, k: int) -> torch.Tensor:
    """Each token attends to the first k patches (including self). Trivial
    structure suitable for shape/dtype/NaN tests."""
    adj = np.zeros((n_patches, k), dtype=np.int64)
    for i in range(n_patches):
        adj[i, 0] = i  # self first
        # Fill remaining slots with neighbours rolled across the patch ring
        for j in range(1, k):
            adj[i, j] = (i + j) % n_patches
    return torch.from_numpy(adj)


def test_knn_attention_forward_shape() -> None:
    d_model, n_patches, k = 16, 8, 4
    attn = KNNAttention(d_model=d_model, k=k, adjacency=_identity_adjacency(n_patches, k))
    x = torch.randn(2, 5, n_patches, d_model)
    out = attn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_knn_attention_rejects_wrong_n_patches() -> None:
    d_model, n_patches, k = 16, 8, 4
    attn = KNNAttention(d_model=d_model, k=k, adjacency=_identity_adjacency(n_patches, k))
    bad = torch.randn(2, 5, n_patches + 1, d_model)
    with pytest.raises((ValueError, RuntimeError, IndexError)):
        attn(bad)


def test_knn_attention_param_count_quadratic_in_d_model() -> None:
    """Q/K/V/O projections: 4 × (d_model² + d_model). LayerNorm adds 2·d_model."""
    d_model, n_patches, k = 32, 16, 4
    attn = KNNAttention(d_model=d_model, k=k, adjacency=_identity_adjacency(n_patches, k))
    expected = 4 * (d_model * d_model + d_model) + 2 * d_model
    got = sum(p.numel() for p in attn.parameters())
    assert got == expected, f"expected {expected} params, got {got}"
