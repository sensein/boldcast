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


@pytest.mark.gpu
def test_mamba_block_forward_shape() -> None:
    """MambaBlock preserves (B, T, P, d_model) shape on a CUDA tensor."""
    from boldcast.models.temporal import MambaBlock

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    d_model = 16
    block = MambaBlock(d_model=d_model).cuda()
    x = torch.randn(2, 5, 8, d_model, device="cuda")
    out = block(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_boldcast_demo_param_count_in_budget() -> None:
    """CPU instantiation: param count must be in [0.5e6, 1.5e6] per ADR 0004."""
    from boldcast.models.boldcast_demo import BOLDcastDemo

    # NOTE: This test needs mamba_ssm to construct the full model. Skip under
    # uv where mamba_ssm is not installed (no GPU on the login node).
    try:
        from boldcast.models.temporal import MambaBlock  # noqa: F401
    except ImportError:
        pytest.skip("mamba_ssm not installed (uv login-node env)")

    adjacency = _identity_adjacency(n_patches=1024, k=8)
    m = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=1024,
        k_neighbors=8,
        adjacency=adjacency,
        horizons=(1, 5),
    )
    n_params = sum(p.numel() for p in m.parameters())
    assert 0.5e6 <= n_params <= 1.5e6, (
        f"param count {n_params/1e6:.3f}M outside Day-3 budget [0.5M, 1.5M]"
    )


def test_boldcast_demo_embed_returns_d_model() -> None:
    """``embed(x)`` returns (B, T, P, d_model); ``forward(x)`` returns (B, T, P, H, d_in).

    CPU-runnable using n_layers=0 (skip Mamba+kNN stack, exercise embed + head
    only) — this path doesn't touch boldcast.models.temporal at all and works
    under uv without mamba_ssm."""
    from boldcast.models.boldcast_demo import BOLDcastDemo

    adjacency = _identity_adjacency(n_patches=8, k=4)
    m = BOLDcastDemo(
        d_in=1,
        d_model=8,
        n_layers=0,
        n_patches=8,
        k_neighbors=4,
        adjacency=adjacency,
        horizons=(1,),
    )
    x = torch.randn(2, 5, 8, 1)
    embed = m.embed(x)
    out = m(x)
    assert embed.shape == (2, 5, 8, 8)
    assert out.shape == (2, 5, 8, 1, 1)
    assert torch.isfinite(out).all()


@pytest.mark.gpu
def test_boldcast_demo_full_forward_on_cuda() -> None:
    """Full 4-layer model forward on (B=2, T=256, P=1024, 1) under CUDA."""
    from boldcast.models.boldcast_demo import BOLDcastDemo

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    adjacency = _identity_adjacency(n_patches=1024, k=8).cuda()
    m = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=1024,
        k_neighbors=8,
        adjacency=adjacency,
        horizons=(1, 5),
    ).cuda()
    x = torch.randn(2, 256, 1024, 1, device="cuda")
    out = m(x)
    assert out.shape == (2, 256, 1024, 2, 1)
    assert torch.isfinite(out).all()


def test_baseline_schaefer_param_count_in_budget() -> None:
    """Same param budget as BOLDcastDemo — only n_patches differs (400 vs 1024)."""
    pytest.importorskip("mamba_ssm")
    from boldcast.models.baseline import BaselineSchaefer400

    adjacency = _identity_adjacency(n_patches=400, k=8)
    m = BaselineSchaefer400(
        d_in=1, d_model=128, n_layers=4, k_neighbors=8, adjacency=adjacency,
        horizons=(1, 5),
    )
    n_params = sum(p.numel() for p in m.parameters())
    assert 0.5e6 <= n_params <= 1.5e6


@pytest.mark.gpu
def test_baseline_schaefer_forward_on_cuda() -> None:
    from boldcast.models.baseline import BaselineSchaefer400

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    adjacency = _identity_adjacency(n_patches=400, k=8).cuda()
    m = BaselineSchaefer400(
        d_in=1, d_model=128, n_layers=4, k_neighbors=8, adjacency=adjacency,
        horizons=(1, 5),
    ).cuda()
    x = torch.randn(2, 256, 400, 1, device="cuda")
    out = m(x)
    assert out.shape == (2, 256, 400, 2, 1)


def test_boldcast_demo_forward_multi_horizon_shape() -> None:
    """ADR 0005 D2: forward returns (B, T, P, H, d_in); H axis always present.

    CPU-runnable with n_layers=0 (skips Mamba)."""
    from boldcast.models.boldcast_demo import BOLDcastDemo

    adjacency = _identity_adjacency(n_patches=8, k=4)
    m = BOLDcastDemo(
        d_in=1,
        d_model=8,
        n_layers=0,
        n_patches=8,
        k_neighbors=4,
        adjacency=adjacency,
        horizons=(1, 5),
    )
    x = torch.randn(2, 7, 8, 1)
    out = m(x)
    assert out.shape == (2, 7, 8, 2, 1)
    assert torch.isfinite(out).all()


def test_boldcast_demo_forward_single_horizon_preserves_h_axis() -> None:
    """H axis present even at H=1 (ADR 0005 D2)."""
    from boldcast.models.boldcast_demo import BOLDcastDemo

    adjacency = _identity_adjacency(n_patches=8, k=4)
    m = BOLDcastDemo(
        d_in=1,
        d_model=8,
        n_layers=0,
        n_patches=8,
        k_neighbors=4,
        adjacency=adjacency,
        horizons=(1,),
    )
    x = torch.randn(2, 7, 8, 1)
    out = m(x)
    assert out.shape == (2, 7, 8, 1, 1)


def test_boldcast_demo_rejects_empty_horizons() -> None:
    """horizons=() must raise ValueError at construction (ADR 0005 D2)."""
    from boldcast.models.boldcast_demo import BOLDcastDemo

    adjacency = _identity_adjacency(n_patches=8, k=4)
    with pytest.raises(ValueError, match=r"horizons must be non-empty"):
        BOLDcastDemo(
            d_in=1,
            d_model=8,
            n_layers=0,
            n_patches=8,
            k_neighbors=4,
            adjacency=adjacency,
            horizons=(),
        )


def test_boldcast_demo_rejects_nonpositive_horizons() -> None:
    """A zero or negative horizon must raise ValueError at construction."""
    from boldcast.models.boldcast_demo import BOLDcastDemo

    adjacency = _identity_adjacency(n_patches=8, k=4)
    with pytest.raises(ValueError, match=r"horizons must be positive"):
        BOLDcastDemo(
            d_in=1,
            d_model=8,
            n_layers=0,
            n_patches=8,
            k_neighbors=4,
            adjacency=adjacency,
            horizons=(1, 0),
        )
