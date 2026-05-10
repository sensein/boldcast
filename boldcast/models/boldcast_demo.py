"""BOLDcast-Demo model: 4-layer interleaved Mamba + kNN spatial attention.

Patch-shared embed and head; param count asserted in [0.5e6, 1.5e6] at
construction (ADR 0004). The ``Mamba``-importing ``MambaBlock`` is
imported lazily inside the constructor so this module can be loaded
under uv (CPU-only) for non-Mamba code paths (embed/head shape tests,
param-count audit at ``n_layers=0``).
"""

from __future__ import annotations

# nn.Module.__call__ returns Any; cast() keeps mypy strict happy without
# runtime overhead.
from typing import cast as _cast

import torch
from torch import nn

from boldcast.models.spatial import KNNAttention

__all__ = ["BOLDcastDemo"]


class BOLDcastDemo(nn.Module):
    """Demo BOLDcast backbone.

    Parameters
    ----------
    d_in : int
        Per-token input channel count (1 for raw patch-mean BOLD).
    d_model : int
    n_layers : int
        Number of (MambaBlock + KNNAttention) pairs. Setting ``n_layers=0``
        is supported for shape-only / param-budget tests that don't need
        the Mamba CUDA path.
    n_patches : int
    k_neighbors : int
    adjacency : torch.Tensor of shape ``(n_patches, k_neighbors)`` long
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        n_patches: int,
        k_neighbors: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_patches = n_patches
        self.embed_proj = nn.Linear(d_in, d_model, bias=True)
        self.layers = nn.ModuleList()
        if n_layers > 0:
            from boldcast.models.temporal import MambaBlock

            for _ in range(n_layers):
                self.layers.append(MambaBlock(d_model=d_model))
                self.layers.append(KNNAttention(
                    d_model=d_model, k=k_neighbors, adjacency=adjacency
                ))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, d_in, bias=True)

        # Param budget audit. ADR 0004 D1/D2/D4/D5: ~0.7M target for the
        # default 4-layer / d_model=128 / P=1024 / k=8 config; allow
        # 0.5M-1.5M to accommodate Mamba's per-block param variance.
        if n_layers > 0:
            n_params = sum(p.numel() for p in self.parameters())
            if not (0.5e6 <= n_params <= 1.5e6):
                raise AssertionError(
                    f"BOLDcastDemo param count {n_params/1e6:.3f}M is "
                    "outside the Day-3 budget [0.5M, 1.5M]. Spec drift?"
                )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Run everything except the final head. Returns
        ``(B, T, P, d_model)`` — used by Day-7 fingerprint eval."""
        if x.shape[-1] != self.d_in:
            raise ValueError(
                f"input last-dim {x.shape[-1]} != d_in={self.d_in}"
            )
        h = self.embed_proj(x)  # (B, T, P, d_model)
        for layer in self.layers:
            # MambaBlock already adds the residual; KNNAttention does not.
            if isinstance(layer, KNNAttention):
                h = h + layer(h)
            else:
                h = layer(h)
        h = _cast(torch.Tensor, self.final_norm(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        out: torch.Tensor = self.head(h)
        return out
