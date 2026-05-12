"""BOLDcast-Demo model: 4-layer interleaved Mamba + kNN spatial attention.

Patch-shared embed and head; param count asserted in [0.5e6, 1.5e6] at
construction (ADR 0004). Head emits all forecasting horizons in parallel
(``Linear(d_model, H * d_in)`` reshaped to ``(B, T, P, H, d_in)``); H axis
is materialized even at H=1 for a shape-stable forward contract
(ADR 0005 D2). The ``Mamba``-importing ``MambaBlock`` is imported lazily
inside the constructor so this module can be loaded under uv (CPU-only)
for non-Mamba code paths (embed/head shape tests, param-count audit at
``n_layers=0``).

Memory: ``use_checkpoint=True`` wraps each (MambaBlock + KNNAttention)
pair in ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)``.
This recomputes the pair's forward during backward instead of saving
activations — essential for Mamba's per-timestep SSM hidden state,
which otherwise dominates training-mode peak memory (~8 GB per block
at the demo shape). Off by default so CPU unit tests don't incur
checkpoint's training-mode requirement; on for the Day-3 validation
script and Day-5 DDP training (matches the canonical training recipe
in docs/methods.md "Long-Context Mamba Backbone").
"""

from __future__ import annotations

# nn.Module.__call__ returns Any; cast() keeps mypy strict happy without
# runtime overhead.
from collections.abc import Sequence
from typing import cast as _cast

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from boldcast.models.spatial import KNNAttention

__all__ = ["BOLDcastDemo"]


class _MambaKnnPair(nn.Module):
    """One layer of the interleaved backbone: MambaBlock + KNNAttention
    with the kNN residual. Packaged as a single Module so activation
    checkpointing wraps the full pair as one recompute unit.

    MambaBlock returns ``x + mamba_residual`` internally; KNNAttention
    returns just the attention output, so this wrapper applies the kNN
    residual externally.
    """

    def __init__(
        self,
        d_model: int,
        k_neighbors: int,
        adjacency: torch.Tensor,
    ) -> None:
        super().__init__()
        # Lazy import: under uv (no mamba_ssm) BOLDcastDemo with n_layers=0
        # does not construct a _MambaKnnPair, so this import never fires.
        from boldcast.models.temporal import MambaBlock

        self.mamba = MambaBlock(d_model=d_model)
        self.knn = KNNAttention(
            d_model=d_model, k=k_neighbors, adjacency=adjacency
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mamba(x)
        out: torch.Tensor = x + self.knn(x)
        return out


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
    horizons : Sequence[int]
        Positive integer forecast offsets emitted by the head, in order.
        Required (no default). The head produces ``len(horizons) * d_in``
        scalars per token; ``forward`` reshapes to ``(B, T, P, H, d_in)``.
    use_checkpoint : bool, default False
        If True, wrap each MambaBlock+KNNAttention pair in
        ``torch.utils.checkpoint.checkpoint`` (non-reentrant). Recomputes
        forward during backward — required to fit training-mode F+B
        memory at ``(2, 256, 1024, 128)`` activation shape on H200.
        Only active during ``self.training=True``.
    """

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        n_patches: int,
        k_neighbors: int,
        adjacency: torch.Tensor,
        horizons: Sequence[int],
        use_checkpoint: bool = False,
    ) -> None:
        # Validate horizons before any nn.Module allocation so that a ValueError
        # on invalid input doesn't leave a half-constructed module on the caller's
        # exception frame (matters under DDP / hyperparameter sweeps).
        horizons_t = tuple(int(h) for h in horizons)
        if len(horizons_t) == 0:
            raise ValueError("horizons must be non-empty")
        if any(h <= 0 for h in horizons_t):
            raise ValueError("horizons must be positive (got at least one h<=0)")

        super().__init__()
        self.d_in = d_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_patches = n_patches
        self.use_checkpoint = use_checkpoint
        self.horizons = horizons_t
        self.embed_proj = nn.Linear(d_in, d_model, bias=True)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(_MambaKnnPair(
                d_model=d_model, k_neighbors=k_neighbors, adjacency=adjacency
            ))
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, len(self.horizons) * d_in, bias=True)

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
        use_cp = self.use_checkpoint and self.training
        for layer in self.layers:
            if use_cp:
                h = _cast(
                    torch.Tensor,
                    checkpoint(layer, h, use_reentrant=False),
                )
            else:
                h = layer(h)
        h = _cast(torch.Tensor, self.final_norm(h))
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        out_flat: torch.Tensor = self.head(h)
        b, t, p, _ = out_flat.shape
        return out_flat.view(b, t, p, len(self.horizons), self.d_in)
