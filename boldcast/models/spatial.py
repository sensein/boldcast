"""Local kNN spatial attention for cortical-patch tokens.

Single-head scaled dot-product attention restricted to the ``k`` precomputed
neighbors per token (including self). Pre-LayerNorm; residual is applied by
the caller (the model block) — this module returns the attention output
only, not ``x + attn(x)``.

Pure torch; runs on CPU and GPU. No mamba-ssm dependency.
"""

from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

__all__ = ["KNNAttention"]


class KNNAttention(nn.Module):
    """Per-token attention over a fixed k-neighbor set.

    Parameters
    ----------
    d_model : int
        Token feature dimension.
    k : int
        Number of neighbors per token (including self).
    adjacency : torch.Tensor of shape ``(P, k)`` long
        For each of ``P`` patches, the patch indices of its k neighbors.
        ``adjacency[i, 0]`` should equal ``i`` per ADR 0004 D3 (self-link).
    """

    def __init__(self, d_model: int, k: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        if adjacency.dtype != torch.long:
            adjacency = adjacency.long()
        if adjacency.ndim != 2 or adjacency.shape[1] != k:
            raise ValueError(f"adjacency shape {tuple(adjacency.shape)} does not match (P, k={k})")
        self.d_model = d_model
        self.k = k
        self.n_patches = adjacency.shape[0]
        self.norm = nn.LayerNorm(d_model)
        self.q = nn.Linear(d_model, d_model, bias=True)
        self.kproj = nn.Linear(d_model, d_model, bias=True)
        self.v = nn.Linear(d_model, d_model, bias=True)
        self.out = nn.Linear(d_model, d_model, bias=True)
        self.register_buffer("adjacency", adjacency, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` of shape ``(B, T, P, d_model)`` → same shape.

        For each token ``(b, t, p)``, attend to ``adjacency[p]`` (k entries)
        and return a weighted sum of those neighbors' value projections.

        Implementation note (memory): the obvious form
        ``k_gather = k_all[:, :, adj, :]`` materializes a
        ``(B, T, P, k, d_model)`` tensor — ``k=8 ×`` the size of the input
        activation. At Day-3 demo shapes ``(2, 256, 1024, 8, 128) fp32``
        that's ~2.15 GB per gather, 4.3 GB per kNN block in fp32, and 17 GB
        across the four kNN layers — enough to push forward+backward peak
        memory past 60 GB. Instead we loop over the ``k`` neighbor slots
        and accumulate ``(B, T, P, d)`` intermediates, reducing peak per-
        layer gather memory by ~k×. Kernel-launch overhead (``k`` advanced-
        index gathers + ``k`` matmuls per block) is negligible vs. the
        memory win.
        """
        if x.shape[-2] != self.n_patches:
            raise ValueError(
                f"input has P={x.shape[-2]} tokens but adjacency was built for P={self.n_patches}"
            )
        b, t, p, d = x.shape
        h = self.norm(x)  # (B, T, P, d)
        q = self.q(h)  # (B, T, P, d)
        k_all = self.kproj(h)  # (B, T, P, d)
        v_all = self.v(h)  # (B, T, P, d)

        # register_buffer typing returns `Tensor | Module`; cast for mypy.
        adj = cast(torch.Tensor, self.adjacency)  # (P, k) long
        scale = 1.0 / math.sqrt(d)

        # Pass 1: compute scores slot-by-slot. Avoids materializing
        # (B, T, P, k, d) at any point; per-iteration peak is (B, T, P, d).
        score_slabs: list[torch.Tensor] = []
        for j in range(self.k):
            neighbors_j = adj[:, j]  # (P,) — index of jth neighbor per patch
            k_j = k_all[:, :, neighbors_j, :]  # (B, T, P, d)
            score_slabs.append((q * k_j).sum(dim=-1) * scale)
        scores = torch.stack(score_slabs, dim=-1)  # (B, T, P, k)
        weights = torch.softmax(scores, dim=-1)

        # Pass 2: weighted sum of values, also slot-by-slot. Use a running
        # sum rather than in-place mutation to keep autograd happy.
        attended = torch.zeros(b, t, p, d, device=x.device, dtype=q.dtype)
        for j in range(self.k):
            neighbors_j = adj[:, j]
            v_j = v_all[:, :, neighbors_j, :]  # (B, T, P, d)
            attended = attended + weights[..., j : j + 1] * v_j

        out: torch.Tensor = self.out(attended)
        return out
