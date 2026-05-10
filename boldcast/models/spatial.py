"""Local kNN spatial attention for cortical-patch tokens.

Single-head scaled dot-product attention restricted to the ``k`` precomputed
neighbors per token (including self). Pre-LayerNorm; residual is applied by
the caller (the model block) — this module returns the attention output
only, not ``x + attn(x)``.

Pure torch; runs on CPU and GPU. No mamba-ssm dependency.
"""

from __future__ import annotations

import math

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
            raise ValueError(
                f"adjacency shape {tuple(adjacency.shape)} does not match "
                f"(P, k={k})"
            )
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
        """
        if x.shape[-2] != self.n_patches:
            raise ValueError(
                f"input has P={x.shape[-2]} tokens but adjacency was built "
                f"for P={self.n_patches}"
            )
        b, t, p, d = x.shape
        h = self.norm(x)  # (B, T, P, d)
        q = self.q(h)  # (B, T, P, d)
        k_all = self.kproj(h)  # (B, T, P, d)
        v_all = self.v(h)  # (B, T, P, d)

        # Gather neighbor keys/values: adjacency is (P, k).
        adj = self.adjacency  # (P, k) long
        k_gather = k_all[:, :, adj, :]  # (B, T, P, k, d)
        v_gather = v_all[:, :, adj, :]  # (B, T, P, k, d)

        # Scaled dot product between q (P) and its k neighbors.
        # q shape (B, T, P, d) → broadcast to (B, T, P, 1, d).
        scores = (q.unsqueeze(-2) * k_gather).sum(dim=-1) / math.sqrt(d)
        # scores: (B, T, P, k)
        weights = torch.softmax(scores, dim=-1)
        # Weighted sum of values: (B, T, P, k) × (B, T, P, k, d) → (B, T, P, d)
        attended = (weights.unsqueeze(-1) * v_gather).sum(dim=-2)
        out: torch.Tensor = self.out(attended)
        return out
