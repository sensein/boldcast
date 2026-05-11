"""Causal Mamba block for temporal mixing across TRs.

Pre-LN + ``mamba_ssm.Mamba`` + residual. Operates per-token: the same
Mamba module is applied independently over the time axis for each of
the ``P`` cortical patches in the input.

Imports ``mamba_ssm`` at module load — CUDA-only. Login-node uv gates
should not import this module.
"""

from __future__ import annotations

import torch
from mamba_ssm import Mamba  # type: ignore[import-not-found,import-untyped,unused-ignore]
from torch import nn

__all__ = ["MambaBlock"]


class MambaBlock(nn.Module):
    """LayerNorm → mamba_ssm.Mamba → residual.

    Parameters
    ----------
    d_model : int
    d_state : int, default 16
    d_conv : int, default 4
    expand : int, default 2

    Notes
    -----
    Input ``x`` has shape ``(B, T, P, d_model)``. Mamba operates on
    a 3-D ``(B', L, D)`` input where ``B' = B * P`` (each patch gets
    its own independent temporal sequence) and ``L = T``. We reshape
    in and out — Mamba's internal selective scan handles the temporal
    causality.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, p, d = x.shape
        h = self.norm(x)
        # Move P into the batch axis so each patch gets its own temporal scan.
        h = h.permute(0, 2, 1, 3).reshape(b * p, t, d)
        h = self.mamba(h)
        h = h.reshape(b, p, t, d).permute(0, 2, 1, 3).contiguous()
        out: torch.Tensor = x + h
        return out
