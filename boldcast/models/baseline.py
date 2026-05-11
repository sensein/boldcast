"""Schaefer-400 ROI baseline: BOLDcast backbone applied to 400 ROI tokens.

Same MLP-embed → 4 × (Mamba + kNN) → head architecture as BOLDcastDemo, just
with ``n_patches = 400`` instead of 1024. Distinct class so configs and
training scripts disambiguate without a string lookup on n_patches.
"""

from __future__ import annotations

from typing import cast as _cast

import torch
from torch import nn

from boldcast.models.boldcast_demo import BOLDcastDemo

__all__ = ["BaselineSchaefer400"]


class BaselineSchaefer400(nn.Module):
    """Schaefer-400 baseline. Backbone identical to ``BOLDcastDemo``; only
    the spatial token count differs (``P=400``)."""

    def __init__(
        self,
        d_in: int,
        d_model: int,
        n_layers: int,
        k_neighbors: int,
        adjacency: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self._inner = BOLDcastDemo(
            d_in=d_in,
            d_model=d_model,
            n_layers=n_layers,
            n_patches=400,
            k_neighbors=k_neighbors,
            adjacency=adjacency,
            use_checkpoint=use_checkpoint,
        )

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self._inner.embed(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _cast(torch.Tensor, self._inner(x))
