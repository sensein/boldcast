"""Cortical-patch mean pooling (vertex BOLD → patch BOLD)."""

from __future__ import annotations

import torch

__all__ = ["Patcher"]


class Patcher(torch.nn.Module):
    """Mean-pool per-vertex BOLD into per-patch tokens.

    Parameters
    ----------
    patch_assignment : torch.Tensor of shape ``(V_cortex,)``
        Patch index for each cortex grayordinate, in ``[0, n_patches)``.
        Coerced to ``torch.long`` if not already.
    n_patches : int
        Total patch count. Every patch must contain at least one vertex,
        and every value in ``patch_assignment`` must be in
        ``[0, n_patches)``.

    Notes
    -----
    Forward call: ``forward(x: (T, V_cortex)) -> (T, n_patches)``.
    Implemented via ``torch.Tensor.index_add_`` plus per-patch counts.
    """

    def __init__(self, patch_assignment: torch.Tensor, n_patches: int) -> None:
        super().__init__()
        if patch_assignment.dtype != torch.long:
            patch_assignment = patch_assignment.long()
        if (
            patch_assignment.min().item() < 0
            or patch_assignment.max().item() >= n_patches
        ):
            raise ValueError(
                f"patch_assignment values out of range [0, {n_patches}): "
                f"min={patch_assignment.min().item()}, "
                f"max={patch_assignment.max().item()}"
            )
        counts = torch.bincount(patch_assignment, minlength=n_patches)
        if (counts == 0).any():
            empty = torch.nonzero(counts == 0).flatten().tolist()
            raise ValueError(f"empty patch(es) in assignment: {empty[:5]}...")

        self.register_buffer("patch_assignment", patch_assignment)
        self.register_buffer("counts", counts.to(torch.float32))
        self.n_patches = n_patches

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool ``x`` of shape ``(T, V_cortex)`` to ``(T, n_patches)``."""
        if x.ndim != 2 or x.shape[1] != self.patch_assignment.shape[0]:
            raise ValueError(
                f"expected x of shape (T, {self.patch_assignment.shape[0]}); "
                f"got {tuple(x.shape)}"
            )
        sums = torch.zeros(
            x.shape[0], self.n_patches, dtype=x.dtype, device=x.device
        )
        sums.index_add_(1, self.patch_assignment, x)
        return sums / self.counts.to(x.dtype).clamp_min(1.0)
