"""BOLDcast ↔ BRAINMARKS adapter — pure core.

Importable without ``brainmarks`` or ``mamba-ssm``: contains only the input
transform and embedding-pooling logic, so it is unit-testable in the dev venv.
The brainmarks/model-coupled registration lives in ``brainmarks_plugin/``.

Design rationale: ``docs/superpowers/specs/2026-05-30-brainmarks-adapter-design.md``
(internal design note, not published).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from boldcast.tokenize.patcher import Patcher

__all__ = ["cortex_slice", "patch_tokens", "pool_embeddings", "BOLDcastTransform"]


def cortex_slice(
    arr: NDArray[np.floating[Any]], cortex_index: NDArray[np.integer[Any]]
) -> NDArray[np.float32]:
    """Select cortical grayordinate columns from a ``(T, G)`` BOLD array.

    Parameters
    ----------
    arr : ndarray of shape ``(T, G)``
        Full grayordinate series (e.g. BRAINMARKS ``fslr91k``: ``G = 91282``).
    cortex_index : ndarray of int
        Column indices of the cortical grayordinates, in BOLDcast's
        ``patch_assignment`` order (LH-then-RH). From
        ``scripts/verify_brainmarks_grayordinate_order.py``.

    Returns
    -------
    ndarray of shape ``(T, len(cortex_index))`` float32
    """
    arr = np.asarray(arr)
    cortex_index = np.asarray(cortex_index)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D (T, G) array; got shape {arr.shape}")
    if cortex_index.ndim != 1:
        raise ValueError(f"cortex_index must be 1D; got shape {cortex_index.shape}")
    if cortex_index.size and (
        int(cortex_index.min()) < 0 or int(cortex_index.max()) >= arr.shape[1]
    ):
        raise ValueError(
            f"cortex_index out of range [0, {arr.shape[1]}): "
            f"min={int(cortex_index.min())}, max={int(cortex_index.max())}"
        )
    return arr[:, cortex_index].astype(np.float32, copy=False)


def patch_tokens(
    cortex: NDArray[np.floating[Any]],
    patch_assignment: NDArray[np.integer[Any]],
    n_patches: int,
) -> torch.Tensor:
    """Mean-pool ``(T, V_cortex)`` cortex BOLD into ``(T, n_patches)`` tokens.

    Thin wrapper over :class:`boldcast.tokenize.patcher.Patcher`. ``cortex`` must
    have ``V_cortex == len(patch_assignment)`` columns (the cortex slice from
    :func:`cortex_slice`).

    Returns
    -------
    torch.Tensor of shape ``(T, n_patches)``, dtype ``torch.float32``
    """
    patcher = Patcher(torch.as_tensor(np.asarray(patch_assignment)), n_patches)
    x = torch.as_tensor(np.asarray(cortex), dtype=torch.float32)
    return patcher.forward(x)


def pool_embeddings(h: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool BOLDcast embeddings ``h: (B, T, P, d)`` for a probe.

    ``mode="trait"`` -> ``(B, 1, d)`` (mean over time and patches; the
    K99 ``mean_tp`` vector, fed to BRAINMARKS as ``cls_embeds``).
    ``mode="state"`` -> ``(B, T, d)`` (mean over patches, time preserved;
    fed as ``patch_embeds`` for per-TR state decoding).
    """
    if h.ndim != 4:
        raise ValueError(f"expected h of shape (B, T, P, d); got {tuple(h.shape)}")
    if mode == "trait":
        return h.mean(dim=(1, 2)).unsqueeze(1)
    if mode == "state":
        return h.mean(dim=2)
    raise ValueError(f"unknown mode {mode!r}; expected 'trait' or 'state'")


class BOLDcastTransform:
    """BRAINMARKS ``ModelTransform``: grayordinate series -> BOLDcast tokens.

    Slices the cortical grayordinates and mean-pools them into patch tokens
    using BOLDcast's precomputed ``patch_assignment``. Does **not** re-normalize:
    BRAINMARKS stores BOLD per-dimension z-scored over time, which is what
    BOLDcast's ``standardize_run`` produces.

    Parameters
    ----------
    cortex_index, patch_assignment : ndarray
        Loaded from the verify-script and patch caches (subject-invariant).
    n_patches : int
        Patch count (e.g. 1024).
    bold_key : str
        Sample key holding the ``(T, G)`` series. Default ``"bold"``.
    window : int or None
        Reserved; windowing is deferred. Any non-None value raises
        ``NotImplementedError`` (see spec — defaults OFF for M2).
    """

    def __init__(
        self,
        cortex_index: NDArray[np.integer[Any]],
        patch_assignment: NDArray[np.integer[Any]],
        n_patches: int,
        bold_key: str = "bold",
        window: int | None = None,
    ) -> None:
        if window is not None:
            raise NotImplementedError(
                "windowing is deferred post-M2; pass window=None (whole clip)"
            )
        self.cortex_index = np.asarray(cortex_index)
        self.patch_assignment = np.asarray(patch_assignment)
        self.n_patches = int(n_patches)
        self.bold_key = bold_key
        self._patcher = Patcher(torch.as_tensor(self.patch_assignment), self.n_patches)

    def __call__(self, sample: dict[str, Any]) -> dict[str, Any]:
        bold = np.asarray(sample[self.bold_key])
        cortex = cortex_slice(bold, self.cortex_index)
        tokens = self._patcher.forward(torch.as_tensor(cortex, dtype=torch.float32))
        return {**sample, "tokens": tokens}
