"""BRAINMARKS registration shim for BOLDcast (executed on a GPU node only).

CONFIRM before live run:
  - brainmarks import paths + discovery mechanism (namespace pkg vs entry point)
  - the checkpoint path / cwd resolution under the BRAINMARKS runner
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from boldcast.eval.brainmarks_adapter import BOLDcastTransform, pool_embeddings
from boldcast.models.boldcast_demo import BOLDcastDemo
from brainmarks.models.base import Embeddings, ModelWrapper  # CONFIRM path
from brainmarks.models.registry import register_model  # CONFIRM path
from numpy.typing import NDArray


class BOLDcastBrainmarks(ModelWrapper):  # type: ignore[misc]
    __space__ = "fslr91k"

    def __init__(
        self,
        ckpt_path: str,
        adjacency: NDArray[np.integer[Any]] | torch.Tensor,
        n_patches: int = 1024,
        d_model: int = 128,
        n_layers: int = 4,
        k_neighbors: int = 8,
    ) -> None:
        super().__init__()
        self.backbone = BOLDcastDemo(
            d_in=1,
            d_model=d_model,
            n_layers=n_layers,
            n_patches=n_patches,
            k_neighbors=k_neighbors,
            adjacency=torch.as_tensor(np.asarray(adjacency)),
            horizons=(1, 5),
        )
        state = torch.load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(state.get("model", state))
        self.backbone.eval()

    @torch.no_grad()
    def forward(self, batch: dict[str, torch.Tensor]) -> Any:  # noqa: ANN401
        # Populate BOTH cls (trait / mean_tp) and patch (state, per-TR) so a single
        # registered model serves both BRAINMARKS probe families; the probe selects
        # the field it needs. Both pools are cheap means over the same tensor.
        h = self.backbone.embed(batch["tokens"].unsqueeze(-1))  # (B,T,P,d_model)
        return Embeddings(
            cls_embeds=pool_embeddings(h, "trait"),
            reg_embeds=None,
            patch_embeds=pool_embeddings(h, "state"),
        )


@register_model("boldcast_demo")  # type: ignore[misc]
def build_boldcast() -> tuple[BOLDcastTransform, BOLDcastBrainmarks]:
    # CONFIRM: these cache/ckpt paths are relative to the process cwd. Under the
    # BRAINMARKS runner the cwd may not be the repo root — resolve relative to the
    # repo root or accept an env override before the live run, else loads fail.
    pa = np.load("cache/patches_fsLR_32k_n1024_seed0_geo.npz")["assignment"]
    ci = np.load("cache/brainmarks_cortex_index.npz")["cortex_index"]
    adj = np.load("cache/knn_k8_n1024.npz")["adjacency"]  # key per boldcast/tokenize/knn.py
    transform = BOLDcastTransform(cortex_index=ci, patch_assignment=pa, n_patches=1024)
    model = BOLDcastBrainmarks(
        ckpt_path="results/day5_train/ckpt_final.pt", adjacency=adj
    )
    return transform, model
