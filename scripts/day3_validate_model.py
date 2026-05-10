"""Day-3 BOLDcastDemo validation on a CUDA device.

NOTE: this script depends on mamba-ssm and a CUDA GPU. Run on an ORCD
GPU compute node under the micromamba env. Claude does not execute it
(no GPU at the login node; uv env lacks mamba-ssm).

Loads the demo config, builds patch + kNN caches (or uses existing),
instantiates BOLDcastDemo on cuda, runs one forward pass on
(B=2, T=256, P=1024, 1), and prints:
  * Parameter count
  * Output shape
  * Wall-time of the forward pass
  * Peak CUDA memory during forward (training-mode)

Acceptance (per docs/10_day_plan.md Day 3):
  * <1 second wall-time on H200
  * <8 GB peak memory for forward+backward
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from boldcast._upstream.cifti_io import (
    cortex_grayordinate_indices,
    load_dtseries,
)
from boldcast.models.boldcast_demo import BOLDcastDemo
from boldcast.tokenize.geodesic import build_or_load_patches
from boldcast.tokenize.knn import build_or_load_knn
from omegaconf import OmegaConf


def _read_subject_list(path: str) -> list[str]:
    out: list[str] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available():
        raise SystemExit("[day3] CUDA not available — run on a GPU node.")

    train_subjects = _read_subject_list(str(cfg.data.subjects_train_file))
    ref_subject = train_subjects[0]
    ref_run = cfg.data.runs[0]
    ref_path = str(cfg.data.dtseries_pattern).format(
        subject=ref_subject, run=ref_run
    )
    print(f"[day3] reference subject={ref_subject}, run={ref_run}")
    _, header = load_dtseries(ref_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)

    surface_dir = str(cfg.data.surface_dir_template).format(subject=ref_subject)
    lh_mesh = f"{surface_dir}/{ref_subject}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    rh_mesh = f"{surface_dir}/{ref_subject}.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

    print(f"[day3] loading patch assignment from {cfg.tokenize.patch_cache}")
    assignment = build_or_load_patches(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(cfg.tokenize.patch_cache),
        n_patches=int(cfg.tokenize.n_patches_cortex),
    )

    print(f"[day3] building/loading kNN from {cfg.tokenize.knn_cache}")
    adjacency = build_or_load_knn(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        patch_assignment=assignment,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k=int(cfg.tokenize.knn_k),
        cache_path=str(cfg.tokenize.knn_cache),
    )

    print("[day3] building BOLDcastDemo on cuda ...")
    model = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=torch.from_numpy(np.asarray(adjacency)).long().cuda(),
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[day3]   params: {n_params/1e6:.3f} M")

    x = torch.randn(
        2, 256, int(cfg.tokenize.n_patches_cortex), 1, device="cuda"
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(x)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[day3] forward {tuple(x.shape)} -> {tuple(out.shape)} "
        f"in {dt*1000:.1f} ms, peak forward memory {peak_gb:.2f} GB"
    )

    # Forward+backward memory probe (training-mode acceptance criterion).
    x_b = torch.randn(
        2, 256, int(cfg.tokenize.n_patches_cortex), 1,
        device="cuda", requires_grad=True,
    )
    torch.cuda.reset_peak_memory_stats()
    out_b = model(x_b)
    loss = out_b.pow(2).mean()
    loss.backward()
    fwd_bwd_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(
        f"[day3] forward+backward peak memory {fwd_bwd_gb:.2f} GB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
