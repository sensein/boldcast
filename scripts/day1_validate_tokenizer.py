"""Day-1 tokenizer validation on real HCP data.

NOTE: This script LOADS HCP data from $HCP_ROOT — it must be run by the DUA
holder (Yibei), never by Claude. See CLAUDE.md "HCP Data Use Agreement".

Reads the demo config, picks the requested subject + run, builds (or loads
from cache) the cortical patch assignment, runs the Patcher on the
dtseries, and writes:

* ``figures/day1_patches.png`` — random-color patch visualization on the
  inflated cortical surface.
* ``results/day1_validate.json`` — patch-size mean/std, tokenizer wall-time,
  round-trip residual.

FPS runs over the chosen surface variant (default midthickness, which sits
between pial and white and is the conventional choice for distance
computations).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from boldcast.io.cifti import (
    cortex_grayordinate_indices,
    extract_cortex_grayordinates,
    load_dtseries,
)
from boldcast.tokenize.geodesic import build_or_load_patches
from boldcast.tokenize.patcher import Patcher
from omegaconf import OmegaConf


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/demo.yaml")
    p.add_argument("--subject", type=str, required=True, help="HCP subject ID, e.g. 100307")
    p.add_argument(
        "--task", type=str, required=True, help="Run name, e.g. rfMRI_REST1_7T_PA"
    )
    p.add_argument("--out-fig", type=str, default="figures/day1_patches.png")
    p.add_argument("--out-json", type=str, default="results/day1_validate.json")
    p.add_argument(
        "--metric",
        type=str,
        default="geodesic_dijkstra",
        choices=["geodesic_dijkstra", "euclidean3d"],
        help="FPS metric (overrides cache filename if it doesn't match the cache).",
    )
    p.add_argument(
        "--surface-variant",
        type=str,
        default="midthickness",
        choices=["midthickness", "pial", "white", "inflated"],
        dest="surface_variant",
        help=(
            "Surface variant used for geodesic FPS. Default is midthickness, which "
            "sits between pial and white and is the conventional choice for distance "
            "computations."
        ),
    )
    p.add_argument(
        "--surface-msm",
        type=str,
        default="MSMAll",
        choices=["MSMAll", "MSMSulc", "none"],
        dest="surface_msm",
        help=(
            "Surface registration suffix. MSMAll matches the Atlas_MSMAll dtseries; "
            "use MSMSulc or 'none' (FreeSurfer-aligned) only if the MSMAll surface "
            "isn't available on the local datalad pull. Note: mesh topology is "
            "identical across registrations; only vertex coordinates differ slightly."
        ),
    )
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    dtseries_path = cfg.data.dtseries_pattern.format(
        subject=args.subject, run=args.task
    )
    surface_dir = cfg.data.surface_dir_template.format(subject=args.subject)
    msm_suffix = "" if args.surface_msm == "none" else f"_{args.surface_msm}"
    lh_mesh = (
        f"{surface_dir}/{args.subject}.L.{args.surface_variant}{msm_suffix}.32k_fs_LR.surf.gii"
    )
    rh_mesh = (
        f"{surface_dir}/{args.subject}.R.{args.surface_variant}{msm_suffix}.32k_fs_LR.surf.gii"
    )

    print(f"[day1] loading dtseries: {dtseries_path}")
    data, header = load_dtseries(dtseries_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)
    cortex_data = extract_cortex_grayordinates(data, header)
    print(
        f"[day1]   dtseries shape={data.shape} (full grayordinates); "
        f"cortex shape={cortex_data.shape} "
        f"(n_lh_cortex={len(cortex_lh)}, n_rh_cortex={len(cortex_rh)})"
    )

    print(f"[day1] building/loading patches → {cfg.tokenize.patch_cache}")
    t_patches_start = time.perf_counter()
    assignment = build_or_load_patches(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=cfg.tokenize.patch_cache,
        n_patches=cfg.tokenize.n_patches_cortex,
        seed=cfg.tokenize.fps_seed,
        metric=args.metric,
    )
    t_patches = time.perf_counter() - t_patches_start
    print(f"[day1]   patch build/load took {t_patches:.2f} s")

    sizes = np.bincount(assignment, minlength=cfg.tokenize.n_patches_cortex)
    print(
        f"[day1] patch sizes: mean={sizes.mean():.2f}, std={sizes.std():.2f}, "
        f"min={int(sizes.min())}, max={int(sizes.max())}"
    )

    patcher = Patcher(
        torch.from_numpy(assignment), n_patches=cfg.tokenize.n_patches_cortex
    )
    x = torch.from_numpy(cortex_data)

    t0 = time.perf_counter()
    patch_means = patcher.forward(x)
    t_tokenize = time.perf_counter() - t0
    print(
        f"[day1] tokenize {tuple(x.shape)} -> {tuple(patch_means.shape)} "
        f"in {t_tokenize:.3f} s"
    )

    # Float32 round-trip — realistic pipeline cost. On raw HCP BOLD (O(5000)),
    # the residual is dominated by index_add_ accumulation rounding:
    # ~ scale * patch_size * eps_f32 ~ 5000 * 200 * 1.2e-7 ~ 0.1.
    reconstructed = patch_means[:, assignment]
    patch_means_2 = patcher.forward(reconstructed)
    residual_f32 = (patch_means - patch_means_2).abs().max().item()

    # Float64 round-trip — algorithm correctness check, decoupled from data scale.
    # Should be ~ 1e-10 if the patcher implements scatter-mean correctly.
    x64 = x.to(torch.float64)
    pm64_1 = patcher.forward(x64)
    pm64_2 = patcher.forward(pm64_1[:, assignment])
    residual_f64 = (pm64_1 - pm64_2).abs().max().item()

    print(f"[day1] round-trip residual (float32, raw scale): {residual_f32:.3e}")
    print(
        f"[day1] round-trip residual (float64, algo-correctness): {residual_f64:.3e}"
    )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(
            {
                "subject": args.subject,
                "task": args.task,
                "metric": args.metric,
                "n_tr": int(data.shape[0]),
                "n_grayordinates": int(data.shape[1]),
                "n_patches": int(cfg.tokenize.n_patches_cortex),
                "patch_size_mean": float(sizes.mean()),
                "patch_size_std": float(sizes.std()),
                "patch_size_min": int(sizes.min()),
                "patch_size_max": int(sizes.max()),
                "tokenize_seconds": float(t_tokenize),
                "patch_build_seconds": float(t_patches),
                "round_trip_max_abs_residual_f32": float(residual_f32),
                "round_trip_max_abs_residual_f64": float(residual_f64),
            },
            f,
            indent=2,
        )
    print(f"[day1] wrote {args.out_json}")

    print(f"[day1] writing patch visualization → {args.out_fig}")
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    _render_patch_figure(assignment, lh_mesh, rh_mesh, cortex_lh, cortex_rh, args.out_fig)

    return 0


def _render_patch_figure(
    assignment: np.ndarray,
    lh_mesh: str,
    rh_mesh: str,
    cortex_lh: np.ndarray,
    cortex_rh: np.ndarray,
    out: str,
) -> None:
    """Save a 2-panel LH/RH cortex figure colored by patch ID."""
    import matplotlib.pyplot as plt
    from boldcast.io.cifti import load_gifti_surface
    from matplotlib.colors import ListedColormap

    n_lh = cortex_lh.shape[0]
    lh_assignment = assignment[:n_lh]
    rh_assignment = assignment[n_lh:]

    rng = np.random.default_rng(0)
    n_patches = int(assignment.max()) + 1
    cmap = ListedColormap(rng.random((n_patches, 3)))

    fig = plt.figure(figsize=(12, 6))
    for col, (mesh_path, hem_assignment, cortex_idx, title) in enumerate(
        [
            (lh_mesh, lh_assignment, cortex_lh, "LH"),
            (rh_mesh, rh_assignment, cortex_rh, "RH"),
        ]
    ):
        verts, faces = load_gifti_surface(mesh_path)
        per_vertex = np.full(verts.shape[0], -1, dtype=np.int32)
        per_vertex[cortex_idx] = hem_assignment

        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        face_color = per_vertex[faces[:, 0]].astype(np.float32)
        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            array=face_color,
            cmap=cmap,
            shade=False,
            linewidth=0,
        )
        ax.set_title(title)
        ax.set_axis_off()

    fig.suptitle(f"Day 1: {n_patches} cortical patches (random colormap)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
