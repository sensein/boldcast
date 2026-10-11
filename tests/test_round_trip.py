"""Day-1 acceptance: dtseries → Patcher → de-patch → Patcher reproduces patch means."""

from __future__ import annotations

from collections.abc import Callable
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


def test_patch_mean_idempotency_on_synthetic_data(
    synthetic_dtseries: Path,
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    synthetic_gifti_path_factory: Callable[[tuple[np.ndarray, np.ndarray], str], Path],
    tmp_path: Path,
) -> None:
    lh_path = synthetic_gifti_path_factory(synthetic_mesh_lh, "lh")
    rh_path = synthetic_gifti_path_factory(synthetic_mesh_rh, "rh")

    data, header = load_dtseries(str(synthetic_dtseries))
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)
    cortex_data = extract_cortex_grayordinates(data, header)
    n_patches = 8

    assignment = build_or_load_patches(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(tmp_path / "patches.npz"),
        n_patches=n_patches,
        seed=0,
        metric="euclidean3d",
    )
    patcher = Patcher(torch.from_numpy(assignment), n_patches=n_patches)

    x = torch.from_numpy(cortex_data)
    patch_means_1 = patcher.forward(x)  # (T, P)

    # De-patch: scatter patch means back to grayordinates by patch ID.
    reconstructed = patch_means_1[:, assignment]  # (T, V_cortex)
    patch_means_2 = patcher.forward(reconstructed)

    torch.testing.assert_close(patch_means_1, patch_means_2, rtol=1e-6, atol=1e-6)


def test_patch_mean_round_trip_float64_on_large_scale(
    synthetic_dtseries: Path,
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    synthetic_gifti_path_factory: Callable[[tuple[np.ndarray, np.ndarray], str], Path],
    tmp_path: Path,
) -> None:
    """Algorithm-correctness invariant: float64 round-trip residual must be
    near-zero regardless of data scale. The float32 round-trip residual is
    data-scale-dependent (``scale × patch_size × ε_f32``), but float64
    decouples algorithm correctness from accumulator precision. Locks in
    the invariant the day-1 validation script relies on (see
    ``scripts/day1_validate_tokenizer.py``).
    """
    lh_path = synthetic_gifti_path_factory(synthetic_mesh_lh, "lh")
    rh_path = synthetic_gifti_path_factory(synthetic_mesh_rh, "rh")

    data, header = load_dtseries(str(synthetic_dtseries))
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)
    cortex_data = extract_cortex_grayordinates(data, header)
    n_patches = 8

    assignment = build_or_load_patches(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(tmp_path / "patches.npz"),
        n_patches=n_patches,
        seed=0,
        metric="euclidean3d",
    )
    patcher = Patcher(torch.from_numpy(assignment), n_patches=n_patches)

    # Scale to ~5000 to mimic raw HCP BOLD magnitude — the regime where
    # float32 residuals would be ~0.05 but float64 should still be ~1e-10.
    x = torch.from_numpy(cortex_data).to(torch.float64) * 5000.0
    pm1 = patcher.forward(x)
    pm2 = patcher.forward(pm1[:, assignment])
    residual = (pm1 - pm2).abs().max().item()
    assert residual < 1e-9, (
        f"float64 round-trip residual {residual:.3e} exceeds 1e-9 — "
        "scatter-mean algorithm is not exact in double precision"
    )
