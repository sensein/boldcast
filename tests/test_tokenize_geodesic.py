"""Tests for ``boldcast/tokenize/geodesic.py`` cache wrapper."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from boldcast.tokenize.geodesic import build_or_load_patches


def _save_synthetic_gifti(
    tmp_path: Path, mesh: tuple[np.ndarray, np.ndarray], name: str
) -> Path:
    verts, faces = mesh
    gii = nib.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            verts.astype(np.float32), intent="NIFTI_INTENT_POINTSET"
        )
    )
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"
        )
    )
    out = tmp_path / f"{name}.surf.gii"
    nib.save(gii, str(out))
    return out


def test_cache_miss_then_hit_uses_cache(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    lh_path = _save_synthetic_gifti(tmp_path, synthetic_mesh_lh, "lh")
    rh_path = _save_synthetic_gifti(tmp_path, synthetic_mesh_rh, "rh")
    cache = tmp_path / "patches.npz"

    n_lh = synthetic_mesh_lh[0].shape[0]
    n_rh = synthetic_mesh_rh[0].shape[0]
    a1 = build_or_load_patches(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=np.arange(n_lh),
        cortex_indices_rh=np.arange(n_rh),
        cache_path=str(cache),
        n_patches=8,
        seed=0,
        metric="euclidean3d",
    )
    assert cache.exists()
    a2 = build_or_load_patches(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=np.arange(n_lh),
        cortex_indices_rh=np.arange(n_rh),
        cache_path=str(cache),
        n_patches=8,
        seed=0,
        metric="euclidean3d",
    )
    np.testing.assert_array_equal(a1, a2)


def test_cache_metadata_mismatch_raises(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    tmp_path: Path,
) -> None:
    lh_path = _save_synthetic_gifti(tmp_path, synthetic_mesh_lh, "lh")
    rh_path = _save_synthetic_gifti(tmp_path, synthetic_mesh_rh, "rh")
    cache = tmp_path / "patches.npz"
    n_lh = synthetic_mesh_lh[0].shape[0]
    n_rh = synthetic_mesh_rh[0].shape[0]

    build_or_load_patches(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=np.arange(n_lh),
        cortex_indices_rh=np.arange(n_rh),
        cache_path=str(cache),
        n_patches=8,
        seed=0,
        metric="euclidean3d",
    )
    with pytest.raises(ValueError, match="cache metadata mismatch"):
        build_or_load_patches(
            mesh_lh_path=str(lh_path),
            mesh_rh_path=str(rh_path),
            cortex_indices_lh=np.arange(n_lh),
            cortex_indices_rh=np.arange(n_rh),
            cache_path=str(cache),
            n_patches=8,
            seed=1,  # different seed
            metric="euclidean3d",
        )
