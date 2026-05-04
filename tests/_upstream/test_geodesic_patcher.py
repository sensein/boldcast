"""Tests for ``boldcast/_upstream/geodesic_patcher.py``."""

from __future__ import annotations

import numpy as np
import pytest
from boldcast._upstream.geodesic_patcher import precompute_patches


@pytest.mark.parametrize("metric", ["geodesic_dijkstra", "euclidean3d"])
def test_patch_assignment_shape_and_range(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    metric: str,
) -> None:
    n_patches = 8
    assignment = precompute_patches(
        mesh_lh=synthetic_mesh_lh,
        mesh_rh=synthetic_mesh_rh,
        cortex_indices_lh=np.arange(synthetic_mesh_lh[0].shape[0]),
        cortex_indices_rh=np.arange(synthetic_mesh_rh[0].shape[0]),
        n_patches=n_patches,
        seed=0,
        metric=metric,
    )
    n_lh_verts = synthetic_mesh_lh[0].shape[0]
    n_rh_verts = synthetic_mesh_rh[0].shape[0]
    assert assignment.shape == (n_lh_verts + n_rh_verts,)
    assert assignment.dtype.kind == "i"
    assert int(assignment.min()) >= 0
    assert int(assignment.max()) < n_patches


def test_per_hemisphere_id_offset(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
) -> None:
    n_patches = 8
    assignment = precompute_patches(
        mesh_lh=synthetic_mesh_lh,
        mesh_rh=synthetic_mesh_rh,
        cortex_indices_lh=np.arange(synthetic_mesh_lh[0].shape[0]),
        cortex_indices_rh=np.arange(synthetic_mesh_rh[0].shape[0]),
        n_patches=n_patches,
        seed=0,
    )
    n_lh_verts = synthetic_mesh_lh[0].shape[0]
    assert int(assignment[:n_lh_verts].max()) < n_patches // 2
    assert int(assignment[n_lh_verts:].min()) >= n_patches // 2


def test_no_empty_patches(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
) -> None:
    n_patches = 8
    assignment = precompute_patches(
        mesh_lh=synthetic_mesh_lh,
        mesh_rh=synthetic_mesh_rh,
        cortex_indices_lh=np.arange(synthetic_mesh_lh[0].shape[0]),
        cortex_indices_rh=np.arange(synthetic_mesh_rh[0].shape[0]),
        n_patches=n_patches,
        seed=0,
    )
    counts = np.bincount(assignment, minlength=n_patches)
    assert int(counts.min()) > 0


def test_seed_determinism(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
) -> None:
    a1 = precompute_patches(
        synthetic_mesh_lh,
        synthetic_mesh_rh,
        np.arange(synthetic_mesh_lh[0].shape[0]),
        np.arange(synthetic_mesh_rh[0].shape[0]),
        n_patches=8,
        seed=42,
    )
    a2 = precompute_patches(
        synthetic_mesh_lh,
        synthetic_mesh_rh,
        np.arange(synthetic_mesh_lh[0].shape[0]),
        np.arange(synthetic_mesh_rh[0].shape[0]),
        n_patches=8,
        seed=42,
    )
    np.testing.assert_array_equal(a1, a2)


def test_invalid_metric_raises(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="metric"):
        precompute_patches(
            synthetic_mesh_lh,
            synthetic_mesh_rh,
            np.arange(synthetic_mesh_lh[0].shape[0]),
            np.arange(synthetic_mesh_rh[0].shape[0]),
            n_patches=8,
            seed=0,
            metric="not_a_real_metric",  # type: ignore[arg-type]
        )


def test_odd_n_patches_raises(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="even"):
        precompute_patches(
            synthetic_mesh_lh,
            synthetic_mesh_rh,
            np.arange(synthetic_mesh_lh[0].shape[0]),
            np.arange(synthetic_mesh_rh[0].shape[0]),
            n_patches=7,
            seed=0,
        )
