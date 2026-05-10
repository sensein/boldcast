"""Tests for ``boldcast/tokenize/knn.py``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from boldcast.tokenize.knn import build_or_load_knn


def test_knn_shape_and_self_inclusion(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    synthetic_gifti_path_factory: Callable[
        [tuple[np.ndarray, np.ndarray], str], Path
    ],
    tmp_path: Path,
) -> None:
    """kNN output is (P, k) int; row i contains i (self-link first)."""
    lh_path = synthetic_gifti_path_factory(synthetic_mesh_lh, "lh")
    rh_path = synthetic_gifti_path_factory(synthetic_mesh_rh, "rh")

    n_lh = synthetic_mesh_lh[0].shape[0]
    n_rh = synthetic_mesh_rh[0].shape[0]
    n_patches = 8
    # Round-robin assignment over all mesh vertices (n_lh + n_rh).
    assignment = np.arange(n_lh + n_rh, dtype=np.int32) % n_patches

    adjacency = build_or_load_knn(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=np.arange(n_lh),
        cortex_indices_rh=np.arange(n_rh),
        patch_assignment=assignment,
        n_patches=n_patches,
        k=4,
        cache_path=str(tmp_path / "knn.npz"),
    )
    assert adjacency.shape == (n_patches, 4)
    assert adjacency.dtype.kind == "i"
    # Self-link first: row i's first entry is i.
    for i in range(n_patches):
        assert int(adjacency[i, 0]) == i, (
            f"row {i} should begin with self, got {adjacency[i].tolist()}"
        )
    # Indices are in range
    assert int(adjacency.min()) >= 0
    assert int(adjacency.max()) < n_patches


def test_knn_cache_metadata_mismatch_raises(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    synthetic_gifti_path_factory: Callable[
        [tuple[np.ndarray, np.ndarray], str], Path
    ],
    tmp_path: Path,
) -> None:
    """Different k or assignment → cache mismatch raises ValueError."""
    lh_path = synthetic_gifti_path_factory(synthetic_mesh_lh, "lh")
    rh_path = synthetic_gifti_path_factory(synthetic_mesh_rh, "rh")
    n_lh = synthetic_mesh_lh[0].shape[0]
    n_rh = synthetic_mesh_rh[0].shape[0]
    n_patches = 8
    assignment = np.arange(n_lh + n_rh, dtype=np.int32) % n_patches

    cache = tmp_path / "knn.npz"
    base_kwargs: dict[str, object] = dict(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=np.arange(n_lh),
        cortex_indices_rh=np.arange(n_rh),
        patch_assignment=assignment,
        n_patches=n_patches,
        k=4,
        cache_path=str(cache),
    )
    build_or_load_knn(**base_kwargs)  # type: ignore[arg-type]

    # Changing k must raise.
    bad = dict(base_kwargs)
    bad["k"] = 6
    with pytest.raises(ValueError, match="cache metadata mismatch"):
        build_or_load_knn(**bad)  # type: ignore[arg-type]

    # Changing the assignment must raise.
    rotated = (assignment + 1) % n_patches
    bad2 = dict(base_kwargs)
    bad2["patch_assignment"] = rotated.astype(np.int32)
    with pytest.raises(ValueError, match="cache metadata mismatch"):
        build_or_load_knn(**bad2)  # type: ignore[arg-type]


def test_knn_neighbors_are_geometrically_closest() -> None:
    """On a deterministic 4-patch grid, the kNN of each patch should be the
    two adjacent grid cells (plus self)."""
    # Build a 2x2 grid of patches: patches at (0,0,0), (1,0,0), (0,1,0), (1,1,0).
    # Each patch is one mesh vertex.
    verts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    from boldcast.tokenize.knn import _compute_knn_from_centroids

    centroids = verts.copy()
    n_patches = 4
    adjacency = _compute_knn_from_centroids(centroids, k=3)
    assert adjacency.shape == (n_patches, 3)
    # Row 0 (patch at origin) closest: self, (1,0,0) or (0,1,0) tied, then (1,1,0)
    assert adjacency[0, 0] == 0
    assert set(adjacency[0, 1:].tolist()) <= {1, 2, 3}
    # With k=3 and 4 patches, row 0 = [0, near1, near2]; the diagonal (3) is
    # excluded because the two grid-adjacent neighbors (1, 2) are closer.
    assert 3 not in adjacency[0].tolist(), (
        f"row 0 should pick the two grid-adjacent neighbors, not the diagonal: "
        f"{adjacency[0].tolist()}"
    )
