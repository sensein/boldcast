"""Tests for ``boldcast/tokenize/geodesic.py`` cache wrapper."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from boldcast.tokenize.geodesic import build_or_load_patches


def test_cache_miss_then_hit_uses_cache(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    synthetic_gifti_path_factory: Callable[[tuple[np.ndarray, np.ndarray], str], Path],
    tmp_path: Path,
) -> None:
    lh_path = synthetic_gifti_path_factory(synthetic_mesh_lh, "lh")
    rh_path = synthetic_gifti_path_factory(synthetic_mesh_rh, "rh")
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


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("seed", 1),
        ("n_patches", 16),
        ("metric", "geodesic_dijkstra"),
        ("lloyd_iters", 5),
        ("n_lh_cortex_drop", 10),
        ("n_rh_cortex_drop", 10),
    ],
)
def test_cache_metadata_mismatch_raises(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    synthetic_gifti_path_factory: Callable[[tuple[np.ndarray, np.ndarray], str], Path],
    tmp_path: Path,
    field: str,
    bad_value: int | str,
) -> None:
    lh_path = synthetic_gifti_path_factory(synthetic_mesh_lh, "lh")
    rh_path = synthetic_gifti_path_factory(synthetic_mesh_rh, "rh")
    cache = tmp_path / "patches.npz"
    n_lh = synthetic_mesh_lh[0].shape[0]
    n_rh = synthetic_mesh_rh[0].shape[0]

    base_kwargs: dict[str, object] = dict(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=np.arange(n_lh),
        cortex_indices_rh=np.arange(n_rh),
        cache_path=str(cache),
        n_patches=8,
        seed=0,
        metric="euclidean3d",
    )
    build_or_load_patches(**base_kwargs)  # type: ignore[arg-type]

    bad_kwargs = dict(base_kwargs)
    if field == "n_lh_cortex_drop":
        assert isinstance(bad_value, int)
        bad_kwargs["cortex_indices_lh"] = np.arange(n_lh - bad_value)
    elif field == "n_rh_cortex_drop":
        assert isinstance(bad_value, int)
        bad_kwargs["cortex_indices_rh"] = np.arange(n_rh - bad_value)
    else:
        bad_kwargs[field] = bad_value

    with pytest.raises(ValueError, match="cache metadata mismatch"):
        build_or_load_patches(**bad_kwargs)  # type: ignore[arg-type]


def test_cache_mismatch_when_mesh_content_changes(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    synthetic_gifti_path_factory: Callable[[tuple[np.ndarray, np.ndarray], str], Path],
    tmp_path: Path,
) -> None:
    """Swapping the mesh file contents (e.g. midthickness → pial, or different
    MSM registration) under the same cache path must raise — otherwise stale
    patches from a different surface silently win the cache.
    """
    lh_path = synthetic_gifti_path_factory(synthetic_mesh_lh, "lh")
    rh_path = synthetic_gifti_path_factory(synthetic_mesh_rh, "rh")
    cache = tmp_path / "patches.npz"
    n_lh = synthetic_mesh_lh[0].shape[0]
    n_rh = synthetic_mesh_rh[0].shape[0]

    base_kwargs: dict[str, object] = dict(
        mesh_lh_path=str(lh_path),
        mesh_rh_path=str(rh_path),
        cortex_indices_lh=np.arange(n_lh),
        cortex_indices_rh=np.arange(n_rh),
        cache_path=str(cache),
        n_patches=8,
        seed=0,
        metric="euclidean3d",
    )
    build_or_load_patches(**base_kwargs)  # type: ignore[arg-type]

    # Same mesh shape, different vertex coordinates — mimics swapping
    # midthickness → pial (topology preserved, coords shifted).
    lh_verts, lh_faces = synthetic_mesh_lh
    perturbed_lh = (lh_verts + np.float32(0.01), lh_faces)
    perturbed_lh_path = synthetic_gifti_path_factory(perturbed_lh, "lh_perturbed")

    bad_kwargs = dict(base_kwargs)
    bad_kwargs["mesh_lh_path"] = str(perturbed_lh_path)
    with pytest.raises(ValueError, match="cache metadata mismatch"):
        build_or_load_patches(**bad_kwargs)  # type: ignore[arg-type]
