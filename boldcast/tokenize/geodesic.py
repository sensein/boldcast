"""Project-side cortical-patch tokenizer cache wrapper.

Wraps :func:`boldcast._upstream.geodesic_patcher.precompute_patches` with a
versioned cache keyed on ``(n_patches, seed, metric, lloyd_iters,
hemisphere-cortex sizes, mesh-file SHAs)``. Mesh SHAs guard against silent
stale-cache reuse when the surface variant (midthickness / pial / inflated)
or MSM registration changes under a fixed cache path. Metadata mismatch
raises rather than silently rebuilding so cache invalidation is always
explicit (delete the file).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

import numpy as np

from boldcast._upstream.cifti_io import load_gifti_surface
from boldcast._upstream.geodesic_patcher import precompute_patches

__all__ = ["build_or_load_patches"]

Metric = Literal["geodesic_dijkstra", "euclidean3d"]


def _mesh_file_sha(path: str) -> str:
    """Short SHA-256 of mesh file bytes, used to fingerprint the surface
    in the cache metadata. Catches surface-variant or MSM-registration
    swaps that would otherwise silently reuse stale patches."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build_or_load_patches(
    mesh_lh_path: str,
    mesh_rh_path: str,
    cortex_indices_lh: np.ndarray,
    cortex_indices_rh: np.ndarray,
    cache_path: str,
    n_patches: int = 1024,
    seed: int = 0,
    metric: Metric = "geodesic_dijkstra",
    lloyd_iters: int = 10,
) -> np.ndarray:
    """Return per-grayordinate patch IDs, building and caching on first call.

    Parameters
    ----------
    mesh_lh_path, mesh_rh_path : str
        Paths to GIFTI surface meshes (``*.surf.gii``).
    cortex_indices_lh, cortex_indices_rh : ndarray
        Mesh-vertex indices for cortex grayordinates per hemisphere.
    cache_path : str
        Where the per-grayordinate assignment ``.npz`` is read/written.
    n_patches, seed, metric, lloyd_iters
        Forwarded to ``precompute_patches`` and stored in the cache as
        metadata. A cached file with mismatching metadata raises
        ``ValueError`` rather than silently rebuilding.

    Returns
    -------
    patch_assignment : ndarray of shape ``(V_cortex_lh + V_cortex_rh,)`` int32
    """
    cache = Path(cache_path)
    metadata: dict[str, int | str] = {
        "n_patches": int(n_patches),
        "seed": int(seed),
        "metric": metric,
        "lloyd_iters": int(lloyd_iters),
        "n_lh_cortex": int(cortex_indices_lh.shape[0]),
        "n_rh_cortex": int(cortex_indices_rh.shape[0]),
        "mesh_lh_sha": _mesh_file_sha(mesh_lh_path),
        "mesh_rh_sha": _mesh_file_sha(mesh_rh_path),
    }
    if cache.exists():
        loaded = np.load(cache, allow_pickle=False)
        cached_meta = {k: loaded[k].item() for k in metadata if k in loaded}
        if cached_meta != metadata:
            raise ValueError(
                f"cache metadata mismatch at {cache}: "
                f"requested {metadata}, cached {cached_meta}. "
                "Delete the cache file to rebuild."
            )
        cached_assignment: np.ndarray = loaded["assignment"]
        return cached_assignment

    verts_lh, faces_lh = load_gifti_surface(mesh_lh_path)
    verts_rh, faces_rh = load_gifti_surface(mesh_rh_path)
    assignment = precompute_patches(
        mesh_lh=(verts_lh, faces_lh),
        mesh_rh=(verts_rh, faces_rh),
        cortex_indices_lh=cortex_indices_lh,
        cortex_indices_rh=cortex_indices_rh,
        n_patches=n_patches,
        seed=seed,
        metric=metric,
        lloyd_iters=lloyd_iters,
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "assignment": assignment,
        **{k: np.asarray(v) for k, v in metadata.items()},
    }
    # numpy's savez stub typing is loose; runtime accepts arbitrary array kwargs.
    np.savez(str(cache), **arrays)  # type: ignore[arg-type]
    return assignment
