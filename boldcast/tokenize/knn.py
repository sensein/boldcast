"""Cortical-patch kNN adjacency precompute.

Computes per-patch 3D centroids from cortex-grayordinate mesh coordinates,
then returns the ``k`` nearest patches (including self) for each patch by
Euclidean distance. Cached on disk with metadata-keyed invalidation,
mirroring ``boldcast/tokenize/geodesic.py``.

Self-contained: depends only on numpy + the Day-1 ``load_gifti_surface``
helper. No torch / no boldcast.models imports — runs cleanly under the
uv login-node venv.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from boldcast._upstream.cifti_io import load_gifti_surface

__all__ = ["build_or_load_knn"]


def _short_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def build_or_load_knn(
    mesh_lh_path: str,
    mesh_rh_path: str,
    cortex_indices_lh: NDArray[np.integer],
    cortex_indices_rh: NDArray[np.integer],
    patch_assignment: NDArray[np.integer],
    n_patches: int,
    k: int,
    cache_path: str,
) -> NDArray[np.int64]:
    """Return ``(P, k) int64`` per-patch nearest-neighbor indices.

    Each row begins with the patch's own index (``adjacency[i, 0] == i``),
    followed by its ``k-1`` nearest other patches by Euclidean distance
    between patch centroids in 3D coordinate space.

    Parameters
    ----------
    mesh_lh_path, mesh_rh_path : str
        Paths to ``*.surf.gii`` files for the two cortical hemispheres.
    cortex_indices_lh, cortex_indices_rh : ndarray of int
        Mesh-vertex indices for the cortex grayordinates per hemisphere.
    patch_assignment : ndarray of shape ``(V_cortex,)`` int
        Day-1 ``boldcast.tokenize.geodesic`` per-grayordinate patch ID.
    n_patches : int
    k : int
    cache_path : str
        Cache file (``.npz``). Mismatching metadata raises ``ValueError``.

    Returns
    -------
    adjacency : ndarray of shape ``(n_patches, k)`` int64
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if k > n_patches:
        raise ValueError(f"k={k} exceeds n_patches={n_patches}")

    cache = Path(cache_path)
    assignment_arr = np.asarray(patch_assignment, dtype=np.int64)
    metadata: dict[str, int | str] = {
        "n_patches": int(n_patches),
        "k": int(k),
        "patch_assignment_sha": _short_sha(assignment_arr.tobytes()),
    }

    if cache.exists():
        loaded = np.load(cache, allow_pickle=False)
        cached_meta: dict[str, int | str] = {
            key: loaded[key].item() for key in metadata if key in loaded
        }
        if cached_meta != metadata:
            raise ValueError(
                f"cache metadata mismatch at {cache}: "
                f"requested {metadata}, cached {cached_meta}. "
                "Delete the cache file to rebuild."
            )
        adjacency: NDArray[np.int64] = loaded["adjacency"]
        return adjacency

    # Build path: per-patch 3D centroids, then kNN.
    verts_lh, _ = load_gifti_surface(mesh_lh_path)
    verts_rh, _ = load_gifti_surface(mesh_rh_path)
    cortex_lh_arr = np.asarray(cortex_indices_lh, dtype=np.int64)
    cortex_rh_arr = np.asarray(cortex_indices_rh, dtype=np.int64)
    # Stack all cortex vertices in the same order as patch_assignment was
    # built in Day-1 (LH then RH).
    cortex_coords = np.concatenate([verts_lh[cortex_lh_arr], verts_rh[cortex_rh_arr]], axis=0)
    if cortex_coords.shape[0] != assignment_arr.shape[0]:
        raise ValueError(
            f"cortex vertex count {cortex_coords.shape[0]} does not match "
            f"patch_assignment length {assignment_arr.shape[0]}"
        )
    centroids = np.zeros((n_patches, 3), dtype=np.float64)
    counts = np.zeros(n_patches, dtype=np.int64)
    for v_idx in range(assignment_arr.shape[0]):
        p_id = int(assignment_arr[v_idx])
        centroids[p_id] += cortex_coords[v_idx]
        counts[p_id] += 1
    if (counts == 0).any():
        empty = np.where(counts == 0)[0].tolist()
        raise ValueError(f"empty patch(es) in assignment: {empty[:5]}...")
    centroids /= counts[:, None]

    adjacency_out = _compute_knn_from_centroids(centroids.astype(np.float32), k=k)

    cache.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, object] = {"adjacency": adjacency_out}
    for key, value in metadata.items():
        save_kwargs[key] = np.asarray(value)
    np.savez(str(cache), **save_kwargs)  # type: ignore[arg-type]
    return adjacency_out


def _compute_knn_from_centroids(centroids: NDArray[np.floating], k: int) -> NDArray[np.int64]:
    """For each row of ``centroids``, return the ``k`` nearest rows (including
    self at index 0) by Euclidean distance."""
    diff = centroids[:, None, :] - centroids[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    order = np.argsort(dist, axis=1, kind="stable")
    out: NDArray[np.int64] = order[:, :k].astype(np.int64)
    return out
