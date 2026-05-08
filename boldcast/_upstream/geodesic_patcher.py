"""Per-hemisphere geodesic farthest-point sampling for cortical patch tokenization.

Self-contained module: no imports from any other ``boldcast`` submodule.
Targeted for upstream contribution to ``nobrainer.layers`` (see
``boldcast/_upstream/README.md``).

Two metrics are supported:

* ``"geodesic_dijkstra"`` — edge-graph Dijkstra on the mesh, weighted by
  Euclidean distance between adjacent vertices. Default. Methodologically
  faithful to the cortical sheet (distances do not jump across sulci).
  One-time cost on a ``32k_fs_LR`` hemisphere is ~1–3 minutes.
* ``"euclidean3d"`` — vectorized FPS in 3D Euclidean space on vertex
  coordinates. Sub-second runtime; documented fallback if the geodesic
  build is intolerable. Distances jump across sulci.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

__all__ = ["precompute_patches"]

Metric = Literal["geodesic_dijkstra", "euclidean3d"]


def precompute_patches(
    mesh_lh: tuple[np.ndarray, np.ndarray],
    mesh_rh: tuple[np.ndarray, np.ndarray],
    cortex_indices_lh: np.ndarray,
    cortex_indices_rh: np.ndarray,
    n_patches: int = 1024,
    seed: int = 0,
    metric: Metric = "geodesic_dijkstra",
    lloyd_iters: int = 10,
) -> np.ndarray:
    """Compute per-grayordinate cortical patch IDs via per-hemisphere FPS+Lloyd.

    FPS runs on the full hemisphere mesh, then patch IDs are subset to the
    cortex-grayordinate vertices. This keeps geodesic distances accurate
    across the medial-wall boundary while ensuring the returned array is
    indexed by grayordinate, ready for ``boldcast.tokenize.patcher.Patcher``.

    After FPS, optional **Lloyd relaxation** (default 10 iterations with
    early-stop on convergence) shifts each source toward the 3D centroid
    of its current patch members, dramatically reducing patch-size variance
    on non-uniform meshes (e.g. real cortex). Lloyd is run in 3D Euclidean
    space (cheap) and the final per-vertex assignment is done in the
    requested ``metric``, so geodesic boundaries are preserved while
    benefiting from the Lloyd-balanced source layout.

    Parameters
    ----------
    mesh_lh, mesh_rh
        ``(vertices, faces)`` tuples for each hemisphere. ``vertices`` shape
        ``(V_h, 3)`` float, ``faces`` shape ``(F_h, 3)`` int.
    cortex_indices_lh, cortex_indices_rh
        Mesh-vertex indices that map cortex grayordinates onto the parent
        mesh (typically excludes medial-wall vertices). Shape
        ``(V_cortex_h,)`` int.
    n_patches
        Total cortical patches across both hemispheres. Must be even
        (``n_patches // 2`` per hemisphere).
    seed
        Seed for the FPS first-source pick.
    metric
        ``"geodesic_dijkstra"`` (default) or ``"euclidean3d"``.
    lloyd_iters
        Maximum Lloyd iterations after FPS (default 10). Use ``0`` for
        pure FPS without relaxation. Lloyd typically converges in 3–10
        iterations and is early-stopped when the source set stabilizes.

    Returns
    -------
    patch_assignment : ndarray of shape ``(V_cortex_lh + V_cortex_rh,)`` int32
        Per-grayordinate patch ID. LH grayordinates get IDs in
        ``[0, n_patches // 2)``; RH grayordinates get IDs in
        ``[n_patches // 2, n_patches)``.

    Notes
    -----
    Each hemisphere mesh must be a **single connected component**. The
    ``32k_fs_LR`` standard meshes satisfy this; multi-component meshes
    (e.g., two icospheres with no shared vertices) will raise a
    ``ValueError`` when the ``"geodesic_dijkstra"`` metric is used.
    Geodesic distances across disconnected components are undefined
    (``scipy`` returns ``inf``), which would produce silent garbage in
    the FPS and assignment steps.
    """
    if metric not in ("geodesic_dijkstra", "euclidean3d"):
        raise ValueError(
            f"Unknown metric {metric!r}; expected 'geodesic_dijkstra' or 'euclidean3d'"
        )
    if n_patches % 2 != 0:
        raise ValueError(f"n_patches must be even, got {n_patches}")
    if lloyd_iters < 0:
        raise ValueError(f"lloyd_iters must be >= 0, got {lloyd_iters}")
    n_per_hem = n_patches // 2

    rng = np.random.default_rng(seed)
    lh_assignment = _fps_one_hemisphere(
        mesh_lh, cortex_indices_lh, n_per_hem, rng, metric, lloyd_iters, hemisphere_offset=0
    )
    rh_assignment = _fps_one_hemisphere(
        mesh_rh, cortex_indices_rh, n_per_hem, rng, metric, lloyd_iters, hemisphere_offset=n_per_hem
    )
    return np.concatenate([lh_assignment, rh_assignment]).astype(np.int32)


def _fps_one_hemisphere(
    mesh: tuple[np.ndarray, np.ndarray],
    cortex_indices: np.ndarray,
    n_patches_hem: int,
    rng: np.random.Generator,
    metric: Metric,
    lloyd_iters: int,
    hemisphere_offset: int,
) -> np.ndarray:
    verts, faces = mesh
    first_source = int(rng.choice(cortex_indices))

    if metric == "geodesic_dijkstra":
        adj = _build_edge_graph(verts, faces)
        n_components, _ = connected_components(adj, directed=False)
        if n_components != 1:
            raise ValueError(
                f"Hemisphere mesh has {n_components} connected components; "
                "FPS requires a single connected component. "
                "Check that the input mesh is a valid closed surface "
                "(e.g., 32k_fs_LR) with no isolated sub-meshes."
            )
        sources, per_source_dists = _fps_dijkstra(
            adj, n_patches_hem, first_source, cortex_indices
        )
        if lloyd_iters > 0:
            sources = _lloyd_relax(verts, sources, cortex_indices, lloyd_iters)
            # Recompute geodesic distances from the relaxed sources for the
            # final assignment step.
            per_source_dists = dijkstra(adj, indices=sources, directed=False)
        per_vertex_assignment = _assign_to_nearest_source_dijkstra(per_source_dists)
    else:
        sources = _fps_euclidean3d(
            verts, n_patches_hem, first_source, cortex_indices
        )
        if lloyd_iters > 0:
            sources = _lloyd_relax(verts, sources, cortex_indices, lloyd_iters)
        per_vertex_assignment = _assign_to_nearest_source_euclidean(verts, sources)

    result: np.ndarray = per_vertex_assignment[cortex_indices] + hemisphere_offset
    return result


def _build_edge_graph(verts: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """Sparse adjacency weighted by Euclidean edge length (symmetric)."""
    edges = np.concatenate(
        [faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0
    )
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)
    edges = np.unique(edges, axis=0)
    weights = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    n = verts.shape[0]
    return csr_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n))


def _fps_dijkstra(
    adj: csr_matrix,
    n_patches: int,
    first_source: int,
    candidate_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Incremental FPS on a graph: one Dijkstra per new source.

    Sources are picked only from ``candidate_indices`` (the cortex
    grayordinate vertices). Picking sources from outside that set —
    e.g., medial-wall vertices in HCP ``32k_fs_LR`` — produces patches
    whose cortex membership can be empty after grayordinate subsetting,
    because the source's nearest cortex neighbours may pick a different,
    geometrically closer source. Restricting source selection to
    ``candidate_indices`` guarantees every patch contains at least its
    own source vertex (a cortex grayordinate).

    Returns
    -------
    sources : ndarray of shape (n_patches,)
    per_source_dists : ndarray of shape (n_patches, n_verts)
        Distance from each source to every vertex. Reused by the
        assignment step to avoid running a second round of Dijkstras.
    """
    n_verts = adj.shape[0]
    sources = np.empty(n_patches, dtype=np.int64)
    per_source_dists = np.empty((n_patches, n_verts), dtype=np.float64)
    sources[0] = first_source
    per_source_dists[0] = dijkstra(adj, indices=first_source, directed=False)
    min_dist = per_source_dists[0].copy()
    for k in range(1, n_patches):
        # argmax over candidate (cortex) vertices only.
        nxt = int(candidate_indices[np.argmax(min_dist[candidate_indices])])
        sources[k] = nxt
        per_source_dists[k] = dijkstra(adj, indices=nxt, directed=False)
        min_dist = np.minimum(min_dist, per_source_dists[k])
    return sources, per_source_dists


def _assign_to_nearest_source_dijkstra(per_source_dists: np.ndarray) -> np.ndarray:
    """For each vertex, return the index of the closest source."""
    out: np.ndarray = np.argmin(per_source_dists, axis=0).astype(np.int32)
    return out


def _fps_euclidean3d(
    verts: np.ndarray,
    n_patches: int,
    first_source: int,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    """Vectorized 3D-Euclidean FPS, restricted to ``candidate_indices`` for source picks.

    See ``_fps_dijkstra`` for why source picks must be cortex-only.
    """
    sources = np.empty(n_patches, dtype=np.int64)
    sources[0] = first_source
    min_dist = np.linalg.norm(verts - verts[first_source], axis=1)
    for k in range(1, n_patches):
        nxt = int(candidate_indices[np.argmax(min_dist[candidate_indices])])
        sources[k] = nxt
        d_new = np.linalg.norm(verts - verts[nxt], axis=1)
        min_dist = np.minimum(min_dist, d_new)
    return sources


def _assign_to_nearest_source_euclidean(
    verts: np.ndarray, sources: np.ndarray
) -> np.ndarray:
    diff = verts[None, :, :] - verts[sources][:, None, :]  # (n_sources, n_verts, 3)
    dists = np.linalg.norm(diff, axis=-1)
    out: np.ndarray = np.argmin(dists, axis=0).astype(np.int32)
    return out


def _lloyd_relax(
    verts: np.ndarray,
    sources: np.ndarray,
    candidate_indices: np.ndarray,
    n_iters: int,
) -> np.ndarray:
    """Lloyd relaxation in 3D Euclidean space — moves sources to patch centroids.

    Each iteration: (1) reassign every vertex to its nearest source by 3D
    Euclidean distance, (2) for each patch, compute the 3D centroid of its
    members and replace the source with the nearest cortex vertex to that
    centroid. Early-stops when the source set stabilizes.

    Lloyd is run in 3D Euclidean rather than along the geodesic graph because
    (i) at the patch scale the surface is approximately locally Euclidean,
    (ii) it's much cheaper than re-running multi-source Dijkstra each
    iteration, and (iii) the final per-vertex assignment is still done in
    the requested metric (geodesic_dijkstra or euclidean3d) by the caller, so
    patch boundaries respect the surface geometry. Empirically reduces
    patch-size std by ~2× on real cortex.

    Parameters
    ----------
    verts : ndarray of shape ``(V, 3)``
        Mesh vertex coordinates.
    sources : ndarray of shape ``(n_patches,)`` int
        Initial source indices (typically from FPS).
    candidate_indices : ndarray of int
        Mesh-vertex indices that sources may be drawn from (cortex only).
    n_iters : int
        Maximum iterations. Early-stop on convergence.

    Returns
    -------
    sources : ndarray of shape ``(n_patches,)`` int64
        Relaxed source indices, all in ``candidate_indices``.
    """
    cand_verts = verts[candidate_indices]  # (V_cand, 3) — precompute once
    sources = sources.astype(np.int64).copy()
    for _ in range(n_iters):
        # Reassignment step: 3D Euclidean argmin per vertex.
        diff = verts[None, :, :] - verts[sources][:, None, :]  # (S, V, 3)
        dists = np.linalg.norm(diff, axis=-1)  # (S, V)
        per_vertex_assignment = np.argmin(dists, axis=0)

        # Centroid step: for each patch, find the cortex vertex closest to the
        # patch's 3D centroid.
        new_sources = sources.copy()
        for k in range(len(sources)):
            members = np.where(per_vertex_assignment == k)[0]
            if len(members) == 0:
                # Source restriction in FPS prevents this on connected meshes,
                # but keep the source unchanged if it ever happens.
                continue
            centroid = verts[members].mean(axis=0)
            new_sources[k] = candidate_indices[
                int(np.argmin(np.linalg.norm(cand_verts - centroid, axis=1)))
            ]

        if np.array_equal(new_sources, sources):
            break  # converged
        sources = new_sources

    return sources
