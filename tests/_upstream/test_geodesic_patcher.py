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


def test_geodesic_and_euclidean_disagree_on_nonconvex_mesh() -> None:
    """On a torus (genus-1), geodesic and Euclidean assignments must differ on >0 vertices."""
    import trimesh

    m = trimesh.creation.torus(major_radius=1.0, minor_radius=0.3,
                                major_sections=16, minor_sections=8)
    verts = m.vertices.astype(np.float32)
    faces = m.faces.astype(np.int32)
    cortex_indices = np.arange(verts.shape[0])

    geo = precompute_patches(
        mesh_lh=(verts, faces),
        mesh_rh=(verts, faces),
        cortex_indices_lh=cortex_indices,
        cortex_indices_rh=cortex_indices,
        n_patches=16,
        seed=0,
        metric="geodesic_dijkstra",
    )
    eu = precompute_patches(
        mesh_lh=(verts, faces),
        mesh_rh=(verts, faces),
        cortex_indices_lh=cortex_indices,
        cortex_indices_rh=cortex_indices,
        n_patches=16,
        seed=0,
        metric="euclidean3d",
    )
    n = cortex_indices.shape[0]
    assert (geo[:n] != eu[:n]).any(), (
        "geodesic and euclidean assignments should differ on at least one vertex of a torus"
    )


@pytest.mark.parametrize("metric", ["geodesic_dijkstra", "euclidean3d"])
def test_no_empty_patches_when_cortex_excludes_some_mesh_vertices(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
    metric: str,
) -> None:
    """Mimics the HCP medial wall: cortex_indices is a strict subset of mesh vertices.

    Without restricting FPS source picks to cortex_indices, sources can land on
    excluded ("medial wall") vertices and produce empty patches after subsetting
    to grayordinates. This test would have caught the empty-patch bug surfaced
    by the Day-1 real-HCP validation on subject 115825.
    """
    n_lh = synthetic_mesh_lh[0].shape[0]
    n_rh = synthetic_mesh_rh[0].shape[0]
    # Drop ~20% of mesh vertices from cortex_indices to simulate a medial wall.
    rng = np.random.default_rng(0)
    medial_wall_lh = rng.choice(n_lh, size=n_lh // 5, replace=False)
    medial_wall_rh = rng.choice(n_rh, size=n_rh // 5, replace=False)
    cortex_lh = np.array(sorted(set(range(n_lh)) - set(medial_wall_lh.tolist())))
    cortex_rh = np.array(sorted(set(range(n_rh)) - set(medial_wall_rh.tolist())))

    n_patches = 8
    assignment = precompute_patches(
        mesh_lh=synthetic_mesh_lh,
        mesh_rh=synthetic_mesh_rh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        n_patches=n_patches,
        seed=0,
        metric=metric,
    )
    counts = np.bincount(assignment, minlength=n_patches)
    assert int(counts.min()) > 0, (
        f"empty patch(es) when cortex_indices excludes some mesh vertices "
        f"({metric}): counts={counts.tolist()}"
    )


@pytest.mark.parametrize("metric", ["geodesic_dijkstra", "euclidean3d"])
def test_lloyd_reduces_patch_size_variance_on_squashed_mesh(metric: str) -> None:
    """Lloyd-relaxation must lower patch-size std on a non-uniform mesh.

    A heavily squashed icosphere has high vertex density near the poles —
    plain FPS produces uneven patch sizes there. Lloyd should reduce the
    std by a meaningful margin.
    """
    import trimesh

    m = trimesh.creation.icosphere(subdivisions=3)  # ~642 vertices
    verts = m.vertices.astype(np.float32).copy()
    verts[:, 2] *= 0.2  # flatten — vertex density highly non-uniform
    faces = m.faces.astype(np.int32)
    n_v = verts.shape[0]
    cortex = np.arange(n_v)

    n_patches = 16
    a_no_lloyd = precompute_patches(
        mesh_lh=(verts, faces),
        mesh_rh=(verts, faces),
        cortex_indices_lh=cortex,
        cortex_indices_rh=cortex,
        n_patches=n_patches,
        seed=0,
        metric=metric,
        lloyd_iters=0,
    )
    a_lloyd = precompute_patches(
        mesh_lh=(verts, faces),
        mesh_rh=(verts, faces),
        cortex_indices_lh=cortex,
        cortex_indices_rh=cortex,
        n_patches=n_patches,
        seed=0,
        metric=metric,
        lloyd_iters=10,
    )
    std_no_lloyd = float(np.bincount(a_no_lloyd, minlength=n_patches).std())
    std_lloyd = float(np.bincount(a_lloyd, minlength=n_patches).std())
    assert std_lloyd < std_no_lloyd, (
        f"Lloyd should reduce patch-size std but did not "
        f"({metric}): no_lloyd={std_no_lloyd:.2f}, lloyd={std_lloyd:.2f}"
    )


def test_lloyd_iters_zero_is_pure_fps(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
) -> None:
    """lloyd_iters=0 should match the FPS-only behavior (deterministic for fixed seed)."""
    a = precompute_patches(
        synthetic_mesh_lh,
        synthetic_mesh_rh,
        np.arange(synthetic_mesh_lh[0].shape[0]),
        np.arange(synthetic_mesh_rh[0].shape[0]),
        n_patches=8,
        seed=0,
        lloyd_iters=0,
    )
    counts = np.bincount(a, minlength=8)
    assert int(counts.min()) > 0


def test_negative_lloyd_iters_raises(
    synthetic_mesh_lh: tuple[np.ndarray, np.ndarray],
    synthetic_mesh_rh: tuple[np.ndarray, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="lloyd_iters"):
        precompute_patches(
            synthetic_mesh_lh,
            synthetic_mesh_rh,
            np.arange(synthetic_mesh_lh[0].shape[0]),
            np.arange(synthetic_mesh_rh[0].shape[0]),
            n_patches=8,
            seed=0,
            lloyd_iters=-1,
        )


def test_disconnected_mesh_raises_for_geodesic_dijkstra() -> None:
    """Two separate icospheres with no shared vertices → geodesic FPS must raise."""
    import trimesh

    m1 = trimesh.creation.icosphere(subdivisions=1)
    m2 = trimesh.creation.icosphere(subdivisions=1)
    # Offset m2 so vertex coordinates don't overlap
    v2 = m2.vertices + np.array([10.0, 0.0, 0.0])
    n1 = m1.vertices.shape[0]
    n2 = m2.vertices.shape[0]
    # Combine into one multi-component mesh
    verts = np.concatenate([m1.vertices, v2], axis=0).astype(np.float32)
    faces = np.concatenate([m1.faces, m2.faces + n1], axis=0).astype(np.int32)
    cortex_indices = np.arange(n1 + n2)

    disconnected_mesh = (verts, faces)
    with pytest.raises(ValueError, match="connected component"):
        precompute_patches(
            mesh_lh=disconnected_mesh,
            mesh_rh=disconnected_mesh,
            cortex_indices_lh=cortex_indices,
            cortex_indices_rh=cortex_indices,
            n_patches=8,
            seed=0,
            metric="geodesic_dijkstra",
        )
