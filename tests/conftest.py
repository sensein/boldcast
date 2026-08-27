"""Shared synthetic fixtures for ``boldcast/_upstream/`` tests.

Synthetic-only by design: Claude does not load HCP data files (DUA).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh
from nibabel.cifti2 import cifti2_axes


@pytest.fixture
def synthetic_dtseries(tmp_path: Path) -> Path:
    """Tiny in-memory dtseries: T=10 TRs × V=100 grayordinates (50 LH + 50 RH cortex).

    Each hemisphere's parent mesh is 60 vertices, with grayordinates 0..49
    drawn from mesh vertices 0..49 (10 vertices act as "medial wall" and
    are excluded from the dtseries).
    """
    n_tr, n_lh_grayordinates, n_rh_grayordinates = 10, 50, 50
    n_lh_vertices, n_rh_vertices = 60, 60

    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_tr, n_lh_grayordinates + n_rh_grayordinates), dtype=np.float32)

    bm_lh = cifti2_axes.BrainModelAxis.from_surface(
        vertices=np.arange(n_lh_grayordinates),
        nvertex=n_lh_vertices,
        name="CortexLeft",
    )
    bm_rh = cifti2_axes.BrainModelAxis.from_surface(
        vertices=np.arange(n_rh_grayordinates),
        nvertex=n_rh_vertices,
        name="CortexRight",
    )
    brain_axis = bm_lh + bm_rh
    series_axis = cifti2_axes.SeriesAxis(start=0.0, step=1.0, size=n_tr)

    header = nib.cifti2.Cifti2Header.from_axes((series_axis, brain_axis))
    img = nib.cifti2.Cifti2Image(data, header)
    out = tmp_path / "synthetic.dtseries.nii"
    nib.save(img, str(out))
    return out


@pytest.fixture
def synthetic_mesh_lh() -> tuple[np.ndarray, np.ndarray]:
    """Small icosphere as ``(vertices: (V, 3) float, faces: (F, 3) int)``."""
    m = trimesh.creation.icosphere(subdivisions=2)
    return m.vertices.astype(np.float32), m.faces.astype(np.int32)


@pytest.fixture
def synthetic_mesh_rh() -> tuple[np.ndarray, np.ndarray]:
    """Same shape as LH; translated so vertex coordinates are distinct."""
    m = trimesh.creation.icosphere(subdivisions=2)
    verts = m.vertices.astype(np.float32) + np.array([2.0, 0.0, 0.0], dtype=np.float32)
    return verts, m.faces.astype(np.int32)


@pytest.fixture
def synthetic_gifti_path_factory(
    tmp_path: Path,
) -> Callable[[tuple[np.ndarray, np.ndarray], str], Path]:
    """Factory: given a ``(verts, faces)`` mesh and a name, save a GIFTI surface file.

    Returns the saved file's ``Path``.  Built only from ``nibabel`` and
    ``numpy``, so the factory is portable for upstream test suites.
    """

    def _make(mesh: tuple[np.ndarray, np.ndarray], name: str) -> Path:
        verts, faces = mesh
        gii = nib.gifti.GiftiImage()
        gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(verts.astype(np.float32), intent="NIFTI_INTENT_POINTSET")
        )
        gii.add_gifti_data_array(
            nib.gifti.GiftiDataArray(faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE")
        )
        out = tmp_path / f"{name}.surf.gii"
        nib.save(gii, str(out))
        return out

    return _make


@pytest.fixture
def synthetic_hcp_layout(tmp_path: Path) -> tuple[Path, list[str], list[str]]:
    """Two synthetic subjects × two synthetic runs in HCP-like directory layout.

    Returns ``(hcp_root, subjects, runs)`` where ``hcp_root`` is the parent
    directory holding the standard HCP path
    ``{subject}/MNINonLinear/Results/{run}/{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii``,
    and ``subjects`` / ``runs`` are the string IDs to be plugged into a
    config-style ``dtseries_pattern``.

    Each synthetic dtseries has T=20 TRs and V=100 grayordinates (50 LH + 50 RH
    cortex), matching the schema of ``synthetic_dtseries`` so the Day-1 patcher
    can be reused unchanged.
    """
    hcp_root = tmp_path / "hcp_root"
    subjects = ["999001", "999002"]
    runs = ["rfMRI_FAKE1_PA", "rfMRI_FAKE2_AP"]
    n_tr, n_lh, n_rh = 20, 50, 50
    n_lh_verts, n_rh_verts = 60, 60

    rng = np.random.default_rng(0)
    bm_lh = cifti2_axes.BrainModelAxis.from_surface(
        vertices=np.arange(n_lh), nvertex=n_lh_verts, name="CortexLeft"
    )
    bm_rh = cifti2_axes.BrainModelAxis.from_surface(
        vertices=np.arange(n_rh), nvertex=n_rh_verts, name="CortexRight"
    )
    brain_axis = bm_lh + bm_rh
    series_axis = cifti2_axes.SeriesAxis(start=0.0, step=1.0, size=n_tr)
    header = nib.cifti2.Cifti2Header.from_axes((series_axis, brain_axis))

    for subject in subjects:
        for run in runs:
            run_dir = hcp_root / subject / "MNINonLinear" / "Results" / run
            run_dir.mkdir(parents=True, exist_ok=True)
            data = rng.standard_normal((n_tr, n_lh + n_rh), dtype=np.float32)
            img = nib.cifti2.Cifti2Image(data, header)
            nib.save(
                img,
                str(run_dir / f"{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"),
            )

    return hcp_root, subjects, runs


@pytest.fixture
def script_env(tmp_path: Path) -> dict[str, str]:
    """Environment for smoke tests that run a script in a subprocess.

    ``configs/demo.yaml`` interpolates ``${oc.env:HCP_ROOT}`` with no default
    (unlike ``SCRATCH_DIR`` and ``DATA``, which fall back to ``/tmp``), so the
    config cannot resolve unless HCP_ROOT is set. On a dev machine the repo
    ``.env`` supplies it, which made these tests pass locally while failing on
    any machine without one — CI caught exactly that.

    The scripts under test never read HCP data in ``--dry-run`` / no-CUDA
    mode, so an empty tmp_path is enough to let interpolation succeed. Keeping
    it a placeholder also means the test cannot accidentally touch real
    DUA-bound data.
    """
    return {**os.environ, "HCP_ROOT": str(tmp_path / "hcp_root_placeholder")}
