"""Tests for ``boldcast/_upstream/cifti_io.py``.

Synthetic-only — Claude does not load HCP data files (DUA).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh

from boldcast._upstream.cifti_io import (
    cortex_grayordinate_indices,
    load_dtseries,
    load_gifti_surface,
    save_dtseries,
)


def test_load_dtseries_returns_array_and_header(synthetic_dtseries: Path) -> None:
    data, header = load_dtseries(str(synthetic_dtseries))
    assert data.shape == (10, 100)
    assert data.dtype == np.float32
    assert isinstance(header, dict)
    assert header["n_grayordinates"] == 100
    assert header["n_tr"] == 10


def test_cortex_grayordinate_indices_partitions_lh_rh(synthetic_dtseries: Path) -> None:
    _, header = load_dtseries(str(synthetic_dtseries))
    lh_mesh_idx, rh_mesh_idx = cortex_grayordinate_indices(header)
    assert lh_mesh_idx.shape == (50,)
    assert rh_mesh_idx.shape == (50,)
    assert lh_mesh_idx.dtype.kind == "i"
    np.testing.assert_array_equal(lh_mesh_idx, np.arange(50))
    np.testing.assert_array_equal(rh_mesh_idx, np.arange(50))


def test_save_dtseries_roundtrip(
    synthetic_dtseries: Path, tmp_path: Path
) -> None:
    data_in, _ = load_dtseries(str(synthetic_dtseries))
    out = tmp_path / "roundtrip.dtseries.nii"
    save_dtseries(data_in, template=str(synthetic_dtseries), out=str(out))
    data_out, _ = load_dtseries(str(out))
    np.testing.assert_array_equal(data_in, data_out)


def test_load_gifti_surface_returns_verts_faces(tmp_path: Path) -> None:
    m = trimesh.creation.icosphere(subdivisions=2)
    gii = nib.gifti.GiftiImage()
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            m.vertices.astype(np.float32), intent="NIFTI_INTENT_POINTSET"
        )
    )
    gii.add_gifti_data_array(
        nib.gifti.GiftiDataArray(
            m.faces.astype(np.int32), intent="NIFTI_INTENT_TRIANGLE"
        )
    )
    out = tmp_path / "test.surf.gii"
    nib.save(gii, str(out))

    verts, faces = load_gifti_surface(str(out))
    assert verts.shape == (len(m.vertices), 3)
    assert faces.shape == (len(m.faces), 3)
    np.testing.assert_allclose(verts, m.vertices)
    np.testing.assert_array_equal(faces, m.faces)
