"""CIFTI dtseries and GIFTI surface I/O.

Self-contained module: no imports from any other ``boldcast`` submodule.
Targeted for upstream contribution to ``nobrainer.io`` (see
``boldcast/_upstream/README.md``).
"""

from __future__ import annotations

import nibabel as nib
import numpy as np

__all__ = [
    "cortex_grayordinate_indices",
    "load_dtseries",
    "load_gifti_surface",
    "save_dtseries",
]


def load_dtseries(path: str) -> tuple[np.ndarray, dict]:
    """Load a CIFTI dtseries file.

    Parameters
    ----------
    path : str
        Path to a ``*.dtseries.nii`` file.

    Returns
    -------
    data : ndarray of shape ``(T, V)``
        BOLD signal across ``T`` TRs and ``V`` grayordinates, ``float32``.
    header : dict
        Keys: ``"n_grayordinates"``, ``"n_tr"``, ``"tr_seconds"``, and
        ``"brain_models"`` (a list of per-structure dicts holding
        ``name``, ``slice``, ``vertex`` mesh-index array, and ``nvertex``
        of the parent mesh).
    """
    img = nib.load(path)
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    series_axis = img.header.get_axis(0)
    brain_axis = img.header.get_axis(1)

    brain_models: list[dict] = []
    for name, slc, struct in brain_axis.iter_structures():
        if struct.nvertices:
            nvertex = int(next(iter(struct.nvertices.values())))
            vertex = np.asarray(struct.vertex, dtype=np.int64)
        else:
            nvertex = 0
            vertex = None
        brain_models.append(
            {"name": name, "slice": slc, "vertex": vertex, "nvertex": nvertex}
        )

    header = {
        "n_grayordinates": int(brain_axis.size),
        "n_tr": int(series_axis.size),
        "tr_seconds": float(series_axis.step),
        "brain_models": brain_models,
    }
    return data, header


def save_dtseries(data: np.ndarray, template: str, out: str) -> None:
    """Save ``data`` as a CIFTI dtseries using the brain/series axes of ``template``.

    Parameters
    ----------
    data : ndarray of shape ``(T, V)``
        Must match the template's series and brain-model axes.
    template : str
        Path to a CIFTI file whose header to copy.
    out : str
        Output path (``*.dtseries.nii``).
    """
    template_img = nib.load(template)
    img = nib.cifti2.Cifti2Image(np.asarray(data, dtype=np.float32), template_img.header)
    nib.save(img, out)


def cortex_grayordinate_indices(header: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return mesh-vertex indices for LH and RH cortex grayordinates.

    Parameters
    ----------
    header : dict
        As returned by :func:`load_dtseries`.

    Returns
    -------
    lh_vertex : ndarray of int
        Mesh-vertex indices into the LH parent surface for each cortex-LH
        grayordinate.
    rh_vertex : ndarray of int
        Same for RH.
    """
    lh = next(
        bm
        for bm in header["brain_models"]
        if bm["name"] == "CIFTI_STRUCTURE_CORTEX_LEFT"
    )
    rh = next(
        bm
        for bm in header["brain_models"]
        if bm["name"] == "CIFTI_STRUCTURE_CORTEX_RIGHT"
    )
    return lh["vertex"], rh["vertex"]


def load_gifti_surface(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a GIFTI surface mesh.

    Parameters
    ----------
    path : str
        Path to a ``*.surf.gii`` file.

    Returns
    -------
    vertices : ndarray of shape ``(V, 3)`` float32
    faces : ndarray of shape ``(F, 3)`` int32
    """
    img = nib.load(path)
    pointset_intent = nib.nifti1.intent_codes.code["pointset"]
    triangle_intent = nib.nifti1.intent_codes.code["triangle"]
    verts = next(d for d in img.darrays if d.intent == pointset_intent).data
    faces = next(d for d in img.darrays if d.intent == triangle_intent).data
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)
