"""CIFTI dtseries and GIFTI surface I/O.

Self-contained module: no imports from any other ``boldcast`` submodule.
Targeted for upstream contribution to ``nobrainer.io`` (see
``boldcast/_upstream/README.md``).

Note on type checking: nibabel ships sparse type stubs, so several
``# type: ignore[attr-defined]`` markers below cover symbols that exist
at runtime but are not declared in nibabel's stubs (``nib.load``,
``Cifti2Image``, ``get_fdata``, ``get_axis``, ``darrays``, etc.). These
are upstream stub gaps, not bugs in our code.
"""

from __future__ import annotations

from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

__all__ = [
    "cortex_grayordinate_indices",
    "extract_cortex_grayordinates",
    "load_dtseries",
    "load_gifti_surface",
    "save_dtseries",
]


def load_dtseries(path: str) -> tuple[NDArray[np.float32], dict[str, Any]]:
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
    img = nib.load(path)  # type: ignore[attr-defined]
    data = np.asarray(img.get_fdata(), dtype=np.float32)  # type: ignore[attr-defined]
    series_axis = img.header.get_axis(0)  # type: ignore[attr-defined]
    brain_axis = img.header.get_axis(1)  # type: ignore[attr-defined]

    brain_models: list[dict[str, Any]] = []
    for name, slc, struct in brain_axis.iter_structures():
        if struct.nvertices:
            nvertex = int(next(iter(struct.nvertices.values())))
            vertex: NDArray[np.int64] | None = np.asarray(
                struct.vertex, dtype=np.int64
            )
        else:
            nvertex = 0
            vertex = None
        brain_models.append(
            {"name": name, "slice": slc, "vertex": vertex, "nvertex": nvertex}
        )

    header: dict[str, Any] = {
        "n_grayordinates": int(brain_axis.size),
        "n_tr": int(series_axis.size),
        "tr_seconds": float(series_axis.step),
        "brain_models": brain_models,
    }
    return data, header


def save_dtseries(data: NDArray[np.floating[Any]], template: str, out: str) -> None:
    """Save ``data`` as a CIFTI dtseries using the brain/series axes of ``template``.

    Parameters
    ----------
    data : ndarray of shape ``(T, V)``
        Must match the template's series and brain-model axes. Raises
        ``ValueError`` if the shape does not match — nibabel will also
        catch this downstream, but we check early to suppress its
        ``UserWarning`` and produce a clearer error from this wrapper.
    template : str
        Path to a CIFTI file whose header to copy.
    out : str
        Output path (``*.dtseries.nii``).
    """
    template_img = nib.load(template)  # type: ignore[attr-defined]
    series_axis = template_img.header.get_axis(0)  # type: ignore[attr-defined]
    brain_axis = template_img.header.get_axis(1)  # type: ignore[attr-defined]
    expected = (int(series_axis.size), int(brain_axis.size))
    if data.shape != expected:
        raise ValueError(
            f"data shape {tuple(data.shape)} does not match template axes "
            f"{expected} (template={template!r})"
        )
    img = nib.cifti2.Cifti2Image(  # type: ignore[attr-defined,no-untyped-call]
        np.asarray(data, dtype=np.float32), template_img.header
    )
    nib.save(img, out)  # type: ignore[attr-defined]


def cortex_grayordinate_indices(
    header: dict[str, Any],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
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


def extract_cortex_grayordinates(
    data: NDArray[np.floating[Any]], header: dict[str, Any]
) -> NDArray[np.float32]:
    """Return the LH+RH cortex slice of a CIFTI dtseries data array.

    Parameters
    ----------
    data : ndarray of shape ``(T, V_total)``
        Full CIFTI dtseries (cortex + subcortex/cerebellum grayordinates).
    header : dict
        As returned by :func:`load_dtseries`.

    Returns
    -------
    cortex : ndarray of shape ``(T, V_cortex)`` float32
        Concatenation of the LH and RH cortex grayordinate columns. For
        the standard HCP grayordinate space, ``V_cortex = 59412``.
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
    return np.concatenate(
        [data[:, lh["slice"]], data[:, rh["slice"]]], axis=1
    ).astype(np.float32)


def load_gifti_surface(path: str) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
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
    img = nib.load(path)  # type: ignore[attr-defined]
    pointset_intent = nib.nifti1.intent_codes.code["pointset"]
    triangle_intent = nib.nifti1.intent_codes.code["triangle"]
    verts = next(d for d in img.darrays if d.intent == pointset_intent).data  # type: ignore[attr-defined]
    faces = next(d for d in img.darrays if d.intent == triangle_intent).data  # type: ignore[attr-defined]
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)
