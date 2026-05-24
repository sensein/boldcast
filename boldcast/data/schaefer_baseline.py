"""Schaefer-400 parcellation loader for the Day-6 ROI baseline.

The Day-6 baseline reuses :class:`boldcast.data.hcp_rest.HCPRestingDataset`
unchanged — its ``patch_assignment`` argument is generic, so substituting
the Schaefer parcellation for the geodesic FPS assignment is sufficient
to swap tokenization without forking the dataset class.

This module provides:

* :func:`load_schaefer_cortex_assignment` — load a Schaefer fsLR_32k
  ``.dlabel.nii`` and return the 0-indexed per-cortex-grayordinate
  parcel ID, in the same LH+RH concatenated order that
  ``HCPRestingDataset`` expects.
* :func:`extract_cortex_labels_from_dlabel` — pure function used inside
  the loader; exposed so the index-mapping logic can be unit-tested
  with synthetic inputs (no real Schaefer file required).
* :func:`build_schaefer_dataset_from_config` — factory that mirrors
  ``HCPRestingDataset.from_config(split=...)`` but reads
  ``cfg.baseline.schaefer_dlabel`` and ``cfg.baseline.cache_dir``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from numpy.typing import NDArray

from boldcast.data.hcp_rest import HCPRestingDataset, _read_subject_list

__all__ = [
    "build_schaefer_dataset_from_config",
    "extract_cortex_labels_from_dlabel",
    "load_schaefer_cortex_assignment",
]


def extract_cortex_labels_from_dlabel(
    label_data: NDArray[np.integer[Any]],
    brain_models: list[dict[str, Any]],
    cortex_indices_lh: NDArray[np.integer[Any]],
    cortex_indices_rh: NDArray[np.integer[Any]],
    n_rois: int,
) -> NDArray[np.int64]:
    """Slice cortex labels from a CIFTI dlabel and remap to 0-indexed parcels.

    The CBIG Schaefer dlabel is published on the *full* fsLR_32k mesh
    (32492 vertices per hemisphere), with medial-wall vertices labeled
    0. HCP CIFTI grayordinates exclude the medial wall (29696 LH,
    29716 RH). To align the two, we index into the full-mesh hemisphere
    using the HCP grayordinate vertex indices from
    :func:`boldcast._upstream.cifti_io.cortex_grayordinate_indices`.

    Parameters
    ----------
    label_data : ndarray of shape ``(1, V_grayordinates)`` or ``(V_grayordinates,)``
        Integer label values per dlabel grayordinate.  Schaefer dlabels
        use 1-indexed parcel IDs (0 = background medial wall); we
        subtract 1 after slicing to HCP cortex vertices.
    brain_models : list of dict
        Brain-models list from the dlabel's CIFTI BrainModelAxis.  Each
        entry has ``name``, ``slice``, ``vertex``, ``nvertex``.  We use
        the ``CIFTI_STRUCTURE_CORTEX_LEFT``/``_RIGHT`` slices to pick
        out the per-hemisphere label arrays.
    cortex_indices_lh : ndarray of int
        HCP grayordinate vertex indices into the LH parent surface mesh,
        as returned by ``cortex_grayordinate_indices`` for an HCP CIFTI
        header.  Used to index *into* the LH dlabel hemisphere.
    cortex_indices_rh : ndarray of int
        Same for RH.
    n_rois : int
        Expected number of parcels (400 for Schaefer-400).  Every
        returned label must be in ``[0, n_rois)``.

    Returns
    -------
    assignment : ndarray of shape ``(V_cortex,)`` int64
        LH then RH concatenated, in the same order that
        ``boldcast._upstream.cifti_io.extract_cortex_grayordinates``
        produces.  Values in ``[0, n_rois)``.
    """
    flat = np.asarray(label_data).squeeze().astype(np.int64)
    if flat.ndim != 1:
        raise ValueError(
            f"label_data must squeeze to 1D; got shape {flat.shape}"
        )

    lh = next(bm for bm in brain_models if bm["name"] == "CIFTI_STRUCTURE_CORTEX_LEFT")
    rh = next(bm for bm in brain_models if bm["name"] == "CIFTI_STRUCTURE_CORTEX_RIGHT")
    lh_hemi = flat[lh["slice"]]
    rh_hemi = flat[rh["slice"]]
    lh_labels = lh_hemi[np.asarray(cortex_indices_lh, dtype=np.int64)]
    rh_labels = rh_hemi[np.asarray(cortex_indices_rh, dtype=np.int64)]
    cortex_labels = np.concatenate([lh_labels, rh_labels]).astype(np.int64)

    # Schaefer dlabels are 1-indexed: 0 is the "no-label" background
    # (medial wall on the full fsLR_32k mesh). HCP cortex indices should
    # skip those positions; if a 0 still leaks through, the indices are
    # wrong and we'd silently collapse the affected grayordinates into
    # a phantom "parcel 0" downstream. Fail loud.
    background = int((cortex_labels == 0).sum())
    if background:
        raise ValueError(
            f"{background} cortex grayordinates carry the background label "
            "(value 0) after indexing into the dlabel with HCP cortex "
            "vertices. Either the HCP indices include medial-wall positions, "
            "or the dlabel is not the fsLR_32k surface release from CBIG."
        )

    cortex_labels_0idx = cortex_labels - 1
    lo, hi = int(cortex_labels_0idx.min()), int(cortex_labels_0idx.max())
    if lo < 0 or hi >= n_rois:
        raise ValueError(
            f"cortex labels out of range [0, {n_rois}): got [{lo}, {hi}]"
        )
    return cortex_labels_0idx


def load_schaefer_cortex_assignment(
    dlabel_path: str | Path,
    cortex_indices_lh: NDArray[np.integer[Any]],
    cortex_indices_rh: NDArray[np.integer[Any]],
    n_rois: int = 400,
) -> NDArray[np.int64]:
    """Load a Schaefer fsLR_32k ``.dlabel.nii`` and return the cortex assignment.

    Thin wrapper: load CIFTI2 image, read its brain-model axis, index
    into each hemisphere using HCP cortex vertices, remap to 0-indexed
    parcels.  See :func:`extract_cortex_labels_from_dlabel` for the
    unit-testable extraction logic.

    Parameters
    ----------
    dlabel_path : str or Path
        Path to ``Schaefer2018_400Parcels_*_fslr32k.dlabel.nii``.
    cortex_indices_lh : ndarray of int
        HCP LH grayordinate vertex indices, from
        ``cortex_grayordinate_indices(hcp_header)``.
    cortex_indices_rh : ndarray of int
        Same for RH.
    n_rois : int, default 400
        Expected parcel count.

    Returns
    -------
    assignment : ndarray of shape ``(V_cortex,)`` int64
    """
    img = nib.load(str(dlabel_path))  # type: ignore[attr-defined,unused-ignore]
    data = np.asarray(img.get_fdata(), dtype=np.int64)  # type: ignore[attr-defined,unused-ignore]
    brain_axis = img.header.get_axis(1)  # type: ignore[attr-defined,unused-ignore]
    brain_models: list[dict[str, Any]] = []
    for name, slc, struct in brain_axis.iter_structures():
        if struct.nvertices:
            nvertex = int(next(iter(struct.nvertices.values())))
            vertex: NDArray[np.int64] | None = np.asarray(struct.vertex, dtype=np.int64)
        else:
            nvertex = 0
            vertex = None
        brain_models.append(
            {"name": name, "slice": slc, "vertex": vertex, "nvertex": nvertex}
        )
    return extract_cortex_labels_from_dlabel(
        data,
        brain_models,
        cortex_indices_lh=cortex_indices_lh,
        cortex_indices_rh=cortex_indices_rh,
        n_rois=n_rois,
    )


def build_schaefer_dataset_from_config(
    config_path: str,
    split: str,
) -> HCPRestingDataset:
    """Build an HCPRestingDataset that tokenizes via Schaefer-400 parcels.

    Mirrors ``HCPRestingDataset.from_config(split=...)`` but reads the
    parcellation from ``cfg.baseline.schaefer_dlabel`` and writes the
    tokenized per-(subject, run) cache under ``cfg.baseline.cache_dir``.

    Parameters
    ----------
    config_path : str
        Path to ``configs/demo.yaml``.
    split : ``"train"`` or ``"heldout"``
    """
    from omegaconf import OmegaConf

    if split not in ("train", "heldout"):
        raise ValueError(f"split must be 'train' or 'heldout', got {split!r}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)

    train_subjects = _read_subject_list(str(cfg.data.subjects_train_file))
    heldout_subjects = _read_subject_list(str(cfg.data.subjects_heldout_file))
    if split == "train":
        subjects = train_subjects
        offset = 0
    else:
        subjects = heldout_subjects
        offset = len(train_subjects)

    n_rois = int(cfg.baseline.n_rois)

    # The CBIG Schaefer dlabel is on the full fsLR_32k mesh (with medial
    # wall = label 0). To index it correctly we need the HCP grayordinate
    # vertex indices, which we read from the reference training subject's
    # CIFTI header.
    from boldcast._upstream.cifti_io import (
        cortex_grayordinate_indices,
        load_dtseries,
    )

    ref_subject = train_subjects[0]
    ref_run = cfg.data.runs[0]
    ref_path = str(cfg.data.dtseries_pattern).format(subject=ref_subject, run=ref_run)
    _, header = load_dtseries(ref_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)

    assignment = load_schaefer_cortex_assignment(
        str(cfg.baseline.schaefer_dlabel),
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        n_rois=n_rois,
    )

    return HCPRestingDataset(
        subjects=subjects,
        runs=list(cfg.data.runs),
        dtseries_pattern=str(cfg.data.dtseries_pattern),
        cache_dir=str(cfg.baseline.cache_dir),
        patch_assignment=assignment,
        n_patches=n_rois,
        window_size=int(cfg.window.size),
        stride=int(cfg.window.stride),
        subject_id_offset=offset,
        standardize_method=str(cfg.tokenize.standardize),
    )
