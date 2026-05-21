"""Unit tests for the Schaefer-400 cortex-label extractor.

Tests the pure ``extract_cortex_labels_from_dlabel`` function with
synthetic CIFTI brain-model dicts — no real Schaefer dlabel required.
The CIFTI loader wrapper (``load_schaefer_cortex_assignment``) is
covered by Day-6 hardware validation when Yibei runs the script.
"""

from __future__ import annotations

import numpy as np
import pytest
from boldcast.data.schaefer_baseline import extract_cortex_labels_from_dlabel


def _make_brain_models(n_lh: int, n_rh: int, n_subcortex: int = 0) -> list[dict]:
    """Build a minimal brain-models list mimicking a CIFTI BrainModelAxis.

    Layout: ``[lh_cortex, rh_cortex, subcortex]`` (subcortex optional).
    Vertex arrays use ``np.arange`` for both hemispheres (the actual
    vertex indices don't matter for the label-extraction logic).
    """
    bm: list[dict] = [
        {
            "name": "CIFTI_STRUCTURE_CORTEX_LEFT",
            "slice": slice(0, n_lh),
            "vertex": np.arange(n_lh, dtype=np.int64),
            "nvertex": n_lh,
        },
        {
            "name": "CIFTI_STRUCTURE_CORTEX_RIGHT",
            "slice": slice(n_lh, n_lh + n_rh),
            "vertex": np.arange(n_rh, dtype=np.int64),
            "nvertex": n_rh,
        },
    ]
    if n_subcortex:
        bm.append(
            {
                "name": "CIFTI_STRUCTURE_BRAIN_STEM",
                "slice": slice(n_lh + n_rh, n_lh + n_rh + n_subcortex),
                "vertex": None,
                "nvertex": 0,
            }
        )
    return bm


def test_basic_extraction_concatenates_lh_then_rh() -> None:
    """Cortex labels come back as LH then RH, 1-indexed → 0-indexed."""
    # 4 LH vertices labeled 1,2,3,4; 4 RH vertices labeled 5,6,7,8; 2 subcortex.
    label_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 99, 100]], dtype=np.int64)
    bm = _make_brain_models(n_lh=4, n_rh=4, n_subcortex=2)
    out = extract_cortex_labels_from_dlabel(label_data, bm, n_rois=8)
    assert out.shape == (8,)
    assert out.dtype == np.int64
    # 1-indexed → 0-indexed: [1..8] - 1 = [0..7]
    assert out.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]


def test_subcortex_labels_are_ignored() -> None:
    """Labels in the subcortex slice never appear in the output."""
    # Set subcortex labels to a sentinel that would be out-of-range for
    # n_rois=8 if accidentally included.
    label_data = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 999, 999]], dtype=np.int64)
    bm = _make_brain_models(n_lh=4, n_rh=4, n_subcortex=2)
    out = extract_cortex_labels_from_dlabel(label_data, bm, n_rois=8)
    assert 999 - 1 not in out.tolist()


def test_squeeze_handles_1d_input() -> None:
    """Function accepts ``(V,)`` (no leading singleton) without erroring."""
    label_data = np.array([1, 2, 3, 4], dtype=np.int64)
    bm = _make_brain_models(n_lh=2, n_rh=2)
    out = extract_cortex_labels_from_dlabel(label_data, bm, n_rois=4)
    assert out.tolist() == [0, 1, 2, 3]


def test_background_label_on_cortex_fails_loud() -> None:
    """A 0 label on any cortex grayordinate raises ValueError."""
    # One LH vertex has the background label.
    label_data = np.array([[0, 2, 3, 4]], dtype=np.int64)
    bm = _make_brain_models(n_lh=2, n_rh=2)
    with pytest.raises(ValueError, match="background label"):
        extract_cortex_labels_from_dlabel(label_data, bm, n_rois=4)


def test_label_above_n_rois_fails_loud() -> None:
    """A label index >= n_rois raises ValueError."""
    label_data = np.array([[1, 2, 3, 5]], dtype=np.int64)
    bm = _make_brain_models(n_lh=2, n_rh=2)
    with pytest.raises(ValueError, match="out of range"):
        extract_cortex_labels_from_dlabel(label_data, bm, n_rois=4)


def test_non_2d_after_squeeze_fails() -> None:
    """If the input doesn't squeeze to 1D, fail rather than mis-slicing."""
    bad = np.zeros((2, 4), dtype=np.int64)
    bm = _make_brain_models(n_lh=2, n_rh=2)
    with pytest.raises(ValueError, match="squeeze to 1D"):
        extract_cortex_labels_from_dlabel(bad, bm, n_rois=4)
