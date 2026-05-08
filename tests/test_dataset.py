"""Tests for ``boldcast.data.hcp_rest.HCPRestingDataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from boldcast.data.hcp_rest import HCPRestingDataset


def _trivial_assignment(n_v: int = 100, n_p: int = 4) -> np.ndarray:
    """Round-robin assignment guaranteeing every patch has at least one vertex."""
    return np.arange(n_v, dtype=np.int32) % n_p


def test_init_and_len(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]], tmp_path: Path
) -> None:
    hcp_root, subjects, runs = synthetic_hcp_layout
    pattern = (
        str(hcp_root)
        + "/{subject}/MNINonLinear/Results/{run}/"
        + "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    )
    n_p = 4
    ds = HCPRestingDataset(
        subjects=subjects,
        runs=runs,
        dtseries_pattern=pattern,
        cache_dir=tmp_path / "cache",
        patch_assignment=_trivial_assignment(n_v=100, n_p=n_p),
        n_patches=n_p,
        window_size=10,
        stride=5,
        subject_id_offset=0,
    )
    # T=20, window_size=10, stride=5 ⇒ floor((20-10)/5) + 1 = 3 windows per run
    # 2 subjects × 2 runs × 3 windows = 12
    assert len(ds) == 12
