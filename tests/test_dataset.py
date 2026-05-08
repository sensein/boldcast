"""Tests for ``boldcast.data.hcp_rest.HCPRestingDataset``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
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


def test_getitem_returns_window_dict(
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
        subject_id_offset=7,  # nonzero — exercises the offset path
    )
    sample = ds[0]
    assert set(sample.keys()) == {"tokens", "subject_id", "run_id", "window_start"}
    assert isinstance(sample["tokens"], torch.Tensor)
    assert sample["tokens"].shape == (10, n_p)
    assert sample["tokens"].dtype == torch.float32
    assert torch.isfinite(sample["tokens"]).all()
    assert sample["subject_id"] == 7  # subject 0 + offset 7
    assert sample["run_id"] == 0
    assert sample["window_start"] == 0


def test_getitem_writes_cache_on_first_access(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]], tmp_path: Path
) -> None:
    hcp_root, subjects, runs = synthetic_hcp_layout
    pattern = (
        str(hcp_root)
        + "/{subject}/MNINonLinear/Results/{run}/"
        + "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    )
    cache_dir = tmp_path / "cache"
    ds = HCPRestingDataset(
        subjects=subjects, runs=runs, dtseries_pattern=pattern,
        cache_dir=cache_dir,
        patch_assignment=_trivial_assignment(n_v=100, n_p=4),
        n_patches=4, window_size=10, stride=5,
    )
    assert not cache_dir.exists() or not any(cache_dir.iterdir())
    _ = ds[0]
    cache_files = list(cache_dir.glob("*.npz"))
    assert len(cache_files) == 1
    assert cache_files[0].name == f"{subjects[0]}_{runs[0]}.npz"


def test_second_access_reads_from_cache_not_dtseries(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hcp_root, subjects, runs = synthetic_hcp_layout
    pattern = (
        str(hcp_root)
        + "/{subject}/MNINonLinear/Results/{run}/"
        + "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    )
    ds = HCPRestingDataset(
        subjects=subjects, runs=runs, dtseries_pattern=pattern,
        cache_dir=tmp_path / "cache",
        patch_assignment=_trivial_assignment(n_v=100, n_p=4),
        n_patches=4, window_size=10, stride=5,
    )
    _ = ds[0]  # first access populates cache
    # Now monkeypatch load_dtseries to raise — second access must hit cache.
    import boldcast.data.hcp_rest as m

    def _boom(*a: object, **k: object) -> None:
        raise AssertionError("second access should not reload the dtseries")

    monkeypatch.setattr(m, "load_dtseries", _boom)
    sample = ds[0]
    assert sample["tokens"].shape == (10, 4)


def test_cache_metadata_mismatch_raises(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]], tmp_path: Path
) -> None:
    hcp_root, subjects, runs = synthetic_hcp_layout
    pattern = (
        str(hcp_root)
        + "/{subject}/MNINonLinear/Results/{run}/"
        + "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    )
    cache_dir = tmp_path / "cache"
    ds1 = HCPRestingDataset(
        subjects=subjects, runs=runs, dtseries_pattern=pattern,
        cache_dir=cache_dir,
        patch_assignment=_trivial_assignment(n_v=100, n_p=4),
        n_patches=4, window_size=10, stride=5,
    )
    _ = ds1[0]  # populate cache with assignment-A
    # Build a fresh dataset with a *different* assignment; first access must
    # hit the cached file, see the assignment_sha mismatch, and raise.
    different = (np.arange(100, dtype=np.int32) % 4 + 1) % 4  # rotated by 1
    ds2 = HCPRestingDataset(
        subjects=subjects, runs=runs, dtseries_pattern=pattern,
        cache_dir=cache_dir,
        patch_assignment=different,
        n_patches=4, window_size=10, stride=5,
    )
    with pytest.raises(ValueError, match="cache metadata mismatch"):
        _ = ds2[0]
