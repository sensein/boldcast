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


def test_from_config_train_split(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]], tmp_path: Path
) -> None:
    """``from_config`` resolves subjects, runs, patches, cortex indices from a
    config + writes a working Dataset. Uses a synthetic config pointing at the
    synthetic HCP layout; sidesteps Day-1's real-mesh FPS by pre-seeding the
    patch cache with a trivial assignment."""
    import yaml

    hcp_root, subjects, runs = synthetic_hcp_layout
    train_file = tmp_path / "subjects_train.txt"
    train_file.write_text("\n".join(subjects) + "\n")
    heldout_file = tmp_path / "subjects_heldout.txt"
    heldout_file.write_text("")  # empty heldout for the test

    n_v = 100  # synthetic dtseries has 100 cortex grayordinates
    n_p = 4
    patch_cache = tmp_path / "patches.npz"
    np.savez(
        str(patch_cache),
        assignment=_trivial_assignment(n_v=n_v, n_p=n_p),
        n_patches=np.asarray(n_p),
        seed=np.asarray(0),
        metric=np.asarray("euclidean3d"),
        lloyd_iters=np.asarray(0),
        n_lh_cortex=np.asarray(50),
        n_rh_cortex=np.asarray(50),
        # SHAs left out — from_config must NOT call build_or_load_patches
        # when the cache file already holds an `assignment` key.
    )

    cfg = {
        "data": {
            "dtseries_pattern": (
                str(hcp_root)
                + "/{subject}/MNINonLinear/Results/{run}/"
                + "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
            ),
            "subjects_train_file": str(train_file),
            "subjects_heldout_file": str(heldout_file),
            "runs": runs,
        },
        "tokenize": {
            "n_patches_cortex": n_p,
            "patch_cache": str(patch_cache),
            "cache_dir": str(tmp_path / "tokens_cache"),
            "standardize": "run_wise",
        },
        "window": {"size": 10, "stride": 5},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    ds = HCPRestingDataset.from_config(str(cfg_path), split="train")
    # 2 subjects × 2 runs × floor((20-10)/5)+1=3 windows = 12
    assert len(ds) == 12
    sample = ds[0]
    assert sample["tokens"].shape == (10, n_p)
    assert torch.isfinite(sample["tokens"]).all()


def test_from_config_heldout_split_uses_disjoint_subject_ids(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]], tmp_path: Path
) -> None:
    """Heldout split must offset subject_ids by len(train_subjects) so the
    model never sees the same int across splits. This is operationally
    load-bearing — Day-7 fingerprinting silently corrupts on ID collision."""
    import yaml

    hcp_root, subjects, runs = synthetic_hcp_layout
    # Put one synthetic subject in train, the other in heldout.
    train_file = tmp_path / "subjects_train.txt"
    train_file.write_text(subjects[0] + "\n")
    heldout_file = tmp_path / "subjects_heldout.txt"
    heldout_file.write_text(subjects[1] + "\n")

    n_p = 4
    patch_cache = tmp_path / "patches.npz"
    np.savez(
        str(patch_cache),
        assignment=_trivial_assignment(n_v=100, n_p=n_p),
        n_patches=np.asarray(n_p),
        seed=np.asarray(0),
        metric=np.asarray("euclidean3d"),
        lloyd_iters=np.asarray(0),
        n_lh_cortex=np.asarray(50),
        n_rh_cortex=np.asarray(50),
    )
    cfg = {
        "data": {
            "dtseries_pattern": (
                str(hcp_root)
                + "/{subject}/MNINonLinear/Results/{run}/"
                + "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
            ),
            "subjects_train_file": str(train_file),
            "subjects_heldout_file": str(heldout_file),
            "runs": runs,
        },
        "tokenize": {
            "n_patches_cortex": n_p,
            "patch_cache": str(patch_cache),
            "cache_dir": str(tmp_path / "tokens_cache"),
            "standardize": "run_wise",
        },
        "window": {"size": 10, "stride": 5},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    ds_train = HCPRestingDataset.from_config(str(cfg_path), split="train")
    ds_heldout = HCPRestingDataset.from_config(str(cfg_path), split="heldout")

    train_ids = {ds_train[i]["subject_id"] for i in range(len(ds_train))}
    heldout_ids = {ds_heldout[i]["subject_id"] for i in range(len(ds_heldout))}
    assert train_ids == {0}, train_ids  # 1 train subject at offset 0
    assert heldout_ids == {1}, heldout_ids  # 1 heldout subject at offset 1
    assert train_ids.isdisjoint(heldout_ids), (
        f"subject_id collision between train ({train_ids}) and heldout "
        f"({heldout_ids}) — would silently corrupt fingerprinting metrics"
    )


def test_from_config_raises_on_both_splits_empty(tmp_path: Path) -> None:
    """If both subject lists are empty, from_config must raise with a clear
    message rather than IndexError-ing in the cache-build branch."""
    import yaml

    train_file = tmp_path / "subjects_train.txt"
    train_file.write_text("")
    heldout_file = tmp_path / "subjects_heldout.txt"
    heldout_file.write_text("")
    cfg = {
        "data": {
            "dtseries_pattern": "/nonexistent/{subject}/{run}",
            "subjects_train_file": str(train_file),
            "subjects_heldout_file": str(heldout_file),
            "runs": ["fake_run"],
        },
        "tokenize": {
            "n_patches_cortex": 4,
            "patch_cache": str(tmp_path / "patches.npz"),
            "cache_dir": str(tmp_path / "tokens_cache"),
            "standardize": "run_wise",
        },
        "window": {"size": 10, "stride": 5},
    }
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="at least one split must contain"):
        HCPRestingDataset.from_config(str(cfg_path), split="train")


def test_iteration_order_is_deterministic_for_fixed_split(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]], tmp_path: Path
) -> None:
    """Two datasets built with the same args must yield identical
    (subject_id, run_id, window_start) sequences over their full iteration."""
    hcp_root, subjects, runs = synthetic_hcp_layout
    pattern = (
        str(hcp_root)
        + "/{subject}/MNINonLinear/Results/{run}/"
        + "{run}_Atlas_MSMAll_hp2000_clean.dtseries.nii"
    )

    def _build(cd: Path) -> HCPRestingDataset:
        return HCPRestingDataset(
            subjects=subjects, runs=runs, dtseries_pattern=pattern,
            cache_dir=cd,
            patch_assignment=_trivial_assignment(n_v=100, n_p=4),
            n_patches=4, window_size=10, stride=5,
        )

    ds_a = _build(tmp_path / "cache_a")
    ds_b = _build(tmp_path / "cache_b")
    keys_a = [(ds_a[i]["subject_id"], ds_a[i]["run_id"], ds_a[i]["window_start"])
              for i in range(len(ds_a))]
    keys_b = [(ds_b[i]["subject_id"], ds_b[i]["run_id"], ds_b[i]["window_start"])
              for i in range(len(ds_b))]
    assert keys_a == keys_b


def test_no_nan_in_any_window(
    synthetic_hcp_layout: tuple[Path, list[str], list[str]], tmp_path: Path
) -> None:
    """Standardization with eps + zero-fill on constant grayordinates must
    guarantee finite output across every window."""
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
    for i in range(len(ds)):
        sample = ds[i]
        assert torch.isfinite(sample["tokens"]).all(), (
            f"NaN/Inf at window {i}: subject_id={sample['subject_id']}, "
            f"run_id={sample['run_id']}, window_start={sample['window_start']}"
        )
