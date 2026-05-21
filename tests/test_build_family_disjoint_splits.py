"""Unit tests for the greedy family-disjoint assignment.

Synthetic family maps only — no HCP data, no Restricted CSV.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module() -> object:
    """Import scripts/build_family_disjoint_splits.py as a module."""
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "build_family_disjoint_splits.py"
    )
    spec = importlib.util.spec_from_file_location("_bfds", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_bfds"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_module()
assign_families = _mod.assign_families  # type: ignore[attr-defined]


def test_whole_families_kept_together() -> None:
    """A family's members must all go to the same split."""
    pool = ["s1", "s2", "s3", "s4", "s5", "s6"]
    # families: F1=[s1,s2], F2=[s3,s4], F3=[s5,s6]
    fam_map = {"s1": "F1", "s2": "F1", "s3": "F2", "s4": "F2", "s5": "F3", "s6": "F3"}
    train, heldout, _ = assign_families(pool, fam_map, target_train=2, target_heldout=2, seed=0)
    # Each split contains exactly one family (2 members each)
    assert len(train) == 2
    assert len(heldout) == 2
    train_fams = {fam_map[s] for s in train}
    heldout_fams = {fam_map[s] for s in heldout}
    assert len(train_fams) == 1
    assert len(heldout_fams) == 1
    assert train_fams.isdisjoint(heldout_fams)


def test_family_overlap_is_zero() -> None:
    """No family ID appears in both splits, regardless of pool size."""
    pool = [f"s{i}" for i in range(20)]
    fam_map = {f"s{i}": f"F{i // 2}" for i in range(20)}  # 10 pairs
    train, heldout, audit = assign_families(
        pool, fam_map, target_train=10, target_heldout=8, seed=0,
    )
    assert audit["family_overlap_count"] == 0
    train_fams = {fam_map[s] for s in train}
    heldout_fams = {fam_map[s] for s in heldout}
    assert train_fams.isdisjoint(heldout_fams)


def test_train_target_satisfied_before_heldout() -> None:
    """Train fills to target_train first; heldout only starts after."""
    pool = [f"s{i}" for i in range(16)]
    fam_map = {f"s{i}": f"F{i}" for i in range(16)}  # 16 singleton families
    train, heldout, _ = assign_families(
        pool, fam_map, target_train=10, target_heldout=4, seed=0,
    )
    assert len(train) == 10
    assert len(heldout) == 4


def test_overflow_reported_when_pool_too_small() -> None:
    """Pool smaller than target_train + target_heldout sets overflow flags."""
    pool = ["s1", "s2", "s3"]
    fam_map = {"s1": "F1", "s2": "F2", "s3": "F3"}
    train, heldout, audit = assign_families(
        pool, fam_map, target_train=10, target_heldout=4, seed=0,
    )
    assert audit["overflow_train"] is True
    assert audit["overflow_heldout"] is True
    assert len(train) + len(heldout) <= 3


def test_seed_deterministic() -> None:
    """Same seed yields same assignment; different seeds may differ."""
    pool = [f"s{i}" for i in range(20)]
    fam_map = {f"s{i}": f"F{i // 2}" for i in range(20)}
    t1, h1, _ = assign_families(pool, fam_map, target_train=8, target_heldout=4, seed=0)
    t2, h2, _ = assign_families(pool, fam_map, target_train=8, target_heldout=4, seed=0)
    assert t1 == t2
    assert h1 == h2


def test_missing_subjects_reported() -> None:
    """Subjects absent from the family map land in audit['subjects_missing_from_csv']."""
    pool = ["s1", "s2", "s_missing"]
    fam_map = {"s1": "F1", "s2": "F2"}
    _, _, audit = assign_families(
        pool, fam_map, target_train=1, target_heldout=1, seed=0,
    )
    assert audit["n_subjects_missing_from_csv"] == 1
    assert audit["subjects_missing_from_csv"] == ["s_missing"]


def test_audit_never_exposes_family_ids() -> None:
    """The audit dict must contain counts only, never the restricted Family_ID values."""
    pool = [f"s{i}" for i in range(10)]
    fam_map = {f"s{i}": f"F_secret_{i}" for i in range(10)}
    _, _, audit = assign_families(
        pool, fam_map, target_train=4, target_heldout=4, seed=0,
    )
    # No key in the audit should leak family IDs.
    forbidden_substrings = ("F_secret_", "family_id", "Family_ID")
    audit_repr = repr(audit)
    for s in forbidden_substrings:
        assert s not in audit_repr, f"DUA violation: '{s}' leaked into audit"
