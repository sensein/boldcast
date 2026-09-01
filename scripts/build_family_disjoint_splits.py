#!/usr/bin/env python3
"""Build family-disjoint train/heldout splits from HCP Restricted family IDs.

Yibei-runs only. Claude never reads ``Restricted_*.csv`` (HCP DUA).

Reads the current 24-subject pool (``configs/subjects_train.txt`` +
``configs/subjects_heldout.txt``), looks up ``Family_ID`` for each
subject in ``Restricted_*.csv``, and reassigns subjects into
family-disjoint train (target 16) and heldout (target 8) sets via
greedy whole-family allocation under a deterministic shuffle (seed 0).

If the current pool's families yield < target_train + target_heldout,
the script optionally expands from an external pool of 7T-eligible
subject IDs (``--expanded-pool``).

Outputs:

* ``configs/subjects_train_familydisjoint.txt``
* ``configs/subjects_heldout_familydisjoint.txt``
* ``configs/family_disjoint_audit.json`` — **counts only**; never
  contains the restricted ``Family_ID`` values themselves.

Usage::

    export HCP_RESTRICTED_CSV=<path to the Restricted CSV, outside this repo>
    python scripts/build_family_disjoint_splits.py

    # or with explicit expanded-pool fallback:
    python scripts/build_family_disjoint_splits.py \\
        --restricted-csv "$HCP_RESTRICTED_CSV" \\
        --expanded-pool configs/hcp7t_eligible_pool.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path


def read_subject_list(path: Path) -> list[str]:
    """Read a one-subject-per-line text file, dropping comments and blanks."""
    out: list[str] = []
    with path.open() as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def load_family_map(csv_path: Path) -> dict[str, str]:
    """Return ``{subject_id: family_id}`` from a HCP-1200 Restricted CSV.

    Schema: a ``Subject`` column and a ``Family_ID`` column. Family IDs
    are themselves restricted under the WU-Minn HCP DUA — they live in
    this in-memory dict only and are never written to disk or stdout.
    """
    fam: dict[str, str] = {}
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "Subject" not in reader.fieldnames:
            raise SystemExit(
                f"ERROR: Restricted CSV missing 'Subject' column (saw fields: {reader.fieldnames})."
            )
        if "Family_ID" not in reader.fieldnames:
            raise SystemExit(
                "ERROR: Restricted CSV missing 'Family_ID' column "
                f"(saw fields: {reader.fieldnames})."
            )
        for row in reader:
            sub = str(row["Subject"]).strip()
            fid = str(row["Family_ID"]).strip()
            if sub and fid:
                fam[sub] = fid
    return fam


def assign_families(
    pool: list[str],
    fam_map: dict[str, str],
    target_train: int,
    target_heldout: int,
    seed: int,
) -> tuple[list[str], list[str], dict[str, object]]:
    """Greedy whole-family allocation under a seeded shuffle.

    Groups pool subjects by family ID, shuffles family IDs with
    ``random.Random(seed)``, then walks families in shuffled order:
    every member of a family goes to train until ``len(train) >=
    target_train``, then to heldout until ``len(heldout) >=
    target_heldout``. Remaining families are unused.

    Audit fields are COUNTS ONLY. Actual ``Family_ID`` values are
    never persisted (DUA).
    """
    family_to_subjects: dict[str, list[str]] = {}
    missing: list[str] = []
    for sub in pool:
        fid = fam_map.get(sub)
        if fid is None:
            missing.append(sub)
            continue
        family_to_subjects.setdefault(fid, []).append(sub)

    families = sorted(family_to_subjects.keys())
    rng = random.Random(seed)
    rng.shuffle(families)

    train: list[str] = []
    heldout: list[str] = []
    train_families: set[str] = set()
    heldout_families: set[str] = set()

    for fid in families:
        members = family_to_subjects[fid]
        if len(train) < target_train:
            train.extend(members)
            train_families.add(fid)
        elif len(heldout) < target_heldout:
            heldout.extend(members)
            heldout_families.add(fid)

    overlap = train_families & heldout_families
    unused = [fid for fid in families if fid not in train_families and fid not in heldout_families]

    audit: dict[str, object] = {
        "target_train": target_train,
        "target_heldout": target_heldout,
        "n_train": len(train),
        "n_heldout": len(heldout),
        "n_train_families": len(train_families),
        "n_heldout_families": len(heldout_families),
        "n_unused_families": len(unused),
        "family_overlap_count": len(overlap),
        "n_subjects_missing_from_csv": len(missing),
        "subjects_missing_from_csv": sorted(missing),
        "overflow_train": len(train) < target_train,
        "overflow_heldout": len(heldout) < target_heldout,
        "seed": seed,
    }
    return sorted(train), sorted(heldout), audit


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--restricted-csv",
        default=os.environ.get("HCP_RESTRICTED_CSV"),
        help="Path to HCP Restricted_*.csv (or set $HCP_RESTRICTED_CSV).",
    )
    p.add_argument(
        "--current-train",
        type=Path,
        default=Path("configs/subjects_train.txt"),
    )
    p.add_argument(
        "--current-heldout",
        type=Path,
        default=Path("configs/subjects_heldout.txt"),
    )
    p.add_argument(
        "--expanded-pool",
        type=Path,
        default=None,
        help=(
            "Optional path to a file listing the 175-subject 7T-eligible "
            "pool. Used only when the current 24-subject pool can't yield "
            "target_train + target_heldout family-disjoint subjects."
        ),
    )
    p.add_argument("--target-train", type=int, default=16)
    p.add_argument("--target-heldout", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out-train",
        type=Path,
        default=Path("configs/subjects_train_familydisjoint.txt"),
    )
    p.add_argument(
        "--out-heldout",
        type=Path,
        default=Path("configs/subjects_heldout_familydisjoint.txt"),
    )
    p.add_argument(
        "--audit-json",
        type=Path,
        default=Path("configs/family_disjoint_audit.json"),
    )
    args = p.parse_args()

    if not args.restricted_csv:
        print(
            "ERROR: --restricted-csv required (or set $HCP_RESTRICTED_CSV).",
            file=sys.stderr,
        )
        return 2

    restricted = Path(args.restricted_csv)
    if not restricted.is_file():
        print(f"ERROR: file not found: {restricted}", file=sys.stderr)
        return 2

    pool = read_subject_list(args.current_train) + read_subject_list(args.current_heldout)
    print(f"[splits] current pool: {len(pool)} subjects")

    fam_map = load_family_map(restricted)
    print(f"[splits] family map: {len(fam_map)} entries from {restricted.name}")

    train, heldout, audit = assign_families(
        pool,
        fam_map,
        target_train=args.target_train,
        target_heldout=args.target_heldout,
        seed=args.seed,
    )

    pool_expanded = False
    if (audit["overflow_train"] or audit["overflow_heldout"]) and args.expanded_pool is not None:
        if not args.expanded_pool.is_file():
            print(
                f"ERROR: --expanded-pool file not found: {args.expanded_pool}",
                file=sys.stderr,
            )
            return 2
        expanded = read_subject_list(args.expanded_pool)
        merged = sorted(set(pool) | set(expanded))
        print(
            f"[splits] overflow with original pool; expanding to "
            f"{len(merged)} subjects from {args.expanded_pool.name}"
        )
        train, heldout, audit = assign_families(
            merged,
            fam_map,
            target_train=args.target_train,
            target_heldout=args.target_heldout,
            seed=args.seed,
        )
        pool_expanded = True

    audit["pool_expanded"] = pool_expanded
    if pool_expanded and args.expanded_pool is not None:
        audit["pool_size_used"] = len(set(pool) | set(read_subject_list(args.expanded_pool)))
    else:
        audit["pool_size_used"] = len(pool)

    # Sanity: zero family overlap (this is THE property the K99 needs).
    if audit["family_overlap_count"] != 0:
        print(
            f"ERROR: family_overlap_count={audit['family_overlap_count']} "
            "after assignment — bug in assign_families.",
            file=sys.stderr,
        )
        return 3

    args.out_train.parent.mkdir(parents=True, exist_ok=True)
    args.out_train.write_text("\n".join(train) + "\n" if train else "", encoding="utf-8")
    args.out_heldout.write_text("\n".join(heldout) + "\n" if heldout else "", encoding="utf-8")
    args.audit_json.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        f"[splits] wrote {len(train)} train + {len(heldout)} heldout subjects "
        f"({audit['n_train_families']} + {audit['n_heldout_families']} families, "
        f"family_overlap=0)"
    )
    print(f"[splits]   train  -> {args.out_train}")
    print(f"[splits]   held   -> {args.out_heldout}")
    print(f"[splits]   audit  -> {args.audit_json}")

    if audit["overflow_train"] or audit["overflow_heldout"]:
        print(
            f"[splits] WARNING: targets {args.target_train}/{args.target_heldout}; "
            f"got {len(train)}/{len(heldout)}. "
            "Pass --expanded-pool with the 175-subject 7T-eligible list to expand.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
