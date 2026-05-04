"""Build deterministic train/heldout subject lists for the BOLDcast demo.

Two modes
---------
``--check-only``
    Scan the HCP root and print availability counts for both 3T and 7T REST
    modalities. Exits without writing any files. Use this first to decide
    which modality the demo should target.

(default)
    Generate ``configs/subjects_train.txt`` (16 subjects) and
    ``configs/subjects_heldout.txt`` (8 subjects) for the chosen modality
    via deterministic random sample (seed 0 by default), plus an audit
    JSON under ``$SCRATCH_DIR/boldcast/subject_list_audit.json``.

DUA / data-handling notes
-------------------------
This script lists the HCP data directory and reads filesystem **metadata**
only (file existence via ``Path.is_file``). It does not load any
neuroimaging file. Output contains 6-digit subject IDs (publicly
de-identified) and per-modality availability counts. The HCP S1200/1200
data is bound by the WU-Minn Data Use Agreement; do not paste raw
neuroimaging contents into chat or version control.

Usage
-----
    micromamba activate $BOLDCAST_ENV
    python scripts/build_subject_lists.py --check-only
    python scripts/build_subject_lists.py --modality 3T   # or 7T
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_HCP_ROOT = Path("$HCP_ROOT")
SUBJECT_RE = re.compile(r"^\d{6}$")

RUNS_3T: tuple[str, ...] = (
    "rfMRI_REST1_LR",
    "rfMRI_REST1_RL",
    "rfMRI_REST2_LR",
    "rfMRI_REST2_RL",
)
# Full 7T REST run set (8 runs, both PE directions per REST scan). Used for
# the --check-only diagnostic so we can see when additional phase-encodings
# get pulled from datalad later.
RUNS_7T: tuple[str, ...] = tuple(
    f"rfMRI_REST{i}_7T_{enc}" for i in (1, 2, 3, 4) for enc in ("PA", "AP")
)
# Subset actually pulled on the local mount as of 2026-05-03: alternating PE
# directions, one run per REST scan (REST1 PA, REST2 AP, REST3 PA, REST4 AP).
# This is what the 10-day demo trains on; switch to RUNS_7T once the other 4
# PEs are pulled.
RUNS_7T_DEMO: tuple[str, ...] = (
    "rfMRI_REST1_7T_PA",
    "rfMRI_REST2_7T_AP",
    "rfMRI_REST3_7T_PA",
    "rfMRI_REST4_7T_AP",
)
DTSERIES_SUFFIX = "_Atlas_MSMAll_hp2000_clean.dtseries.nii"


def has_all_runs(subj_dir: Path, runs: tuple[str, ...]) -> bool:
    """Return True iff the subject dir has every expected dtseries file."""
    return count_present_runs(subj_dir, runs) == len(runs)


def count_present_runs(subj_dir: Path, runs: tuple[str, ...]) -> int:
    """Return how many of the expected dtseries files are present."""
    results = subj_dir / "MNINonLinear" / "Results"
    return sum(
        (results / r / f"{r}{DTSERIES_SUFFIX}").is_file()
        for r in runs
    )


def list_subject_dirs(hcp_root: Path) -> list[Path]:
    """Enumerate 6-digit numeric subject subdirectories under ``hcp_root``."""
    if not hcp_root.is_dir():
        raise SystemExit(f"HCP root does not exist or is not a directory: {hcp_root}")
    return sorted(
        p for p in hcp_root.iterdir()
        if p.is_dir() and SUBJECT_RE.match(p.name)
    )


def filter_eligible(
    subjects: list[Path],
    runs: tuple[str, ...],
    max_workers: int = 32,
) -> list[Path]:
    """Parallelized check of which subjects have all required runs."""
    counts = run_counts(subjects, runs, max_workers=max_workers)
    return [s for s, c in zip(subjects, counts) if c == len(runs)]


def run_counts(
    subjects: list[Path],
    runs: tuple[str, ...],
    max_workers: int = 32,
) -> list[int]:
    """Per-subject count of present dtseries files among ``runs``."""
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(lambda s: count_present_runs(s, runs), subjects))


def histogram(counts: list[int], n_runs: int) -> str:
    """Return a one-line histogram: '0/N=12  1/N=304  ...  N/N=512'."""
    bins = [0] * (n_runs + 1)
    for c in counts:
        bins[c] += 1
    return "  ".join(f"{i}/{n_runs}={n}" for i, n in enumerate(bins))


def per_run_presence(
    subjects: list[Path],
    runs: tuple[str, ...],
    max_workers: int = 32,
) -> tuple[dict[str, int], list[tuple[bool, ...]]]:
    """For each run, count subjects that have it; also return per-subject flags."""
    def subject_flags(subj: Path) -> tuple[bool, ...]:
        return tuple(
            (subj / "MNINonLinear" / "Results" / r / f"{r}{DTSERIES_SUFFIX}").is_file()
            for r in runs
        )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        flags = list(pool.map(subject_flags, subjects))
    counts = {r: 0 for r in runs}
    for f in flags:
        for r, present in zip(runs, f):
            if present:
                counts[r] += 1
    return counts, flags


def write_list(path: Path, ids: list[str], label: str) -> None:
    """Write subject IDs to a configs/subjects_*.txt file with a header."""
    header = (
        f"# BOLDcast 10-day demo: {label} subjects from HCP S1200/1200.\n"
        f"# {len(ids)} subject IDs, one per line, no leading whitespace.\n"
        f"# Generated by scripts/build_subject_lists.py "
        f"(see configs/demo.yaml for the modality + seed used).\n"
        f"# Lines beginning with '#' are comments; blank lines ignored.\n"
        f"\n"
    )
    body = "\n".join(ids) + "\n"
    path.write_text(header + body)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--check-only", action="store_true",
                   help="Print availability counts and exit; do not write lists.")
    p.add_argument("--modality", choices=("3T", "7T"), default="7T",
                   help=("Which HCP REST modality to sample from. 3T uses 4 "
                         "REST runs (REST1/2 LR/RL, TR=0.72s); 7T uses the 4 "
                         "actually-pulled REST runs (REST1_PA, REST2_AP, "
                         "REST3_PA, REST4_AP, TR=1.0s). Default: 7T."))
    p.add_argument("--n-train", type=int, default=16)
    p.add_argument("--n-heldout", type=int, default=8)
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for deterministic random sample (default: 0).")
    p.add_argument("--hcp-root", type=Path, default=DEFAULT_HCP_ROOT,
                   help=f"HCP1200 root directory (default: {DEFAULT_HCP_ROOT}).")
    p.add_argument("--repo-root", type=Path,
                   default=Path(__file__).resolve().parent.parent,
                   help="Repo root; subjects_*.txt are written under repo-root/configs/.")
    p.add_argument("--max-workers", type=int, default=32,
                   help="Parallel stat() workers for the availability scan.")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Scanning {args.hcp_root} ...", file=sys.stderr)
    subjects = list_subject_dirs(args.hcp_root)
    print(f"  Total 6-digit subject dirs: {len(subjects)}", file=sys.stderr)

    print("Counting 3T REST availability ...", file=sys.stderr)
    counts_3t = run_counts(subjects, RUNS_3T, max_workers=args.max_workers)
    eligible_3t = [s for s, c in zip(subjects, counts_3t) if c == len(RUNS_3T)]
    print(f"  Distribution (n_runs_present / 4): {histogram(counts_3t, len(RUNS_3T))}",
          file=sys.stderr)
    print(f"  Subjects with all 4 3T REST dtseries: {len(eligible_3t)}",
          file=sys.stderr)

    print("Counting 7T REST availability ...", file=sys.stderr)
    per_run_7t, flags_7t = per_run_presence(subjects, RUNS_7T, max_workers=args.max_workers)
    counts_7t = [sum(f) for f in flags_7t]
    eligible_7t = [s for s, c in zip(subjects, counts_7t) if c == len(RUNS_7T)]
    print(f"  Distribution (n_runs_present / 8): {histogram(counts_7t, len(RUNS_7T))}",
          file=sys.stderr)
    print("  Per-run 7T REST availability:", file=sys.stderr)
    for r in RUNS_7T:
        print(f"    {r}: {per_run_7t[r]}", file=sys.stderr)
    print(f"  Subjects with all 8 7T REST dtseries: {len(eligible_7t)}",
          file=sys.stderr)

    if args.check_only:
        return 0

    if args.modality == "3T":
        runs = RUNS_3T
        eligible = eligible_3t
    else:
        runs = RUNS_7T_DEMO
        # Recompute eligibility against the 4 actually-pulled 7T runs (not 8).
        counts_demo = run_counts(subjects, runs, max_workers=args.max_workers)
        eligible = [s for s, c in zip(subjects, counts_demo) if c == len(runs)]
        print(f"Subjects with all {len(runs)} 7T-demo REST dtseries: "
              f"{len(eligible)}", file=sys.stderr)
    n_total = args.n_train + args.n_heldout
    if len(eligible) < n_total:
        print(
            f"ERROR: only {len(eligible)} eligible subjects for {args.modality}, "
            f"need {n_total} ({args.n_train} train + {args.n_heldout} heldout).",
            file=sys.stderr,
        )
        return 1

    rng = random.Random(args.seed)
    sample = rng.sample(eligible, n_total)
    train_ids = sorted(p.name for p in sample[: args.n_train])
    heldout_ids = sorted(p.name for p in sample[args.n_train :])

    train_path = args.repo_root / "configs" / "subjects_train.txt"
    heldout_path = args.repo_root / "configs" / "subjects_heldout.txt"
    write_list(train_path, train_ids, "training")
    write_list(heldout_path, heldout_ids, "held-out")

    audit = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "modality": args.modality,
        "runs": list(runs),
        "seed": args.seed,
        "n_eligible": len(eligible),
        "n_train": args.n_train,
        "n_heldout": args.n_heldout,
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "hcp_root": str(args.hcp_root),
        "family_disjoint": False,
        "family_disjoint_note": (
            "Demo splits are random from open-access subjects only; family IDs "
            "require WU-Minn HCP Restricted DUA. See docs/10_day_plan.md "
            "Subject Lists section for the documented caveat."
        ),
    }

    # SCRATCH_DIR (from .env) is already project-scoped (ends in /boldcast),
    # so we don't add another "boldcast" segment. Falls back to repo cache/
    # for users without SCRATCH_DIR set.
    scratch = Path(os.environ.get("SCRATCH_DIR", str(args.repo_root / "cache")))
    audit_path = scratch / "subject_list_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")

    print(f"\nWrote {train_path}", file=sys.stderr)
    print(f"Wrote {heldout_path}", file=sys.stderr)
    print(f"Audit:  {audit_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
