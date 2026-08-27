"""Day-2 cache builder for the demo's HCP 7T REST runs.

NOTE: This script LOADS HCP data from $HCP_ROOT — it must be run by the DUA
holder (Yibei), never by Claude. See CLAUDE.md "HCP Data Use Agreement".

For each (subject, run) in the train + heldout splits, force a single
``Dataset.__getitem__`` call so the per-(subject, run) tokenized cache is
materialized on disk. Subsequent training reads from cache.

Acceptance (per ``docs/10_day_plan.md`` Day 2):
* Wall time < 30 minutes on 1 ORCD node for 24 subjects × 4 runs.
* No NaN in any cached tensor.
* Train cardinality ≈ 384 windows; heldout ≈ 192.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Bootstrap: make ``boldcast`` importable when this script is run as
# ``python scripts/foo.py`` from any worktree, regardless of whether the
# package is also installed editable into the active env. Inserts the repo
# root (parent of scripts/) at the head of sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Auto-load ``{repo}/.env`` so HCP_ROOT and friends survive shell hops
# (fresh compute-node sessions don't inherit the login-node env).
from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import torch
from boldcast.data.hcp_rest import HCPRestingDataset


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/demo.yaml")
    args = p.parse_args()

    for split in ("train", "heldout"):
        ds = HCPRestingDataset.from_config(args.config, split=split)
        print(f"[day2] {split}: len={len(ds)}; building cache...")
        t0 = time.perf_counter()
        # Walk one window per (subject, run) — cheaper than iterating all
        # windows (which would re-load the same cached file ~6×).
        seen: set[tuple[int, int]] = set()
        n_built = 0
        for i in range(len(ds)):
            sample = ds[i]
            key = (int(sample["subject_id"]), int(sample["run_id"]))
            if key in seen:
                continue
            seen.add(key)
            n_built += 1
            assert torch.isfinite(sample["tokens"]).all(), (
                f"NaN/Inf in built window: {key}, start={sample['window_start']}"
            )
        dt = time.perf_counter() - t0
        print(f"[day2] {split}: built {n_built} (subject, run) cache files in {dt:.1f} s")

    print("[day2] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
