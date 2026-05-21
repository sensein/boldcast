"""Day-6 Schaefer cache builder for the demo's HCP 7T REST runs.

NOTE: LOADS HCP data from $HCP_ROOT — DUA holder (Yibei) runs this,
not Claude. See CLAUDE.md "HCP Data Use Agreement".

For each (subject, run) in train + heldout splits, force a single
``Dataset.__getitem__`` call so the per-(subject, run) Schaefer-
parcellated tokenized cache is materialized under
``cfg.baseline.cache_dir``. Subsequent Day-6 training reads from
cache. Mirrors ``scripts/day2_build_cache.py`` for the BOLDcast
geodesic patches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import torch  # noqa: E402
from boldcast.data.schaefer_baseline import (  # noqa: E402
    build_schaefer_dataset_from_config,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/demo.yaml")
    args = p.parse_args()

    for split in ("train", "heldout"):
        ds = build_schaefer_dataset_from_config(args.config, split=split)
        print(f"[day6] {split}: len={len(ds)}; building Schaefer cache...")
        t0 = time.perf_counter()
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
        print(
            f"[day6] {split}: built {n_built} Schaefer cache files in {dt:.1f} s"
        )

    print("[day6] cache build done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
