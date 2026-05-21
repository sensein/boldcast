"""Day-5 DDP scaling-efficiency benchmark (acceptance criterion #4).

Run twice:
  # Single-GPU baseline (T1):
  srun -p mit_normal_gpu --gres=gpu:h200:1 -t 00:30:00 \\
      python scripts/day5_bench_ddp_scaling.py \\
      --out-json results/bench_w1.json

  # Two-GPU DDP (T2):
  srun -p mit_normal_gpu --gres=gpu:h200:2 -t 00:30:00 \\
      torchrun --standalone --nproc-per-node=2 \\
      scripts/day5_bench_ddp_scaling.py \\
      --out-json results/bench_w2.json

Then compute efficiency = w2.tokens_per_second / (2 * w1.tokens_per_second).
Acceptance: >= 0.70.

NOTE: requires CUDA + mamba-ssm. Run on an ORCD GPU compute node under the
micromamba env. Claude does not execute this script (no GPU at login node; uv
env lacks mamba-ssm; HCP DUA is held by Yibei, not Claude).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# sys.path bootstrap — same as day4_overfit.py / day5_train_boldcast.py
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import torch  # noqa: E402
from boldcast.training import (  # noqa: E402
    cleanup_distributed,
    get_local_rank,
    get_world_size,
    init_distributed,
    is_distributed_run,
    is_rank_zero,
    seed_everything,
    setup_model_for_ddp,
)
from omegaconf import OmegaConf  # noqa: E402


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument(
        "--n-warmup",
        type=int,
        default=10,
        help="Warmup steps before timing (mamba JIT-compiles on first call).",
    )
    p.add_argument(
        "--n-timed",
        type=int,
        default=50,
        help="Timed forward+backward steps.",
    )
    p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write {world_size, tokens_per_second, ...} JSON.",
    )
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available():
        raise SystemExit("[bench] CUDA required — run on a GPU node.")

    # Detect DDP mode via env (set by torchrun)
    if is_distributed_run():
        init_distributed()
        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
        world_size = get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda")
        world_size = 1

    seed_everything(int(cfg.seed))

    # --- Deferred heavy imports (BOLDcastDemo needs mamba-ssm / CUDA build) ---
    import numpy as np  # noqa: E402
    from boldcast._upstream.cifti_io import (  # noqa: E402
        cortex_grayordinate_indices,
        load_dtseries,
    )
    from boldcast.models.boldcast_demo import BOLDcastDemo  # noqa: E402
    from boldcast.tokenize.geodesic import build_or_load_patches  # noqa: E402
    from boldcast.tokenize.knn import build_or_load_knn  # noqa: E402

    # Reference subject for patch + kNN cache build (same pattern as day4/day5).
    def _read_subject_list(path: str) -> list[str]:
        out: list[str] = []
        with open(path) as fh:
            for line in fh:
                s = line.strip()
                if s and not s.startswith("#"):
                    out.append(s)
        return out

    train_subjects = _read_subject_list(str(cfg.data.subjects_train_file))
    ref_subject = train_subjects[0]
    ref_run = cfg.data.runs[0]

    if is_rank_zero():
        print(f"[bench] reference subject={ref_subject}, run={ref_run}")

    ref_path = str(cfg.data.dtseries_pattern).format(
        subject=ref_subject, run=ref_run
    )
    _, header = load_dtseries(ref_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)

    surface_dir = str(cfg.data.surface_dir_template).format(subject=ref_subject)
    lh_mesh = (
        f"{surface_dir}/{ref_subject}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    )
    rh_mesh = (
        f"{surface_dir}/{ref_subject}.R.midthickness_MSMAll.32k_fs_LR.surf.gii"
    )

    if is_rank_zero():
        print(f"[bench] loading patch assignment from {cfg.tokenize.patch_cache}")
    assignment = build_or_load_patches(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(cfg.tokenize.patch_cache),
        n_patches=int(cfg.tokenize.n_patches_cortex),
    )

    if is_rank_zero():
        print(f"[bench] loading kNN from {cfg.tokenize.knn_cache}")
    adjacency_np = build_or_load_knn(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        patch_assignment=assignment,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k=int(cfg.tokenize.knn_k),
        cache_path=str(cfg.tokenize.knn_cache),
    )
    adjacency = torch.from_numpy(np.asarray(adjacency_np)).long().to(device)

    # Build a synthetic batch of the right shape — no HCP data needed.
    # Shape: (B, T, P, 1) matching the Patcher output convention.
    B = int(cfg.train.batch_size_per_gpu)
    T = int(cfg.window.size)
    P = int(cfg.tokenize.n_patches_cortex)
    tokens = torch.randn(B, T, P, 1, device=device)

    # Model with checkpointing on (matches Day-3 / Day-5 regime).
    horizons = tuple(int(h) for h in cfg.train.forecasting_horizons)
    if is_rank_zero():
        print(
            f"[bench] world_size={world_size}  B={B}  T={T}  P={P}  "
            f"horizons={horizons}  device={device}"
        )
    _demo_model = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=P,
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=adjacency,
        horizons=horizons,
        use_checkpoint=True,
    ).to(device)
    model: torch.nn.Module = setup_model_for_ddp(
        _demo_model, find_unused_parameters=False
    )

    # Loss target: random, same shape as model output (B, T_valid, P, H, d_in).
    H = len(horizons)
    T_valid = T - max(horizons)
    target = torch.randn(B, T_valid, P, H, 1, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup — lets Mamba JIT-compile selective-scan kernels before timing.
    if is_rank_zero():
        print(f"[bench] warming up ({args.n_warmup} steps) ...")
    for _ in range(args.n_warmup):
        optimizer.zero_grad(set_to_none=True)
        pred = model(tokens)[:, :T_valid]
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed measurement
    if is_rank_zero():
        print(f"[bench] timing ({args.n_timed} steps) ...")
    start = time.perf_counter()
    for _ in range(args.n_timed):
        optimizer.zero_grad(set_to_none=True)
        pred = model(tokens)[:, :T_valid]
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    # Per-rank tokens/sec
    tokens_processed = args.n_timed * B * T * P
    tps_per_rank = tokens_processed / elapsed

    # Aggregate across ranks in DDP mode (sum gives total system throughput).
    import torch.distributed as dist  # noqa: E402

    if is_distributed_run():
        tps_tensor = torch.tensor(tps_per_rank, dtype=torch.float64, device=device)
        dist.all_reduce(tps_tensor, op=dist.ReduceOp.SUM)
        tps_total = float(tps_tensor.item())
    else:
        tps_total = tps_per_rank

    if is_rank_zero():
        print(
            f"[bench] world_size={world_size}  n_timed={args.n_timed}  "
            f"elapsed={elapsed:.2f}s  tokens_per_second={tps_total:.2e}"
        )
        if args.out_json is not None:
            import json

            out = {
                "world_size": world_size,
                "n_warmup": args.n_warmup,
                "n_timed": args.n_timed,
                "elapsed_s": elapsed,
                "tokens_per_second": tps_total,
                "tokens_per_second_per_rank": tps_per_rank,
                "batch_size": B,
                "T": T,
                "P": P,
                "horizons": list(horizons),
            }
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(out, indent=2))
            print(f"[bench] results written to {args.out_json}")

    if is_distributed_run():
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
