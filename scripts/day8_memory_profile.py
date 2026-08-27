"""Day-8 GPU memory profile of BOLDcastDemo across sequence lengths.

Verifies the linear-in-T memory scaling that the proposal's Table 1
claims for the Mamba backbone.  For each sequence length T in a sweep,
measures peak GPU memory at forward-only and forward+backward, with
and without activation checkpointing.

This profiles the DEMO config (``P=1024, d_model=128, n_layers=4``)
— the model that was actually trained on Day 5.  Table-1 cells at
``d_model=256`` and the 26.5 M-param scaled config are the full-
project model and would require a wider param-count budget than
``BOLDcastDemo`` allows; that empirical sweep happens during R00 work.

The K99 prelim claim is the *linear-in-T scaling shape*, not the
specific absolute Table-1 numbers — those are reported as cross-
reference.

NOTE: requires CUDA + mamba-ssm.  Run on an ORCD GPU compute node:

    srun -p mit_normal_gpu --gres=gpu:h200:1 -t 00:30:00 \\
        python scripts/day8_memory_profile.py \\
        --out-json results/day8_memory.json \\
        --out-figure figures/day8_memory_scaling.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402


def _measure_config(
    t: int,
    p: int,
    d_model: int,
    n_layers: int,
    batch_size: int,
    k_neighbors: int,
    use_checkpoint: bool,
    device: torch.device,
    horizons: tuple[int, ...] = (1, 5),
) -> dict[str, float]:
    """Run forward and forward+backward at the requested shapes; return peak GB.

    Uses a random kNN adjacency (no actual cortical geometry) since
    memory peaks depend on shapes only, not on the specific neighbor
    structure.  BF16 autocast matches Day-5 training.
    """
    from boldcast.models.boldcast_demo import BOLDcastDemo

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)

    # Random kNN: first column = self-index, rest = random neighbors.
    adj = torch.zeros(p, k_neighbors, dtype=torch.long, device=device)
    adj[:, 0] = torch.arange(p, device=device)
    if k_neighbors > 1:
        adj[:, 1:] = torch.randint(0, p, (p, k_neighbors - 1), device=device)

    model = BOLDcastDemo(
        d_in=1,
        d_model=d_model,
        n_layers=n_layers,
        n_patches=p,
        k_neighbors=k_neighbors,
        adjacency=adj,
        horizons=horizons,
        use_checkpoint=use_checkpoint,
    ).to(device)

    x = torch.randn(batch_size, t, p, 1, device=device)

    # --- forward only ---
    torch.cuda.reset_peak_memory_stats(device=device)
    model.eval()
    with (
        torch.no_grad(),
        torch.amp.autocast(  # type: ignore[attr-defined,unused-ignore]
            device_type="cuda",
            dtype=torch.bfloat16,
        ),
    ):
        _ = model(x)
    torch.cuda.synchronize(device=device)
    fwd_gb = float(torch.cuda.max_memory_allocated(device=device)) / (1024**3)

    # --- forward + backward ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(  # type: ignore[attr-defined,unused-ignore]
        device_type="cuda",
        dtype=torch.bfloat16,
    ):
        out = model(x)
        # Match Day-5 loss shape contract: out is (B, T, P, H, d_in); pool to scalar.
        loss = out.float().pow(2).mean()
    loss.backward()
    opt.step()
    torch.cuda.synchronize(device=device)
    fwd_bwd_gb = float(torch.cuda.max_memory_allocated(device=device)) / (1024**3)

    n_params = sum(prm.numel() for prm in model.parameters())

    # Free for the next config.
    del model, x, opt
    torch.cuda.empty_cache()

    return {
        "T": t,
        "P": p,
        "d_model": d_model,
        "n_layers": n_layers,
        "batch_size": batch_size,
        "use_checkpoint": use_checkpoint,
        "n_params_M": n_params / 1e6,
        "fwd_gb": fwd_gb,
        "fwd_bwd_gb": fwd_bwd_gb,
    }


def _plot(results: list[dict[str, float]], out_path: Path) -> None:
    """Plot fwd+bwd memory vs T for use_checkpoint True / False."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for ckpt_flag, marker, label in [
        (False, "o", "no checkpointing"),
        (True, "s", "activation checkpointing"),
    ]:
        rows = [r for r in results if r["use_checkpoint"] is ckpt_flag]
        rows.sort(key=lambda r: r["T"])
        if not rows:
            continue
        ts = [r["T"] for r in rows]
        mem = [r["fwd_bwd_gb"] for r in rows]
        ax.plot(ts, mem, marker=marker, label=label)
    ax.set_xlabel("sequence length T (TRs)")
    ax.set_ylabel("peak GPU memory at fwd+bwd (GB)")
    ax.set_title("Day-8 memory scaling (BOLDcastDemo, P=1024, d_model=128, B=2)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--t-sweep",
        type=int,
        nargs="+",
        default=[128, 256, 384, 512],
        help="Sequence lengths (TRs) to sweep.",
    )
    p.add_argument("--n-patches", type=int, default=1024)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/day8_memory.json"),
    )
    p.add_argument(
        "--out-figure",
        type=Path,
        default=Path("figures/day8_memory_scaling.png"),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be measured without touching CUDA.",
    )
    args = p.parse_args()

    configs: list[dict[str, int | bool]] = []
    for t in args.t_sweep:
        for ckpt in (False, True):
            configs.append(
                {
                    "T": int(t),
                    "P": int(args.n_patches),
                    "d_model": int(args.d_model),
                    "n_layers": int(args.n_layers),
                    "batch_size": int(args.batch_size),
                    "k_neighbors": int(args.knn_k),
                    "use_checkpoint": bool(ckpt),
                }
            )

    if args.dry_run:
        print(f"[day8] dry-run: would measure {len(configs)} configs")
        for c in configs:
            print(f"  {c}")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("[day8] CUDA not available - run on a GPU node.")
    device = torch.device("cuda")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_figure.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float]] = []
    for c in configs:
        print(
            f"[day8] T={c['T']}, P={c['P']}, d_model={c['d_model']}, "
            f"layers={c['n_layers']}, ckpt={c['use_checkpoint']} ..."
        )
        row = _measure_config(
            t=int(c["T"]),
            p=int(c["P"]),
            d_model=int(c["d_model"]),
            n_layers=int(c["n_layers"]),
            batch_size=int(c["batch_size"]),
            k_neighbors=int(c["k_neighbors"]),
            use_checkpoint=bool(c["use_checkpoint"]),
            device=device,
        )
        print(
            f"[day8]   params={row['n_params_M']:.3f} M, "
            f"fwd={row['fwd_gb']:.2f} GB, fwd+bwd={row['fwd_bwd_gb']:.2f} GB"
        )
        results.append(row)

    args.out_json.write_text(json.dumps(results, indent=2))
    _plot(results, args.out_figure)
    print(f"[day8] wrote {args.out_json} and {args.out_figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
