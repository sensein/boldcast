"""
Plot multi-GPU throughput scaling from benchmark results.

Generates a figure showing:
- Left: throughput (sequences/s) vs. number of GPUs with ideal linear scaling reference
- Right: parallel efficiency (actual / ideal) vs. number of GPUs

Usage:
    python benchmarks/plot_scaling.py --input "$SCRATCH/output/scaling_results.jsonl"
    python benchmarks/plot_scaling.py --input "$SCRATCH/output/scaling_results.jsonl" --output "$SCRATCH/output/scaling.pdf"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_results(path: str) -> list[dict]:
    """Load JSONL benchmark results."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    # Sort by GPU count
    results.sort(key=lambda r: r["n_gpus"])
    return results


def plot_scaling(results: list[dict], output: str | None = None, title_suffix: str = ""):
    """Generate scaling plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)

    n_gpus = [r["n_gpus"] for r in results]
    throughput = [r["throughput_sequences_per_sec"] for r in results]

    # Ideal linear scaling from single-GPU baseline
    base_throughput = throughput[0]
    ideal = [base_throughput * g / n_gpus[0] for g in n_gpus]
    efficiency = [t / i * 100 for t, i in zip(throughput, ideal)]

    # Config info for title
    r0 = results[0]
    config_str = (
        f"{r0['seq_len']} TRs × {r0['n_spatial_tokens']} tokens × "
        f"dim {r0.get('hidden_dim', '?')}, batch {r0['batch_size_per_gpu']}/GPU"
    )
    gpu_name = r0.get("gpu_name", "GPU")

    # --- Left: Throughput ---
    ax1.plot(n_gpus, ideal, "--", color="#999999", linewidth=1.5, label="Ideal linear", zorder=1)
    ax1.plot(n_gpus, throughput, "o-", color="#2563eb", linewidth=2, markersize=8,
             markerfacecolor="white", markeredgewidth=2, label="Measured", zorder=2)

    # Annotate throughput values
    for g, t in zip(n_gpus, throughput):
        ax1.annotate(f"{t:.1f}", (g, t), textcoords="offset points",
                     xytext=(0, 12), ha="center", fontsize=8.5, color="#2563eb")

    ax1.set_xlabel("Number of GPUs", fontsize=11)
    ax1.set_ylabel("Throughput (sequences / sec)", fontsize=11)
    ax1.set_title("Multi-GPU Throughput Scaling", fontsize=12, fontweight="bold")
    ax1.set_xticks(n_gpus)
    ax1.legend(fontsize=9, loc="upper left")
    ax1.set_xlim(0, max(n_gpus) + 0.5)
    ax1.set_ylim(0, max(max(ideal), max(throughput)) * 1.15)
    ax1.grid(True, alpha=0.3)

    # --- Right: Efficiency ---
    ax2.bar(n_gpus, efficiency, color="#2563eb", alpha=0.7, width=0.6, edgecolor="#1d4ed8")
    ax2.axhline(y=100, color="#999999", linestyle="--", linewidth=1, label="Ideal (100%)")

    for g, e in zip(n_gpus, efficiency):
        ax2.annotate(f"{e:.0f}%", (g, e), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=9, fontweight="bold")

    ax2.set_xlabel("Number of GPUs", fontsize=11)
    ax2.set_ylabel("Parallel Efficiency (%)", fontsize=11)
    ax2.set_title("Scaling Efficiency", fontsize=12, fontweight="bold")
    ax2.set_xticks(n_gpus)
    ax2.set_ylim(0, 115)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"BOLDcast — {gpu_name}\n{config_str}",
        fontsize=10, color="#555555", y=0.02
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot multi-GPU scaling results")
    parser.add_argument("--input", type=str, default="results/scaling_results.jsonl",
                        help="Path to JSONL benchmark results")
    parser.add_argument("--output", type=str, default=None,
                        help="Output figure path (e.g., figures/scaling.pdf). Shows plot if not set.")
    args = parser.parse_args()

    results = load_results(args.input)

    if not results:
        print(f"No results found in {args.input}")
        print("Run: bash benchmarks/run_scaling_sweep.sh")
        return

    print(f"Loaded {len(results)} benchmark results:")
    for r in results:
        print(f"  {r['n_gpus']} GPU(s): {r['throughput_sequences_per_sec']:.2f} seq/s, "
              f"peak {r['peak_memory_gb']:.1f} GB")

    plot_scaling(results, output=args.output)


if __name__ == "__main__":
    main()