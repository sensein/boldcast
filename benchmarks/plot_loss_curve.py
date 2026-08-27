"""
Plot training loss curve from W&B logs or local CSV/JSON.

Generates a figure showing convergence of the preliminary baseline run
(e.g., ROI-based Mamba on HCP resting-state, single-step prediction).

Supports three input sources:
  1. Weights & Biases run (--wandb)
  2. Local CSV with columns: step, train_loss, [val_loss] (--csv)
  3. Local JSON lines with keys: step, train_loss, [val_loss] (--jsonl)

Usage:
    # From W&B
    python benchmarks/plot_loss_curve.py --wandb entity/project/run_id

    # From CSV
    python benchmarks/plot_loss_curve.py --csv results/loss_log.csv

    # From JSONL
    python benchmarks/plot_loss_curve.py --jsonl results/loss_log.jsonl

    # With output file
    python benchmarks/plot_loss_curve.py --csv results/loss_log.csv \
        --output "$SCRATCH/output/loss_curve.pdf"
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def load_from_csv(path: str) -> dict:
    """Load loss data from CSV (step, train_loss, val_loss)."""
    import csv

    data = {"step": [], "train_loss": [], "val_loss": []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data["step"].append(int(row["step"]))
            data["train_loss"].append(float(row["train_loss"]))
            if "val_loss" in row and row["val_loss"]:
                data["val_loss"].append(float(row["val_loss"]))
    return data


def load_from_jsonl(path: str) -> dict:
    """Load loss data from JSON lines."""
    data = {"step": [], "train_loss": [], "val_loss": []}
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            data["step"].append(row["step"])
            data["train_loss"].append(row["train_loss"])
            if "val_loss" in row:
                data["val_loss"].append(row["val_loss"])
    return data


def load_from_wandb(run_path: str) -> dict:
    """Load loss data from W&B run."""
    try:
        import wandb
    except ImportError:
        raise ImportError("Install wandb: pip install wandb")

    api = wandb.Api()
    run = api.run(run_path)

    # Try common metric names
    train_key = None
    val_key = None
    for key in ["train_loss", "train/loss", "loss", "training_loss"]:
        if key in run.summary:
            train_key = key
            break
    for key in ["val_loss", "val/loss", "validation_loss", "eval_loss"]:
        if key in run.summary:
            val_key = key
            break

    if train_key is None:
        available = list(run.summary.keys())
        raise ValueError(f"Could not find train loss key. Available: {available}")

    keys = ["_step", train_key]
    if val_key:
        keys.append(val_key)

    history = run.scan_history(keys=keys)

    data = {"step": [], "train_loss": [], "val_loss": []}
    for row in history:
        if train_key in row and row[train_key] is not None:
            data["step"].append(row.get("_step", len(data["step"])))
            data["train_loss"].append(row[train_key])
            if val_key and val_key in row and row[val_key] is not None:
                data["val_loss"].append(row[val_key])

    print(f"Loaded {len(data['step'])} steps from W&B run: {run_path}")
    print(f"  Train key: {train_key}, Val key: {val_key or 'not found'}")
    return data


def smooth(values: list[float], window: int = 10) -> np.ndarray:
    """Simple moving average smoothing."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    # Pad the beginning to keep same length
    pad = np.array(values[: len(values) - len(smoothed)])
    return np.concatenate([pad, smoothed])


def plot_loss_curve(
    data: dict,
    output: str | None = None,
    smooth_window: int = 10,
    title: str = "BOLDcast Preliminary Training — Baseline Convergence",
    subtitle: str = "",
) -> None:
    """Generate loss curve figure."""
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

    steps = np.array(data["step"])
    train_loss = np.array(data["train_loss"])
    train_smoothed = smooth(train_loss.tolist(), window=smooth_window)

    # Raw loss (light)
    ax.plot(steps, train_loss, color="#93c5fd", alpha=0.3, linewidth=0.5)
    # Smoothed loss
    ax.plot(steps, train_smoothed, color="#2563eb", linewidth=2, label="Train loss (smoothed)")

    if data["val_loss"]:
        val_loss = np.array(data["val_loss"])
        # Val loss might be logged less frequently — use matching steps
        if len(val_loss) == len(steps):
            val_steps = steps
        else:
            # Assume val is logged at regular intervals
            interval = max(1, len(steps) // len(val_loss))
            val_steps = steps[::interval][: len(val_loss)]

        val_smoothed = smooth(val_loss.tolist(), window=max(1, smooth_window // 3))
        ax.plot(val_steps, val_loss, color="#fca5a5", alpha=0.3, linewidth=0.5)
        ax.plot(val_steps, val_smoothed, color="#dc2626", linewidth=2, label="Val loss (smoothed)")

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Loss (MSE)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")

    if subtitle:
        ax.text(
            0.5,
            0.97,
            subtitle,
            transform=ax.transAxes,
            fontsize=9,
            color="#666666",
            ha="center",
            va="top",
        )

    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Log scale if loss spans > 1 order of magnitude
    if train_loss.max() / max(train_loss.min(), 1e-10) > 10:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    # Annotate final loss
    final_train = train_smoothed[-1]
    ax.annotate(
        f"Final: {final_train:.4f}",
        xy=(steps[-1], final_train),
        xytext=(-80, 20),
        textcoords="offset points",
        fontsize=9,
        color="#2563eb",
        arrowprops=dict(arrowstyle="->", color="#2563eb", lw=1),
    )

    fig.tight_layout()

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training loss curve")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--wandb", type=str, help="W&B run path: entity/project/run_id")
    source.add_argument("--csv", type=str, help="Path to CSV with step, train_loss, [val_loss]")
    source.add_argument("--jsonl", type=str, help="Path to JSONL with step, train_loss, [val_loss]")

    parser.add_argument(
        "--output", type=str, default=None, help="Output figure path (e.g., figures/loss_curve.pdf)"
    )
    parser.add_argument("--smooth", type=int, default=10, help="Smoothing window size")
    parser.add_argument(
        "--title", type=str, default="BOLDcast Preliminary Training — Baseline Convergence"
    )
    parser.add_argument(
        "--subtitle",
        type=str,
        default="",
        help="e.g., 'ROI-based Mamba, HCP resting-state, 1-step prediction'",
    )
    args = parser.parse_args()

    if args.wandb:
        data = load_from_wandb(args.wandb)
    elif args.csv:
        data = load_from_csv(args.csv)
    elif args.jsonl:
        data = load_from_jsonl(args.jsonl)

    print(
        f"Steps: {len(data['step'])}, "
        f"Train loss range: [{min(data['train_loss']):.4f}, {max(data['train_loss']):.4f}]"
    )
    if data["val_loss"]:
        print(
            f"Val loss points: {len(data['val_loss'])}, "
            f"range: [{min(data['val_loss']):.4f}, {max(data['val_loss']):.4f}]"
        )

    plot_loss_curve(
        data,
        output=args.output,
        smooth_window=args.smooth,
        title=args.title,
        subtitle=args.subtitle,
    )


if __name__ == "__main__":
    main()
