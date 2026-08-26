"""K99 prelim Figure 1A-D — manuscript-figure panels.

Four panels rendering the BOLDcast architecture overview + preliminary
results from the May-2026 10-day-sprint outputs. Each panel is saved as a
separate .pdf + .png file under figures/manuscript_fig1/ for hand-assembly
into the K99 grant figure.

No panel labels (A/B/C/D) — those are added during assembly.
No figure titles or super-titles.

Run with:
    .venv/bin/python notebooks/manuscript_fig1.py

Panel inventory:
- A: BOLDcast architecture overview — fMRI → Tokenizer → Mamba⊗kNN →
     {Forecast, Stimulus} heads with sMRI·FiLM and CLIP side-inputs.
     Adapted from talks/2026-05-14-seed-pitch/build_figures.py
     :fig_architecture() with the MIT-Red palette swapped for the K99
     neutral grant palette. (Schematic family.)
- B: 1,024 geodesic-patch sizes histogram + round-trip parity annotation
     (Distribution family).
- C: Day-5 training loss curve + trivial-baseline reference lines
     + 30.9% improvement annotation (Line family).
- D: Day-7 BOLDcast fingerprinting top-k accuracy with Clopper-Pearson CIs,
     mean_t vs mean_tp pool, chance line (Bar family).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_OUT_DIR = _REPO_ROOT / "figures" / "manuscript_fig1"


# ---------------------------------------------------------------------------
# Visual style — clean grant default
# ---------------------------------------------------------------------------

# Palette: neutral primary blue + grayscale accents; one warm accent for
# the model / "good" highlight. Colorblind-safe (deuteranopia, protanopia).
PRIMARY = "#1f4e79"        # deep blue — primary data
SECONDARY = "#7c8084"      # mid gray — secondary data, axes
ACCENT = "#c0392b"         # warm red — highlight (model / above-chance)
BASELINE_COLORS = {        # three muted hues for the three trivial baselines
    "zero": "#a8a8a8",
    "input": "#888888",
    "window_mean": "#5e5e5e",
}
POOL_COLORS = {            # two-pool palette for Panel F
    "mean_t": PRIMARY,
    "mean_tp": "#8aa9c9",  # lighter shade of PRIMARY for the secondary pool
}

# Apply style block (called once at top-level)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,  # editable text in vector PDFs
    "ps.fonttype": 42,
})


# ---------------------------------------------------------------------------
# Data loading (one-shot)
# ---------------------------------------------------------------------------

def load_panel_data() -> dict[str, object]:
    """Load all data needed for D, E, F into a single dict."""
    cache = np.load(_REPO_ROOT / "cache/patches_fsLR_32k_n1024_seed0_geo.npz")
    patch_assignment = cache["assignment"]

    with open(_REPO_ROOT / "results/day1_validate.json") as fh:
        day1 = json.load(fh)

    train_steps: list[int] = []
    train_loss: list[float] = []
    val_step: int | None = None
    val_loss: float | None = None
    with open(_REPO_ROOT / "results/day5_train/loss_log.jsonl") as fh:
        for line in fh:
            rec = json.loads(line)
            if "loss" in rec and "val_loss" not in rec:
                train_steps.append(rec["step"])
                train_loss.append(rec["loss"])
            elif "val_loss" in rec:
                val_step = rec["step"]
                val_loss = rec["val_loss"]

    with open(_REPO_ROOT / "results/day5_train/baseline_eval.json") as fh:
        baselines = json.load(fh)

    with open(_REPO_ROOT / "results/day7_fingerprint/boldcast_metrics.json") as fh:
        fp = json.load(fh)

    return {
        "patch_assignment": patch_assignment,
        "day1": day1,
        "train_steps": np.array(train_steps),
        "train_loss": np.array(train_loss),
        "val_step": val_step,
        "val_loss": val_loss,
        "baselines": baselines,
        "fingerprint": fp,
    }


# ---------------------------------------------------------------------------
# Panel A — architecture overview
# ---------------------------------------------------------------------------

def panel_a_architecture_overview() -> None:
    """T-shape architecture diagram for K99 Figure 1A.

    Adapted from talks/2026-05-14-seed-pitch/build_figures.py:fig_architecture
    with the MIT-Red palette swapped for the neutral grant palette
    (PRIMARY blue, SECONDARY gray, ACCENT warm red for side-inputs).
    Same legend: solid = shipped (Phase 1, Days 1–3); dashed = Phase 2
    (funded, infrastructure ready); thin/no-fill border = external/frozen
    side-input.
    """
    from matplotlib.patches import FancyBboxPatch

    shipped = PRIMARY              # blue replaces MIT Red
    phase2 = "#5a5a5a"             # dark gray for Phase-2 dashed
    accent = ACCENT                # warm red for side-inputs (sMRI, CLIP)
    bg_soft = "#f5f5f5"
    body = "#3a3a3a"
    ink = "#1a1a1a"

    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=160)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, w, h, title, sub=None, *, edge=shipped, fill=bg_soft,
              dashed=False, lw=1.4, title_size=9, sub_size=7):
        ls = (0, (4, 2)) if dashed else "-"
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            edgecolor=edge, facecolor=fill, linewidth=lw, linestyle=ls,
        )
        ax.add_patch(box)
        if sub:
            ax.text(x + w / 2, y + h - 0.36, title,
                    ha="center", va="center",
                    fontsize=title_size, color=edge, weight="bold")
            ax.text(x + w / 2, y + h - 0.92, sub,
                    ha="center", va="center",
                    fontsize=sub_size, color=body, style="italic")
        else:
            ax.text(x + w / 2, y + h / 2, title,
                    ha="center", va="center",
                    fontsize=title_size, color=edge, weight="bold")

    def shape_label(x_arrow, y_top, y_bot, text):
        ax.text(x_arrow - 0.25, (y_top + y_bot) / 2, text,
                ha="right", va="center", fontsize=7,
                color=body, family="monospace")

    def arrow(x1, y1, x2, y2, *, color=None, lw=1.4, style="-"):
        color = color if color is not None else shipped
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=0.4,head_width=0.24",
                        color=color, lw=lw, linestyle=style,
                        shrinkA=4, shrinkB=4))

    CX = 8.0
    bw_main = 5.0
    half = bw_main / 2

    Y_INPUT_TOP,  Y_INPUT_BOT  = 9.4, 8.5
    Y_TOK_TOP,    Y_TOK_BOT    = 7.5, 6.5
    Y_BB_TOP,     Y_BB_BOT     = 5.4, 3.9
    Y_HEAD_TOP,   Y_HEAD_BOT   = 2.5, 0.7

    # Main flow: fMRI → Tokenizer → Mamba⊗kNN → fork to two heads
    block(CX - half, Y_INPUT_BOT, bw_main, Y_INPUT_TOP - Y_INPUT_BOT,
          "fMRI", edge=shipped, lw=1.4, title_size=10)

    block(CX - half, Y_TOK_BOT, bw_main, Y_TOK_TOP - Y_TOK_BOT,
          "Tokenizer",
          "1,792 tokens / TR",
          title_size=10, sub_size=7)
    arrow(CX, Y_INPUT_BOT - 0.02, CX, Y_TOK_TOP + 0.02)
    shape_label(CX, Y_INPUT_BOT, Y_TOK_TOP, "(T, 91k)")

    bw_bb = 6.0
    half_bb = bw_bb / 2
    block(CX - half_bb, Y_BB_BOT, bw_bb, Y_BB_TOP - Y_BB_BOT,
          "Mamba ⊗ kNN",
          "× N layers · temporal O(T) · spatial k=8",
          title_size=11, sub_size=7.5)
    arrow(CX, Y_TOK_BOT - 0.02, CX, Y_BB_TOP + 0.02)
    shape_label(CX, Y_TOK_BOT, Y_BB_TOP, "(T, 1,792)")

    # Side input: sMRI → FiLM
    smri_x, smri_y, smri_w, smri_h = 12.6, 4.4, 2.6, 0.95
    block(smri_x, smri_y, smri_w, smri_h,
          "sMRI · FiLM", edge=accent, lw=1.1, title_size=8.5)
    arrow(smri_x, smri_y + smri_h / 2,
          CX + half_bb + 0.02, smri_y + smri_h / 2,
          color=accent, lw=1.1)

    # Two heads
    fc_x, fc_w = 1.6, 4.8
    st_x, st_w = 8.2, 4.4

    block(fc_x, Y_HEAD_BOT, fc_w, Y_HEAD_TOP - Y_HEAD_BOT,
          "Forecast",
          "Phase 1 · multi-step MSE",
          title_size=10, sub_size=7.5)

    block(st_x, Y_HEAD_BOT, st_w, Y_HEAD_TOP - Y_HEAD_BOT,
          "Stimulus",
          "Phase 2 · InfoNCE",
          edge=phase2, dashed=True,
          title_size=10, sub_size=7.5)

    fork_top = Y_BB_BOT - 0.02
    arrow(CX, fork_top, fc_x + fc_w / 2, Y_HEAD_TOP + 0.02)
    arrow(CX, fork_top, st_x + st_w / 2, Y_HEAD_TOP + 0.02,
          color=phase2, style=(0, (4, 2)))
    # Latent-shape label: pin it directly to the LEFT (forecast-side) arrow
    # so it doesn't collide with the dashed Stimulus arrow on the right.
    ax.text(CX - 0.4, Y_BB_BOT - 0.55, "(T, 1,792, d)",
            ha="right", va="center", fontsize=7,
            color=body, family="monospace")

    # Frozen CLIP → stimulus head
    clip_x, clip_w = 13.0, 2.6
    clip_h = 0.95
    clip_y = Y_HEAD_BOT + (Y_HEAD_TOP - Y_HEAD_BOT - clip_h) / 2
    block(clip_x, clip_y, clip_w, clip_h,
          "CLIP", "frozen ViT-L/14",
          edge=accent, lw=1.1, title_size=8.5, sub_size=6.5)
    arrow(clip_x, clip_y + clip_h / 2,
          st_x + st_w + 0.02, clip_y + clip_h / 2,
          color=accent, lw=1.1, style=(0, (3, 2)))

    # Legend (top-left): solid = shipped, dashed = Phase 2, thin = side-input
    lx, ly = 0.3, 9.55
    ax.add_patch(FancyBboxPatch(
        (lx, ly - 0.05), 0.45, 0.22,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        edgecolor=shipped, facecolor=bg_soft, linewidth=1.3,
    ))
    ax.text(lx + 0.6, ly + 0.06, "shipped (Phase 1)",
            ha="left", va="center", fontsize=7.5, color=ink)
    ax.add_patch(FancyBboxPatch(
        (lx, ly - 0.45), 0.45, 0.22,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        edgecolor=phase2, facecolor=bg_soft, linewidth=1.1, linestyle=(0, (4, 2)),
    ))
    ax.text(lx + 0.6, ly - 0.34, "Phase 2 (planned)",
            ha="left", va="center", fontsize=7.5, color=ink)
    ax.add_patch(FancyBboxPatch(
        (lx, ly - 0.85), 0.45, 0.22,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        edgecolor=accent, facecolor=bg_soft, linewidth=1.0,
    ))
    ax.text(lx + 0.6, ly - 0.74, "frozen / side-input",
            ha="left", va="center", fontsize=7.5, color=ink)

    out_base = _OUT_DIR / "fig1_a_architecture_overview"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.05,
                facecolor="white")
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight",
                pad_inches=0.05, facecolor="white")
    plt.close(fig)
    print(f"[panel A] -> {out_base}.{{pdf,png}}  "
          f"(architecture overview; neutral palette)")


# ---------------------------------------------------------------------------
# Panel B — patch-size distribution + round-trip parity
# ---------------------------------------------------------------------------

def panel_b_tokenization_parity(data: dict[str, object]) -> None:
    """Histogram of 1,024 geodesic-patch sizes with parity annotation."""
    assignment = data["patch_assignment"]
    patch_sizes = np.bincount(assignment, minlength=1024)

    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    counts, bins, _ = ax.hist(
        patch_sizes,
        bins=30,
        color=PRIMARY,
        edgecolor="white",
        linewidth=0.5,
        alpha=0.92,
    )
    ax.axvline(
        patch_sizes.mean(),
        color=ACCENT,
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        label=f"mean = {patch_sizes.mean():.0f}",
    )
    ax.set_xlabel("Grayordinates per patch")
    ax.set_ylabel("Number of patches")

    # Add headroom above the tallest bar for the annotation block.
    ax.set_ylim(0, counts.max() * 1.45)

    annotation = (
        f"n = {len(patch_sizes):,} patches\n"
        f"mean = {patch_sizes.mean():.0f}, "
        f"σ = {patch_sizes.std():.0f}\n"
        f"range = [{patch_sizes.min()}, {patch_sizes.max()}]\n"
        f"round-trip max |Δ| ≤ 4.7×10⁻¹¹ (fp64)"
    )
    ax.text(
        0.97, 0.97, annotation,
        transform=ax.transAxes,
        va="top", ha="right",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=3),
    )

    out_base = _OUT_DIR / "fig1_b_tokenization_parity"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[panel B] -> {out_base}.{{pdf,png}}  "
          f"(mean={patch_sizes.mean():.1f}, std={patch_sizes.std():.1f}, "
          f"range=[{patch_sizes.min()}, {patch_sizes.max()}])")


# ---------------------------------------------------------------------------
# Panel C — training loss + trivial baselines
# ---------------------------------------------------------------------------

def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple moving average for loss smoothing."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def panel_c_training_vs_baselines(data: dict[str, object]) -> None:
    """Training loss over 3,000 steps + 3 trivial-baseline reference lines."""
    steps = data["train_steps"]
    loss = data["train_loss"]
    val_loss = data["val_loss"]
    baselines = data["baselines"]

    smooth_window = 50
    smoothed_loss = _moving_average(loss, smooth_window)
    smoothed_steps = steps[smooth_window - 1:]

    fig, ax = plt.subplots(figsize=(5.2, 3.2))

    # Raw train loss (light)
    ax.plot(steps, loss, color=PRIMARY, alpha=0.18, linewidth=0.5, label=None)
    # Smoothed train loss (primary)
    ax.plot(smoothed_steps, smoothed_loss, color=PRIMARY, linewidth=1.4,
            label=f"Train loss (smoothed, w={smooth_window})")

    # Trivial baselines as horizontal reference lines.
    best_baseline_key = min(
        ("zero", "input", "window_mean"),
        key=lambda k: baselines[k],
    )
    best_baseline_val = baselines[best_baseline_key]
    pretty = {"zero": "predict-zero", "input": "predict-input",
              "window_mean": "predict-window-mean"}
    for key in ("zero", "input", "window_mean"):
        ax.axhline(
            baselines[key],
            color=BASELINE_COLORS[key],
            linestyle=":",
            linewidth=1.0,
            alpha=0.85,
            label=f"{pretty[key]} = {baselines[key]:.3f}",
        )

    # Final val_loss marker
    ax.plot(
        [data["val_step"]], [val_loss],
        marker="o", markersize=7,
        markerfacecolor=ACCENT, markeredgecolor="white", markeredgewidth=0.8,
        linestyle="none", zorder=5,
        label=f"Final val loss = {val_loss:.3f}",
    )

    ax.set_xlabel("Training step")
    ax.set_ylabel("MSE (standardized)")
    ax.set_xlim(-50, steps.max() + 50)
    # Trim Y axis lower bound to where the action is (~0.20-0.40 range).
    ax.set_ylim(0.19, 0.41)

    improvement_pct = (best_baseline_val - val_loss) / best_baseline_val * 100.0
    annotation = (
        f"+{improvement_pct:.1f}% over best trivial baseline\n"
        f"(model = {val_loss:.3f} vs. {pretty[best_baseline_key]} = "
        f"{best_baseline_val:.3f})"
    )
    ax.text(
        0.5, 1.02, annotation,
        transform=ax.transAxes,
        va="bottom", ha="center",
        fontsize=8,
    )

    # Legend below the x-axis, two columns, frameless.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=7.5,
        handlelength=1.8,
        labelspacing=0.3,
        ncol=2,
        columnspacing=1.5,
    )

    out_base = _OUT_DIR / "fig1_c_training_vs_baselines"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[panel C] -> {out_base}.{{pdf,png}}  "
          f"(val_loss={val_loss:.4f}, best_baseline={best_baseline_val:.4f} "
          f"({best_baseline_key}), improvement={improvement_pct:.1f}%)")


# ---------------------------------------------------------------------------
# Panel D — fingerprinting top-k with Clopper-Pearson CIs
# ---------------------------------------------------------------------------

def panel_d_fingerprinting_topk(data: dict[str, object]) -> None:
    """Grouped bar chart: top-1/5/10 for mean_t and mean_tp with CIs."""
    fp = data["fingerprint"]
    k_list = fp["k_list"]
    chance = 1.0 / fp["n_heldout_subjects"]
    n_runs = fp["results"]["mean_t"]["n_runs"]
    n_subjects = fp["n_heldout_subjects"]

    pool_order = ["mean_t", "mean_tp"]
    pool_labels = {
        "mean_t": "mean_t",
        "mean_tp": "mean_tp",
    }

    n_k = len(k_list)
    n_pool = len(pool_order)
    bar_width = 0.36
    x = np.arange(n_k)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))

    for j, pool in enumerate(pool_order):
        points = np.array(
            [fp["results"][pool]["ci"][str(k)]["point"] for k in k_list]
        )
        ci_lo = np.array(
            [fp["results"][pool]["ci"][str(k)]["ci_low"] for k in k_list]
        )
        ci_hi = np.array(
            [fp["results"][pool]["ci"][str(k)]["ci_high"] for k in k_list]
        )
        yerr = np.vstack([points - ci_lo, ci_hi - points])

        offset = (j - (n_pool - 1) / 2.0) * bar_width
        ax.bar(
            x + offset, points, bar_width,
            color=POOL_COLORS[pool],
            edgecolor="white",
            linewidth=0.6,
            label=pool_labels[pool],
            zorder=3,
        )
        ax.errorbar(
            x + offset, points, yerr=yerr,
            fmt="none",
            ecolor="#222222",
            elinewidth=0.9,
            capsize=2.5,
            capthick=0.9,
            zorder=4,
        )

    # Chance line — darker + thicker for K99-figure visibility.
    ax.axhline(
        chance,
        color="#3a3a3a",
        linestyle="--",
        linewidth=1.2,
        alpha=0.9,
        zorder=2,
    )
    # Chance annotation: place AT the line, right-aligned to the chart's
    # right edge so it sits over the rightmost (top-10) bars where the
    # chance-line gap is largest. Bbox ensures readability over the bars.
    ax.text(
        n_k - 1, chance,
        f"chance = 1/{n_subjects} = {chance:.3f}  ",
        ha="right", va="center",
        fontsize=8,
        color="#3a3a3a",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.95, pad=2),
        zorder=6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels([f"top-{k}" for k in k_list])
    ax.set_xlabel("Top-k retrieval")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(-0.5, n_k - 0.5)

    annotation = (
        f"n = {n_runs} probes from {n_subjects} held-out subjects; "
        f"95% Clopper-Pearson CIs"
    )
    ax.text(
        0.5, 1.02, annotation,
        transform=ax.transAxes,
        va="bottom", ha="center",
        fontsize=8,
    )

    # Legend below the x-axis (consistent with Panel E).
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        frameon=False,
        fontsize=8,
        handlelength=1.4,
        labelspacing=0.3,
        ncol=2,
        columnspacing=2.0,
        title="Embedding pool",
        title_fontsize=8,
    )

    out_base = _OUT_DIR / "fig1_d_fingerprinting_topk"
    fig.savefig(f"{out_base}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    top1_t = fp["results"]["mean_t"]["ci"]["1"]
    top1_tp = fp["results"]["mean_tp"]["ci"]["1"]
    print(f"[panel D] -> {out_base}.{{pdf,png}}  "
          f"top-1 mean_t = {top1_t['point']:.3f} "
          f"[{top1_t['ci_low']:.3f}, {top1_t['ci_high']:.3f}]; "
          f"top-1 mean_tp = {top1_tp['point']:.3f} "
          f"[{top1_tp['ci_low']:.3f}, {top1_tp['ci_high']:.3f}]")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {_OUT_DIR}")

    # Delete old D/E/F filenames if they exist (from prior runs before
    # the A overview panel was added and panels were renumbered).
    for stale in (
        "fig1_d_tokenization_parity",
        "fig1_e_training_vs_baselines",
        "fig1_f_fingerprinting_topk",
    ):
        for ext in (".pdf", ".png"):
            p = _OUT_DIR / f"{stale}{ext}"
            if p.exists():
                p.unlink()
                print(f"removed stale {p.name}")

    data = load_panel_data()
    print(f"data loaded: {len(data['train_steps'])} train steps, "
          f"{len(data['patch_assignment'])} cortex grayordinates")
    print("---")
    panel_a_architecture_overview()
    panel_b_tokenization_parity(data)
    panel_c_training_vs_baselines(data)
    panel_d_fingerprinting_topk(data)
    print("---")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
