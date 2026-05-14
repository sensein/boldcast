"""Build figures for the BOLDcast seed-grant pitch deck.

No HCP data is loaded. All inputs are constants from
- docs/proposal.md (Table 1)
- Day-3 PR / docs/orcd_benchmarks.md
- docs/methods.md
- back-of-envelope derivations explicitly noted below.

Outputs are written to talk/figures/*.png and talk/figures/*.gif. Run from
repo root with:

    $BOLDCAST_ENV/bin/python talk/build_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `_design` importable when running as `python talk/build_figures.py` from repo root
_TALK_DIR = Path(__file__).resolve().parent
if str(_TALK_DIR) not in sys.path:
    sys.path.insert(0, str(_TALK_DIR))

import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from _design.boldcast_palette import COLORS, apply_rcparams


def _canvas_to_pil(fig, dpi: int = 100) -> Image.Image:
    """Render a matplotlib figure's current canvas to a PIL.Image. Used by the
    manual GIF writer below (we bypass FuncAnimation+PillowWriter because that
    pipeline silently deduplicates consecutive identical frames, which kills the
    PDF/PNG screenshot fallback on slides with animated figures)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches=None)
    buf.seek(0)
    return Image.open(buf).copy()


def _save_gif(out_path, images, durations_ms):
    """Save a list of PIL Images as an animated GIF with per-frame durations.
    `durations_ms` may be an int (same duration for every frame) or a list."""
    images[0].save(
        str(out_path),
        save_all=True,
        append_images=images[1:],
        duration=durations_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Apply MIT palette rcParams (font, sizes, spines, dpi)
apply_rcparams("mit")

# --- Semantic color aliases -------------------------------------------------
# Deck-local override: MIT Red is the primary brand color for this deck. The
# slide CSS in talk/seed_pitch.md remaps --primary to MIT Red; figures match by
# pulling from COLORS["gap"] (also #750014) instead of COLORS["primary"].
C_OURS = COLORS["gap"]         # MIT Red #750014 — Mamba / BOLDcast / ours
C_MAMBA = C_OURS               # back-compat alias
C_OTHER = COLORS["body"]       # Dark Silver Gray #626a73 — transformer / comparator
C_TRANS = C_OTHER              # back-compat alias
C_ACCENT = COLORS["success"]   # MIT Dark Green #004d1a — BOLDcast demo point, forecast highlight
C_GREY = COLORS["accent"]      # Silver Gray #8b959e — muted captions, neutral bars
C_DATA = COLORS["primary"]     # MIT Dark Blue #002896 — secondary "data" role


# ---------------------------------------------------------------------------
# Scene-card filmstrip — replaces viridis tiles with a designed sequence of
# 12 muted film-tone cards. Each card is a two-band composition (sky over
# ground / light over shadow). Reads as a movie sequence rather than as a
# colormap noise pattern.
# ---------------------------------------------------------------------------
SCENE_PALETTES = [
    ("#f5d4a0", "#a06038"),  # dawn — warm pink-orange over rust
    ("#ffe0a0", "#cc8540"),  # morning — golden over ochre
    ("#a8c8e0", "#5878a0"),  # day sky — pale blue over steel
    ("#8aaa6a", "#3a5828"),  # forest — sage over deep green
    ("#5878a0", "#283848"),  # river dusk — steel over indigo
    ("#e8a880", "#7a5038"),  # golden hour — peach over umber
    ("#2a3855", "#0e1828"),  # night — dark blue gradient
    ("#dabea0", "#8a684a"),  # interior — cream over leather
    ("#d8a888", "#8a5840"),  # close-up — skin over shadow
    ("#785838", "#3a201a"),  # corridor — umber over near-black
    ("#8a9080", "#404840"),  # calm — sage gray over slate
    ("#a89898", "#403838"),  # final — warm gray over charcoal
]


def _draw_filmstrip(ax, T: int = 12) -> None:
    """Render T scene cards in `ax` at y ∈ [0.1, 0.9]. One card per TR."""
    palettes = SCENE_PALETTES[:T]
    for i, (top, bot) in enumerate(palettes):
        x = i + 0.05
        w = 0.9
        ax.add_patch(plt.Rectangle((x, 0.5), w, 0.4, facecolor=top, edgecolor="none"))
        ax.add_patch(plt.Rectangle((x, 0.1), w, 0.4, facecolor=bot, edgecolor="none"))
        ax.add_patch(plt.Rectangle((x, 0.1), w, 0.8, facecolor="none",
                                    edgecolor=COLORS["ink"], lw=0.6))


# ---------------------------------------------------------------------------
# Figure 1: memory vs sequence length, Mamba O(T) vs transformer O(T^2)
# ---------------------------------------------------------------------------
def fig_memory_scaling() -> None:
    """Activation-memory scaling curves.

    Anchors (no curve-fitting; ratio-derived, see speaker notes):

    * BOLDcast demo measured: T=256, P=1,024, d=128, F+B = 6.09 GB on H200
      (BF16 + activation checkpointing). Source: Day-3 PR #3.
    * Proposal Table 1 default: T=256, P=1,792, d=256, F+B = 6.9 GB w/ ckpt
      on H200. Source: docs/proposal.md Table 1 row "Default".
    * Transformer attention cost at same (T, P, d): O(T^2 * P) per layer
      for the temporal axis; we plot a relative growth curve normalised to
      the same operating point so the asymptote is visible. The exact
      coefficient is a sketch; the asymptotic shape is the load-bearing
      claim and is exact.
    """
    T_vals = np.array([32, 64, 96, 128, 192, 256, 384, 512, 768, 1024])

    # Mamba is linear in T at fixed (P, d). Anchor: ~6.09 GB @ T=256 on H200
    # demo config (P=1024, d=128). Scale linearly from there.
    mamba_gb = 6.09 * (T_vals / 256.0)

    # Transformer dense attention is O(T^2). Match Mamba at small T (T=32)
    # so the cross-over is visible; the asymptotic gap is the point.
    coef = (6.09 * 32.0 / 256.0) / (32.0**2)
    trans_gb = coef * T_vals**2

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    ax.plot(T_vals, mamba_gb, "-o", color=C_OURS, lw=3, ms=7, label="Mamba — O(T)")
    ax.plot(
        T_vals,
        trans_gb,
        "-s",
        color=C_OTHER,
        lw=3,
        ms=7,
        label="Transformer attention — O(T²)",
    )

    # Mark the BOLDcast demo operating point
    ax.scatter([256], [6.09], s=180, color=C_ACCENT, zorder=5, edgecolor="black")
    ax.annotate(
        "BOLDcast demo\n6.09 GB measured\nT=256, P=1,024",
        xy=(256, 6.09),
        xytext=(300, 50),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_ACCENT, lw=1.5),
    )

    # Naturalistic-fMRI working window band
    ax.axvspan(256, 1024, color=C_OURS, alpha=0.06, zorder=0)
    ax.text(
        500,
        500,
        "naturalistic-fMRI\nworking window",
        color=C_OURS,
        fontsize=10,
        ha="center",
        style="italic",
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["32", "64", "128", "256", "512", "1024"])
    ax.set_ylim(1, 1e4)
    ax.set_xlabel("Sequence length T (TRs)")
    ax.set_ylabel("Activation memory per sample (GB, log scale)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()

    fig.savefig(OUT / "mem_scaling.png")
    plt.close(fig)
    print(f"wrote {OUT/'mem_scaling.png'}")


# ---------------------------------------------------------------------------
# Figure 2: Day-3 shipped metrics
# ---------------------------------------------------------------------------
def fig_day3_metrics() -> None:
    """Three Day-3 numbers as a clean visual.

    Source: Day-3 PR #3 / docs/orcd_benchmarks.md.
    """
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))

    metrics = [
        dict(
            ax=axes[0],
            value=0.733,
            target_lo=0.5,
            target_hi=1.5,
            unit="M",
            label="Parameters",
            ymax=2.0,
            note="asserted 0.5–1.5 M",
        ),
        dict(
            ax=axes[1],
            value=139,
            target_lo=None,
            target_hi=200,
            unit="ms",
            label="Forward pass",
            ymax=260,
            note="B=2, T=256, P=1,024\nBF16 autocast",
        ),
        dict(
            ax=axes[2],
            value=6.09,
            target_lo=None,
            target_hi=8.0,
            unit="GB",
            label="F+B memory",
            ymax=10.5,
            note="BF16 + activation\ncheckpointing",
        ),
    ]

    for m in metrics:
        ax = m["ax"]
        ax.bar([0], [m["value"]], color=C_OURS, width=0.6, edgecolor="black")
        # Target lines anchored at the right edge so their text does not sit on
        # the dashed line. The annotation reads off to the right margin.
        if m["target_hi"] is not None:
            ax.axhline(m["target_hi"], color=C_OTHER, ls="--", lw=1.5, alpha=0.7)
            ax.text(
                1.45,
                m["target_hi"],
                f"≤ {m['target_hi']:g} {m['unit']}",
                color=C_OTHER,
                fontsize=10,
                ha="right",
                va="bottom",
            )
        if m["target_lo"] is not None:
            ax.axhline(m["target_lo"], color=C_OTHER, ls="--", lw=1.5, alpha=0.7)
            ax.text(
                1.45,
                m["target_lo"],
                f"≥ {m['target_lo']:g} {m['unit']}",
                color=C_OTHER,
                fontsize=10,
                ha="right",
                va="top",
            )
        # Value label sits well above the bar to avoid colliding with target lines.
        ax.text(
            0,
            m["value"] + m["ymax"] * 0.04,
            f"{m['value']:g} {m['unit']}",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            0,
            -m["ymax"] * 0.08,
            m["note"],
            ha="center",
            va="top",
            fontsize=9,
            color=C_GREY,
        )
        ax.set_ylim(0, m["ymax"])
        ax.set_xlim(-0.6, 1.5)
        ax.set_xticks([])
        ax.set_title(m["label"])
        ax.spines["bottom"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "day3_metrics.png")
    plt.close(fig)
    print(f"wrote {OUT/'day3_metrics.png'}")


# ---------------------------------------------------------------------------
# Figure 3: hook schematic — movie timeline + brain-state trajectory
# ---------------------------------------------------------------------------
def fig_hook() -> None:
    """Slide-2 hook (static PDF fallback): scene-card filmstrip above, brain-
    state trajectory below, forecast arrow pointing into the future.

    Pure schematic — no real data.
    """
    d = _hook_data()
    T = d["T"]
    t_fine = d["t_fine"]
    y = d["y"]
    obs_end = d["obs_end"]

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 3.6), gridspec_kw=dict(height_ratios=[1, 2], hspace=0.05)
    )
    ax_top, ax_bot = axes

    _draw_filmstrip(ax_top, T)
    ax_top.set_xlim(0, T + 2)
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.text(-0.3, 0.5, "stimulus", ha="right", va="center", fontsize=11, color=C_GREY)

    ax_bot.plot(t_fine + 0.5, y, color=C_OURS, lw=2.5)
    ax_bot.axvspan(0.5, obs_end + 0.5, color=C_OURS, alpha=0.08)
    ax_bot.text(
        (0.5 + obs_end + 0.5) / 2,
        y.max() * 1.05,
        "observed past 30 s",
        ha="center", fontsize=10, color=C_OURS, style="italic",
    )
    ax_bot.axvspan(obs_end + 0.5, T + 0.5, color=C_ACCENT, alpha=0.18)
    ax_bot.annotate(
        "forecast: 5 TRs ahead",
        xy=(T + 0.3, y[-15]),
        xytext=(obs_end + 0.5, y.max() * 1.05),
        ha="left", fontsize=10, color=C_ACCENT,
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.8),
    )
    ax_bot.set_xlim(0, T + 2)
    ax_bot.set_yticks([])
    ax_bot.set_xticks([])
    ax_bot.text(-0.3, np.mean(ax_bot.get_ylim()), "brain", ha="right", va="center", fontsize=11, color=C_GREY)
    ax_bot.set_xlabel("time (TRs) →", labelpad=2, color=C_GREY)

    fig.savefig(OUT / "hook.png")
    plt.close(fig)
    print(f"wrote {OUT/'hook.png'}")


# ---------------------------------------------------------------------------
# Figure 4: tokenization compute wall (slide 4a)
# ---------------------------------------------------------------------------
def fig_tokenization_wall() -> None:
    """Bar chart: 262k voxels → 91k grayordinates → 400 atlas → 1,792 patches.

    Visual story: three muted silver bars (voxels / grayordinates / atlas) and
    one MIT-Red bar (BOLDcast patches) so the audience reads ownership.
    """
    fig, ax = plt.subplots(figsize=(13, 5.2))

    labels = [
        "Voxels\n(volumetric)",
        "Grayordinates\n(CIFTI)",
        "Atlas parcels\n(Schaefer-400)",
        "BOLDcast patches\n(ours)",
    ]
    counts = [262144, 91282, 400, 1792]
    n = len(counts)
    bar_colors = [C_GREY, C_GREY, C_GREY, C_OURS]

    bars = ax.bar(range(n), counts, color=bar_colors, edgecolor="black", width=0.65)
    ax.set_yscale("log")
    ax.set_ylabel("tokens per TR (log)", fontsize=15)
    ax.set_ylim(100, 1e7)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=17, rotation=0)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_xlim(-0.6, n - 0.4)

    for i, (bar, c) in enumerate(zip(bars, counts)):
        is_ours = i == n - 1
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            c * 1.7,
            f"{c:,}",
            ha="center",
            va="bottom",
            fontsize=18,
            fontweight="bold",
            color=C_OURS if is_ours else COLORS["ink"],
        )
        if is_ours:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                c * 6.0,
                "✓",
                ha="center",
                va="bottom",
                fontsize=26,
                color=C_ACCENT,
                fontweight="bold",
            )

    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    fig.savefig(OUT / "tokenization_wall.png")
    plt.close(fig)
    print(f"wrote {OUT/'tokenization_wall.png'}")


# ---------------------------------------------------------------------------
# Figure 5: stimulus alignment gap (slide 2b)
# ---------------------------------------------------------------------------
def fig_stimulus_gap() -> None:
    """Two-panel schematic: static image→snapshot vs continuous stim↔brain."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))

    # LEFT: static image → fMRI snapshot (MindEye/BrainCLIP framing)
    ax = axes[0]
    ax.add_patch(plt.Rectangle((0.05, 0.6), 0.3, 0.3, fc=C_GREY, ec="black"))
    ax.text(0.2, 0.75, "image\nframe", ha="center", va="center", color="white", fontsize=11)
    ax.add_patch(plt.Circle((0.75, 0.75), 0.13, fc=C_MAMBA, ec="black"))
    ax.text(0.75, 0.75, "fMRI\nsnapshot", ha="center", va="center", color="white", fontsize=10)
    ax.annotate(
        "",
        xy=(0.62, 0.75),
        xytext=(0.36, 0.75),
        arrowprops=dict(arrowstyle="->", lw=2, color="black"),
    )
    ax.text(0.5, 0.4, "MindEye / BrainCLIP", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.25, "single frame ↔ single snapshot", ha="center", fontsize=10, color=C_GREY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # RIGHT: continuous stimulus ↔ continuous brain (what's missing)
    ax = axes[1]
    T = 8
    for i in range(T):
        ax.add_patch(
            plt.Rectangle(
                (0.05 + i * 0.09, 0.7),
                0.08,
                0.18,
                fc=plt.cm.viridis(i / T),
                ec="black",
                lw=0.5,
            )
        )
    t = np.linspace(0, T - 1, 100)
    y = 0.25 + 0.08 * np.sin(t * 1.5) + 0.06 * np.cos(t * 0.7)
    ax.plot(0.05 + (t + 0.5) * 0.09, y, color=C_MAMBA, lw=2.5)
    ax.text(0.41, 0.92, "continuous stimulus", ha="center", fontsize=10, color=C_GREY)
    ax.text(0.41, 0.05, "continuous brain", ha="center", fontsize=10, color=C_GREY)

    ax.text(
        0.85,
        0.5,
        "?",
        ha="center",
        va="center",
        fontsize=60,
        color=C_TRANS,
        fontweight="bold",
    )
    ax.text(0.41, 0.55, "co-evolution over time", ha="center", fontsize=11, fontweight="bold", color=C_TRANS)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    fig.suptitle("Stimulus alignment: where the literature stops", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "stimulus_gap.png")
    plt.close(fig)
    print(f"wrote {OUT/'stimulus_gap.png'}")


# ---------------------------------------------------------------------------
# Helper: pre-compute the hook figure data (shared by static + animated)
# ---------------------------------------------------------------------------
def _hook_data() -> dict:
    """Return the deterministic data arrays used by both fig_hook and fig_hook_gif."""
    rng = np.random.default_rng(7)
    T = 12
    t_fine = np.linspace(0, T - 1, 200)
    y = np.cumsum(rng.normal(0, 1, 200))
    y -= y.mean()
    y = np.convolve(y, np.ones(15) / 15, mode="same")
    obs_end = T - 4
    return dict(T=T, t_fine=t_fine, y=y, obs_end=obs_end)


# ---------------------------------------------------------------------------
# Figure 6 (animated): hook.gif
# ---------------------------------------------------------------------------
def _build_hook_axes():
    """Shared setup for the two hook GIFs: filmstrip + brain axes + observation
    band. Returns (fig, ax_bot, d, N).
    """
    d = _hook_data()
    T = d["T"]
    t_fine = d["t_fine"]
    y = d["y"]
    obs_end = d["obs_end"]
    N = len(t_fine)

    fig, axes = plt.subplots(
        2, 1, figsize=(10, 3.6), gridspec_kw=dict(height_ratios=[1, 2], hspace=0.05)
    )
    ax_top, ax_bot = axes
    _draw_filmstrip(ax_top, T)
    ax_top.set_xlim(0, T + 2)
    ax_top.set_ylim(0, 1)
    ax_top.set_yticks([])
    ax_top.set_xticks([])
    ax_top.text(-0.3, 0.5, "stimulus", ha="right", va="center", fontsize=11, color=C_GREY)

    ax_bot.set_xlim(0, T + 2)
    ax_bot.set_ylim(y.min() * 1.3, y.max() * 1.3)
    ax_bot.set_yticks([])
    ax_bot.set_xticks([])
    ax_bot.text(-0.3, np.mean(ax_bot.get_ylim()), "brain",
                ha="right", va="center", fontsize=11, color=C_GREY)
    ax_bot.set_xlabel("time (TRs) →", labelpad=2, color=C_GREY)

    # Observation band (static)
    ax_bot.axvspan(0.5, obs_end + 0.5, color=C_OURS, alpha=0.08)
    ax_bot.text(
        (0.5 + obs_end + 0.5) / 2, y.max() * 1.05,
        "observed past 30 s",
        ha="center", fontsize=10, color=C_OURS, style="italic",
    )
    return fig, ax_bot, d, N


def fig_hook_observe_gif() -> None:
    """Slide 2a: observe the past. Filmstrip + brain trajectory through the
    observation window only. No forecast band, no arrow. Frame 0 holds the
    complete state for PDF fallback.
    """
    fig, ax_bot, d, _ = _build_hook_axes()
    T = d["T"]
    t_fine = d["t_fine"]
    y = d["y"]
    obs_end = d["obs_end"]
    obs_idx = int(np.searchsorted(t_fine, obs_end))

    (traj_line,) = ax_bot.plot([], [], color=C_OURS, lw=2.5)

    def _complete() -> None:
        traj_line.set_data(t_fine[:obs_idx] + 0.5, y[:obs_idx])

    def _empty() -> None:
        traj_line.set_data([], [])

    REVEAL = 30

    def _reveal(i: int) -> None:
        frac = (i + 1) / REVEAL
        n = max(2, int(frac * obs_idx))
        traj_line.set_data(t_fine[:n] + 0.5, y[:n])

    plan = (
        [("complete", 2000)]
        + [("empty", 50)]
        + [(f"reveal_{i}", 50) for i in range(REVEAL)]
        + [("complete", 1500)]
    )

    def _set_state(name: str) -> None:
        if name == "complete":
            _complete()
        elif name == "empty":
            _empty()
        elif name.startswith("reveal_"):
            _reveal(int(name.split("_")[1]))

    images, durations = [], []
    for state, dur in plan:
        _set_state(state)
        fig.canvas.draw()
        images.append(_canvas_to_pil(fig))
        durations.append(dur)
    out_path = OUT / "hook_observe.gif"
    _save_gif(out_path, images, durations)
    plt.close(fig)
    size_kb = out_path.stat().st_size // 1024
    print(f"wrote {out_path}  ({size_kb} KB, {len(images)} frames)")


def fig_hook_forecast_gif() -> None:
    """Slide 2b: …we forecast. Filmstrip + full trajectory + forecast band +
    arrow. Starts from observation-only, reveals trajectory through forecast,
    then forecast band and arrow. Frame 0 holds the complete state.
    """
    fig, ax_bot, d, N = _build_hook_axes()
    T = d["T"]
    t_fine = d["t_fine"]
    y = d["y"]
    obs_end = d["obs_end"]
    obs_idx = int(np.searchsorted(t_fine, obs_end))

    (traj_line,) = ax_bot.plot([], [], color=C_OURS, lw=2.5)
    fore_span = ax_bot.axvspan(obs_end + 0.5, T + 0.5, color=C_ACCENT, alpha=0.0)
    fore_ann = ax_bot.annotate(
        "forecast: 5 TRs ahead",
        xy=(T + 0.3, y[-15]),
        xytext=(obs_end + 0.5, y.max() * 1.05),
        ha="left", fontsize=10, color=C_ACCENT,
        arrowprops=dict(arrowstyle="->", color=C_ACCENT, lw=1.8),
        alpha=0.0,
    )

    def _complete() -> None:
        traj_line.set_data(t_fine + 0.5, y)
        fore_span.set_alpha(0.18)
        fore_ann.set_alpha(1.0)

    def _obs_only() -> None:
        traj_line.set_data(t_fine[:obs_idx] + 0.5, y[:obs_idx])
        fore_span.set_alpha(0.0)
        fore_ann.set_alpha(0.0)

    REVEAL = 18

    def _reveal_traj(i: int) -> None:
        frac = (i + 1) / REVEAL
        n = obs_idx + max(2, int(frac * (N - obs_idx)))
        traj_line.set_data(t_fine[:n] + 0.5, y[:n])
        fore_span.set_alpha(0.0)
        fore_ann.set_alpha(0.0)

    def _traj_full_band() -> None:
        traj_line.set_data(t_fine + 0.5, y)
        fore_span.set_alpha(0.18)
        fore_ann.set_alpha(0.0)

    plan = (
        [("complete", 2000)]
        + [("obs_only", 50)]
        + [(f"reveal_{i}", 50) for i in range(REVEAL)]
        + [("traj_full_band", 350)]
        + [("complete", 1500)]
    )

    def _set_state(name: str) -> None:
        if name == "complete":
            _complete()
        elif name == "obs_only":
            _obs_only()
        elif name == "traj_full_band":
            _traj_full_band()
        elif name.startswith("reveal_"):
            _reveal_traj(int(name.split("_")[1]))

    images, durations = [], []
    for state, dur in plan:
        _set_state(state)
        fig.canvas.draw()
        images.append(_canvas_to_pil(fig))
        durations.append(dur)
    out_path = OUT / "hook_forecast.gif"
    _save_gif(out_path, images, durations)
    plt.close(fig)
    size_kb = out_path.stat().st_size // 1024
    print(f"wrote {out_path}  ({size_kb} KB, {len(images)} frames)")


# ---------------------------------------------------------------------------
# Helper: pre-compute mem_scaling data (shared by static + animated)
# ---------------------------------------------------------------------------
def _mem_scaling_data() -> dict:
    T_vals = np.array([32, 64, 96, 128, 192, 256, 384, 512, 768, 1024])
    mamba_gb = 6.09 * (T_vals / 256.0)
    coef = (6.09 * 32.0 / 256.0) / (32.0**2)
    trans_gb = coef * T_vals**2
    return dict(T_vals=T_vals, mamba_gb=mamba_gb, trans_gb=trans_gb)


# ---------------------------------------------------------------------------
# Figure 7 (animated): mem_scaling.gif
# ---------------------------------------------------------------------------
def fig_mem_scaling_gif() -> None:
    """Animated version of fig_memory_scaling().

    Frame plan (~4 s at 20 fps):
      Frame 0   : complete final state (PDF fallback — first frame)
      Reveal Mamba curve (15 frames)
      Reveal Transformer curve (15 frames)
      Demo point appears
      Hold complete 2 s
    """
    d = _mem_scaling_data()
    T_vals = d["T_vals"]
    mamba_gb = d["mamba_gb"]
    trans_gb = d["trans_gb"]
    N = len(T_vals)

    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    # --- Static background layers (always visible) ---
    ax.axvspan(256, 1024, color=C_OURS, alpha=0.06, zorder=0)
    ax.text(
        500, 500,
        "naturalistic-fMRI\nworking window",
        color=C_OURS, fontsize=10, ha="center", style="italic",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([32, 64, 128, 256, 512, 1024])
    ax.set_xticklabels(["32", "64", "128", "256", "512", "1024"])
    ax.set_ylim(1, 1e4)
    ax.set_xlabel("Sequence length T (TRs)")
    ax.set_ylabel("Activation memory per sample (GB, log scale)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

    # --- Animated artists ---
    (mamba_line,) = ax.plot([], [], "-o", color=C_OURS, lw=3, ms=7, label="Mamba — O(T)")
    (trans_line,) = ax.plot([], [], "-s", color=C_OTHER, lw=3, ms=7, label="Transformer attention — O(T²)")

    # Demo point (scatter) — we use a Line2D with a single point for animability
    (demo_pt,) = ax.plot([], [], "o", color=C_ACCENT, ms=13, zorder=5,
                         markeredgecolor="black", markeredgewidth=1)
    demo_ann = ax.annotate(
        "BOLDcast demo\n6.09 GB measured\nT=256, P=1,024",
        xy=(256, 6.09), xytext=(300, 50), fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", lw=1),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_ACCENT, lw=1.5),
        alpha=0.0,
    )

    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    def _set_complete() -> None:
        mamba_line.set_data(T_vals, mamba_gb)
        trans_line.set_data(T_vals, trans_gb)
        demo_pt.set_data([256], [6.09])
        demo_ann.set_alpha(1.0)

    def _set_empty() -> None:
        mamba_line.set_data([], [])
        trans_line.set_data([], [])
        demo_pt.set_data([], [])
        demo_ann.set_alpha(0.0)

    MAMBA_REVEAL_N = 15
    TRANS_REVEAL_N = 15

    def _set_mamba_reveal(i: int) -> None:
        n = max(2, int((i + 1) / MAMBA_REVEAL_N * N))
        mamba_line.set_data(T_vals[:n], mamba_gb[:n])
        trans_line.set_data([], [])
        demo_pt.set_data([], [])
        demo_ann.set_alpha(0.0)

    def _set_trans_reveal(i: int) -> None:
        mamba_line.set_data(T_vals, mamba_gb)
        n = max(2, int((i + 1) / TRANS_REVEAL_N * N))
        trans_line.set_data(T_vals[:n], trans_gb[:n])
        demo_pt.set_data([], [])
        demo_ann.set_alpha(0.0)

    def _set_curves_only() -> None:
        mamba_line.set_data(T_vals, mamba_gb)
        trans_line.set_data(T_vals, trans_gb)
        demo_pt.set_data([], [])
        demo_ann.set_alpha(0.0)

    # Frame plan: hold complete 2 s → reset → reveal Mamba → reveal Transformer
    # → demo point appears → hold complete 2 s
    plan = (
        [("complete", 2000)]
        + [("empty", 50)]
        + [(f"mamba_{i}", 60) for i in range(MAMBA_REVEAL_N)]
        + [(f"trans_{i}", 60) for i in range(TRANS_REVEAL_N)]
        + [("curves", 400)]
        + [("complete", 2000)]
    )

    def _set_state(name: str) -> None:
        if name == "complete":
            _set_complete()
        elif name == "empty":
            _set_empty()
        elif name == "curves":
            _set_curves_only()
        elif name.startswith("mamba_"):
            _set_mamba_reveal(int(name.split("_")[1]))
        elif name.startswith("trans_"):
            _set_trans_reveal(int(name.split("_")[1]))

    images, durations = [], []
    for state, dur in plan:
        _set_state(state)
        fig.canvas.draw()
        images.append(_canvas_to_pil(fig))
        durations.append(dur)
    out_path = OUT / "mem_scaling.gif"
    _save_gif(out_path, images, durations)
    plt.close(fig)
    size_kb = out_path.stat().st_size // 1024
    print(f"wrote {out_path}  ({size_kb} KB, {len(images)} frames)")


# ---------------------------------------------------------------------------
# Architecture diagram (slide 5 anchor)
# ---------------------------------------------------------------------------
def fig_architecture() -> None:
    """T-shape architecture diagram for slide 5.

    Visual code (deck-local, red-led):
      • solid MIT-Red border          = shipped (Days 1–3)
      • dashed grey border            = Phase 2 (funded, infrastructure ready)
      • thin grey border (no fill)    = external / frozen side-input
    Labels are terse; the legend + slide context carry the explanation.
    """
    from matplotlib.patches import FancyBboxPatch

    shipped = C_OURS                # MIT Red — shipped Phase-1 blocks
    phase2 = COLORS["body"]         # Dark gray — Phase-2 dashed
    accent = COLORS["accent"]
    bg_soft = COLORS["bg_soft"]
    body = COLORS["body"]
    ink = COLORS["ink"]

    fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, w, h, title, sub=None, *, edge=shipped, fill=bg_soft,
              dashed=False, lw=2.2, title_size=16, sub_size=11):
        ls = (0, (6, 3)) if dashed else "-"
        box = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.18",
            edgecolor=edge, facecolor=fill, linewidth=lw, linestyle=ls,
        )
        ax.add_patch(box)
        if sub:
            ax.text(x + w / 2, y + h - 0.38, title,
                    ha="center", va="center",
                    fontsize=title_size, color=edge, weight="bold")
            ax.text(x + w / 2, y + h - 0.95, sub,
                    ha="center", va="center",
                    fontsize=sub_size, color=body, style="italic")
        else:
            ax.text(x + w / 2, y + h / 2, title,
                    ha="center", va="center",
                    fontsize=title_size, color=edge, weight="bold")

    def shape_label(x_arrow, y_top, y_bot, text):
        ax.text(x_arrow - 0.25, (y_top + y_bot) / 2, text,
                ha="right", va="center", fontsize=10,
                color=body, family="monospace")

    def arrow(x1, y1, x2, y2, *, color=None, lw=2.2, style="-"):
        color = color if color is not None else shipped
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=0.45,head_width=0.28",
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
          "fMRI", edge=accent, lw=1.7, title_size=18)

    block(CX - half, Y_TOK_BOT, bw_main, Y_TOK_TOP - Y_TOK_BOT,
          "Tokenizer",
          "1,792 tokens / TR",
          title_size=18, sub_size=11)
    arrow(CX, Y_INPUT_BOT - 0.02, CX, Y_TOK_TOP + 0.02)
    shape_label(CX, Y_INPUT_BOT, Y_TOK_TOP, "(T, 91k)")

    bw_bb = 6.0
    half_bb = bw_bb / 2
    block(CX - half_bb, Y_BB_BOT, bw_bb, Y_BB_TOP - Y_BB_BOT,
          "Mamba ⊗ kNN",
          "× N layers · temporal O(T) · spatial k=8",
          title_size=20, sub_size=12)
    arrow(CX, Y_TOK_BOT - 0.02, CX, Y_BB_TOP + 0.02)
    shape_label(CX, Y_TOK_BOT, Y_BB_TOP, "(T, 1,792)")

    # Side input: sMRI → FiLM
    smri_x, smri_y, smri_w, smri_h = 12.6, 4.4, 2.6, 0.95
    block(smri_x, smri_y, smri_w, smri_h,
          "sMRI · FiLM", edge=accent, lw=1.6, title_size=13)
    arrow(smri_x, smri_y + smri_h / 2,
          CX + half_bb + 0.02, smri_y + smri_h / 2,
          color=accent, lw=1.6)

    # Two heads
    fc_x, fc_w = 1.6, 4.8
    st_x, st_w = 8.2, 4.4

    block(fc_x, Y_HEAD_BOT, fc_w, Y_HEAD_TOP - Y_HEAD_BOT,
          "Forecast",
          "Phase 1 · multi-step MSE",
          title_size=18, sub_size=12)

    block(st_x, Y_HEAD_BOT, st_w, Y_HEAD_TOP - Y_HEAD_BOT,
          "Stimulus",
          "Phase 2 · InfoNCE",
          edge=phase2, dashed=True,
          title_size=18, sub_size=12)

    fork_top = Y_BB_BOT - 0.02
    arrow(CX, fork_top, fc_x + fc_w / 2, Y_HEAD_TOP + 0.02)
    arrow(CX, fork_top, st_x + st_w / 2, Y_HEAD_TOP + 0.02,
          color=phase2, style=(0, (6, 3)))
    ax.text(CX, Y_BB_BOT - 0.55, "latent  (T, 1,792, d)",
            ha="center", va="center", fontsize=10.5,
            color=body, family="monospace")

    # Frozen CLIP → stimulus head
    clip_x, clip_w = 13.0, 2.6
    clip_h = 0.95
    clip_y = Y_HEAD_BOT + (Y_HEAD_TOP - Y_HEAD_BOT - clip_h) / 2
    block(clip_x, clip_y, clip_w, clip_h,
          "CLIP", "frozen ViT-L/14",
          edge=accent, lw=1.6, title_size=13, sub_size=9.5)
    arrow(clip_x, clip_y + clip_h / 2,
          st_x + st_w + 0.02, clip_y + clip_h / 2,
          color=accent, lw=1.6, style=(0, (3, 2)))

    # Legend (top-left): solid = shipped, dashed = Phase 2, gray = frozen
    lx, ly = 0.3, 9.55
    ax.add_patch(FancyBboxPatch(
        (lx, ly - 0.05), 0.45, 0.22,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        edgecolor=shipped, facecolor=bg_soft, linewidth=1.8,
    ))
    ax.text(lx + 0.6, ly + 0.06, "shipped · solid",
            ha="left", va="center", fontsize=11, color=ink)
    ax.add_patch(FancyBboxPatch(
        (lx, ly - 0.45), 0.45, 0.22,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        edgecolor=phase2, facecolor=bg_soft, linewidth=1.6, linestyle=(0, (5, 3)),
    ))
    ax.text(lx + 0.6, ly - 0.34, "Phase 2 · dashed",
            ha="left", va="center", fontsize=11, color=ink)
    ax.add_patch(FancyBboxPatch(
        (lx, ly - 0.85), 0.45, 0.22,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        edgecolor=accent, facecolor=bg_soft, linewidth=1.4,
    ))
    ax.text(lx + 0.6, ly - 0.74, "frozen / side-input",
            ha="left", va="center", fontsize=11, color=ink)

    fig.savefig(OUT / "architecture.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'architecture.png'}")


# ---------------------------------------------------------------------------
# Hybrid block (slide 9 left column) — replaces the ASCII stack
# ---------------------------------------------------------------------------
def fig_hybrid_block() -> None:
    """Mini-diagram of one hybrid Mamba ⊗ kNN ⊗ FiLM block. Compact (near-
    square) so it fits the left column of slide 9 without overflowing.
    """
    from matplotlib.patches import FancyBboxPatch

    shipped = C_OURS
    accent = COLORS["accent"]
    bg_soft = COLORS["bg_soft"]
    body = COLORS["body"]
    ink = COLORS["ink"]

    fig, ax = plt.subplots(figsize=(5.6, 6.0), dpi=140)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 9)
    ax.set_aspect("equal")
    ax.axis("off")

    def block(x, y, w, h, title, sub=None, *, edge=shipped, lw=2.0,
              title_size=15, sub_size=11):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.04,rounding_size=0.12",
            edgecolor=edge, facecolor=bg_soft, linewidth=lw,
        ))
        if sub:
            ax.text(x + w / 2, y + h - 0.32, title,
                    ha="center", va="center",
                    fontsize=title_size, color=edge, weight="bold")
            ax.text(x + w / 2, y + h - 0.78, sub,
                    ha="center", va="center",
                    fontsize=sub_size, color=body, style="italic")
        else:
            ax.text(x + w / 2, y + h / 2, title,
                    ha="center", va="center",
                    fontsize=title_size, color=edge, weight="bold")

    def arrow(x1, y1, x2, y2, *, color=None, lw=1.8):
        c = color if color is not None else shipped
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=0.3,head_width=0.2",
                        color=c, lw=lw, shrinkA=2, shrinkB=2))

    # Layout: main spine centered to leave a left band for the sMRI side-input.
    CX = 4.5
    bw = 3.6
    half = bw / 2

    # Token row at top, with label ABOVE so it does not collide with the arrow
    n_tok = 12
    tok_w = 0.30
    tok_gap = 0.04
    tok_total = n_tok * tok_w + (n_tok - 1) * tok_gap
    tok_x0 = CX - tok_total / 2
    ax.text(CX, 8.55, "tokens (T, P)", ha="center", va="bottom",
            fontsize=11, color=body, family="monospace")
    for i in range(n_tok):
        ax.add_patch(plt.Rectangle(
            (tok_x0 + i * (tok_w + tok_gap), 7.95), tok_w, 0.45,
            facecolor=bg_soft, edgecolor=ink, lw=0.5,
        ))

    # Mamba SSM
    Y_MAMBA_TOP, Y_MAMBA_BOT = 7.2, 6.1
    arrow(CX, 7.93, CX, Y_MAMBA_TOP + 0.02)
    block(CX - half, Y_MAMBA_BOT, bw, Y_MAMBA_TOP - Y_MAMBA_BOT,
          "Mamba SSM",
          r"$h_t = A\,h_{t-1} + B\,x_t$",
          title_size=16, sub_size=12)

    # Residual ⊕ + arrow
    arrow(CX, Y_MAMBA_BOT - 0.02, CX, 5.65)
    ax.text(CX, 5.5, "⊕  residual", ha="center", va="center",
            fontsize=11, color=body, style="italic")

    # kNN spatial mix
    Y_KNN_TOP, Y_KNN_BOT = 5.0, 3.9
    arrow(CX, 5.3, CX, Y_KNN_TOP + 0.02)
    block(CX - half, Y_KNN_BOT, bw, Y_KNN_TOP - Y_KNN_BOT,
          "kNN spatial mix",
          "k = 8 cortical neighbours",
          title_size=16, sub_size=12)

    # Residual ⊕ + arrow
    arrow(CX, Y_KNN_BOT - 0.02, CX, 3.45)
    ax.text(CX, 3.3, "⊕  residual", ha="center", va="center",
            fontsize=11, color=body, style="italic")

    # FiLM modulation
    Y_FILM_TOP, Y_FILM_BOT = 2.8, 1.7
    arrow(CX, 3.1, CX, Y_FILM_TOP + 0.02)
    block(CX - half, Y_FILM_BOT, bw, Y_FILM_TOP - Y_FILM_BOT,
          "FiLM modulation",
          r"$(\gamma, \beta)$ per channel",
          title_size=16, sub_size=12)

    # sMRI side input — placed LEFT of the FiLM block at the same vertical
    # center, with a clean horizontal arrow into FiLM's left edge.
    smri_y = (Y_FILM_TOP + Y_FILM_BOT) / 2  # ~2.25
    smri_h = 0.65
    smri_w = 1.4
    smri_x = 0.4
    ax.add_patch(FancyBboxPatch(
        (smri_x, smri_y - smri_h / 2), smri_w, smri_h,
        boxstyle="round,pad=0.04,rounding_size=0.06",
        edgecolor=accent, facecolor=bg_soft, linewidth=1.4,
    ))
    ax.text(smri_x + smri_w / 2, smri_y, "sMRI",
            ha="center", va="center", fontsize=12, color=accent, weight="bold")
    arrow(smri_x + smri_w + 0.02, smri_y,
          CX - half - 0.02, smri_y, color=accent, lw=1.4)

    # Output arrow
    arrow(CX, Y_FILM_BOT - 0.02, CX, 0.9)
    ax.text(CX, 0.7, "to next layer", ha="center", va="top",
            fontsize=11, color=body, family="monospace")

    fig.savefig(OUT / "hybrid_block.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'hybrid_block.png'}")


if __name__ == "__main__":
    fig_memory_scaling()
    fig_day3_metrics()
    fig_hook()
    fig_tokenization_wall()
    fig_stimulus_gap()
    fig_architecture()
    fig_hybrid_block()
    fig_hook_observe_gif()
    fig_hook_forecast_gif()
    fig_mem_scaling_gif()
    print(f"\nAll figures written to {OUT}/")
