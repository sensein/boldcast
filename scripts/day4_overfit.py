"""Day-4 BOLDcastDemo overfit sanity check.

NOTE: requires CUDA + mamba-ssm + real HCP dtseries access. Run on an
ORCD GPU compute node under the micromamba env. Claude does not execute
this script (no GPU at the login node; uv env lacks mamba-ssm; HCP DUA
is held by Yibei, not Claude).

Builds a 4-window dataset (4 train subjects x 1 run x 1 window each),
trains BOLDcastDemo for up to ``--max-steps`` iterations on that batch,
and asserts the final loss is < 1% of the initial loss (ADR 0005 D5).
Writes:
  - {out_dir}/loss_log.jsonl    per-step {step, loss, lr}
  - {out_dir}/ckpt_overfit.pt   final-step checkpoint (save smoke test)
  - figures/day4_overfit_curve.png   loss curve PNG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# sys.path bootstrap (ADR 0005 inherits the Day-1 pattern). Allows
# ``python scripts/day4_overfit.py`` to work from any worktree without
# an editable install in the active env.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from boldcast._upstream.cifti_io import (  # noqa: E402
    cortex_grayordinate_indices,
    load_dtseries,
)
from boldcast.data.hcp_rest import HCPRestingDataset  # noqa: E402
from boldcast.models.boldcast_demo import BOLDcastDemo  # noqa: E402
from boldcast.tokenize.geodesic import build_or_load_patches  # noqa: E402
from boldcast.tokenize.knn import build_or_load_knn  # noqa: E402
from boldcast.training import (  # noqa: E402
    Trainer,
    build_optimizer,
    save_checkpoint,
    seed_everything,
)
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402


def _read_subject_list(path: str) -> list[str]:
    out: list[str] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def _plot_loss_curve(jsonl_path: Path, png_path: Path) -> None:
    """Read the trainer's JSONL log and render a loss-vs-step PNG."""
    import json

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps: list[int] = []
    losses: list[float] = []
    with open(jsonl_path) as fh:
        for line in fh:
            r = json.loads(line)
            steps.append(int(r["step"]))
            losses.append(float(r["loss"]))

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, losses, linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("forecasting MSE (log)")
    ax.set_title("Day-4 overfit: 4 windows, BF16 + activation checkpointing")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--out-dir", default="results/day4_overfit")
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available():
        raise SystemExit("[day4] CUDA not available - run on a GPU node.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.seed))

    # Reference subject (first train subject) for patch + kNN cache build.
    train_subjects = _read_subject_list(str(cfg.data.subjects_train_file))
    ref_subject = train_subjects[0]
    ref_run = cfg.data.runs[0]
    ref_path = str(cfg.data.dtseries_pattern).format(
        subject=ref_subject, run=ref_run
    )
    print(f"[day4] reference subject={ref_subject}, run={ref_run}")
    _, header = load_dtseries(ref_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)

    surface_dir = str(cfg.data.surface_dir_template).format(
        subject=ref_subject
    )
    lh_mesh = (
        f"{surface_dir}/{ref_subject}.L.midthickness_MSMAll."
        "32k_fs_LR.surf.gii"
    )
    rh_mesh = (
        f"{surface_dir}/{ref_subject}.R.midthickness_MSMAll."
        "32k_fs_LR.surf.gii"
    )

    print(f"[day4] loading patch assignment from {cfg.tokenize.patch_cache}")
    assignment = build_or_load_patches(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(cfg.tokenize.patch_cache),
        n_patches=int(cfg.tokenize.n_patches_cortex),
    )

    print(f"[day4] loading kNN from {cfg.tokenize.knn_cache}")
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
    adjacency = torch.from_numpy(np.asarray(adjacency_np)).long().cuda()

    # 4 subjects x 1 run x 1 window dataset (ADR 0005 D5). stride > T_full
    # forces a single window-start per (subject, run).
    overfit_subjects = train_subjects[:4]
    overfit_runs = [str(cfg.data.runs[0])]
    overfit_stride = int(cfg.window.size) + 10_000
    ds = HCPRestingDataset(
        subjects=overfit_subjects,
        runs=overfit_runs,
        dtseries_pattern=str(cfg.data.dtseries_pattern),
        cache_dir=str(cfg.tokenize.cache_dir),
        patch_assignment=assignment,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        window_size=int(cfg.window.size),
        stride=overfit_stride,
    )
    if len(ds) != 4:
        raise SystemExit(
            f"[day4] expected 4 windows in overfit dataset, got {len(ds)}"
        )
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

    # Model with checkpointing on (matches canonical Day-3 / Day-5 regime).
    horizons = tuple(int(h) for h in cfg.train.forecasting_horizons)
    print(f"[day4] horizons={horizons}, building BOLDcastDemo on cuda ...")
    model = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=adjacency,
        horizons=horizons,
        use_checkpoint=True,
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[day4]   params: {n_params/1e6:.3f} M")

    # Optimizer: cfg.train.lr, but override weight_decay=0 (ADR 0005 D6).
    optimizer = build_optimizer(
        model,
        lr=float(cfg.train.lr),
        weight_decay=0.0,                          # OVERRIDE
        betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
    )
    # Scheduler: override to constant LR (ADR 0005 D6).
    scheduler = None

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cuda"),
        horizons=horizons,
        grad_clip_norm=float(cfg.train.grad_clip_norm),
        precision="bf16",
        log_every=10,
        out_dir=out_dir,
    )

    print(f"[day4] fitting for {args.max_steps} steps ...")
    history = trainer.fit(loader, max_steps=int(args.max_steps))

    initial = history["loss"][0]
    final = history["loss"][-1]
    ratio = final / initial if initial > 0 else float("inf")
    print(
        f"[day4] initial loss = {initial:.6f}  "
        f"final loss = {final:.6f}  ratio = {ratio:.4%}"
    )
    if not (final < 0.01 * initial):
        raise SystemExit(
            f"[day4] overfit FAILED: final loss is {ratio:.4%} of initial "
            "(target < 1%)."
        )
    print("[day4] overfit acceptance PASSED (final < 1% of initial).")

    ckpt_path = out_dir / "ckpt_overfit.pt"
    save_checkpoint(model, optimizer, step=int(args.max_steps), path=ckpt_path)
    print(f"[day4] checkpoint saved to {ckpt_path}")

    png_path = Path("figures") / "day4_overfit_curve.png"
    _plot_loss_curve(out_dir / "loss_log.jsonl", png_path)
    print(f"[day4] loss curve saved to {png_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
