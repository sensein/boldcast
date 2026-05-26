"""Day-6 Schaefer-400 ROI baseline training entry point.

Near-clone of ``scripts/day5_train_boldcast.py`` with:

* Model: :class:`boldcast.models.baseline.BaselineSchaefer400` (P=400)
  instead of :class:`boldcast.models.boldcast_demo.BOLDcastDemo` (P=1024).
* Tokenization: Schaefer-400 parcellation instead of geodesic FPS
  patches.  All other hyperparameters identical (matched-architecture
  comparison; see ``docs/10_day_plan.md`` Day 6).

NOTE: requires CUDA + mamba-ssm + real HCP dtseries access + the
Schaefer dlabel at ``cfg.baseline.schaefer_dlabel``.  Run on an ORCD
GPU compute node under the micromamba env via:

    sbatch scripts/day6_train_schaefer.sh

or directly with torchrun:

    torchrun --standalone --nproc-per-node=2 --nnodes=1 \\
        scripts/day6_train_schaefer.py --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from boldcast.eval.baselines import compute_trivial_baselines  # noqa: E402
from boldcast.training import (  # noqa: E402
    Trainer,
    beats_best_baseline,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    get_local_rank,
    init_distributed,
    is_distributed_run,
    is_rank_zero,
    save_checkpoint,
    seed_everything,
    setup_model_for_ddp,
)
from omegaconf import OmegaConf  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch.utils.data.distributed import DistributedSampler  # noqa: E402


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--val-every", type=int, default=300)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct objects without training; CPU smoke test.",
    )
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("[day6] CUDA not available - run on a GPU node.")

    if is_distributed_run():
        init_distributed()
        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_steps = args.max_steps if args.max_steps is not None else int(cfg.train.max_steps)
    out_dir = Path(args.out_dir if args.out_dir is not None else cfg.baseline.out_dir)
    if is_rank_zero():
        out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(int(cfg.seed))

    if args.dry_run:
        if is_rank_zero():
            print(
                f"[day6] dry-run: max_steps={max_steps}, out_dir={out_dir}, "
                f"val_every={args.val_every}"
            )
        if is_distributed_run():
            cleanup_distributed()
        return 0

    # --- Heavy imports deferred so --dry-run works without CUDA/mamba ---
    from boldcast._upstream.cifti_io import (  # noqa: E402
        cortex_grayordinate_indices,
        load_dtseries,
    )
    from boldcast.data.schaefer_baseline import (  # noqa: E402
        build_schaefer_dataset_from_config,
        load_schaefer_cortex_assignment,
    )
    from boldcast.models.baseline import BaselineSchaefer400  # noqa: E402
    from boldcast.tokenize.knn import build_or_load_knn  # noqa: E402

    n_rois = int(cfg.baseline.n_rois)
    schaefer_dlabel = str(cfg.baseline.schaefer_dlabel)

    # Reference subject's CIFTI header gives us the HCP grayordinate vertex
    # indices we need to (a) load the Schaefer dlabel correctly and (b) build
    # the kNN over the same mesh.
    train_subjects_file = Path(str(cfg.data.subjects_train_file))
    ref_subject = train_subjects_file.read_text().strip().split("\n")[0].strip()
    ref_run = cfg.data.runs[0]
    ref_path = str(cfg.data.dtseries_pattern).format(subject=ref_subject, run=ref_run)
    _, header = load_dtseries(ref_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)

    # Schaefer parcel assignment per HCP cortex grayordinate (LH then RH, 0..399).
    if is_rank_zero():
        print(f"[day6] loading Schaefer dlabel from {schaefer_dlabel}")
    assignment = load_schaefer_cortex_assignment(
        schaefer_dlabel,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        n_rois=n_rois,
    )

    surface_dir = str(cfg.data.surface_dir_template).format(subject=ref_subject)
    lh_mesh = f"{surface_dir}/{ref_subject}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    rh_mesh = f"{surface_dir}/{ref_subject}.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

    if is_rank_zero():
        print(f"[day6] building/loading Schaefer kNN at {cfg.baseline.knn_cache}")
    adjacency_np = build_or_load_knn(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        patch_assignment=assignment,
        n_patches=n_rois,
        k=int(cfg.tokenize.knn_k),
        cache_path=str(cfg.baseline.knn_cache),
    )
    adjacency = torch.from_numpy(np.asarray(adjacency_np)).long().to(device)

    train_ds = build_schaefer_dataset_from_config(args.config, split="train")
    val_ds = build_schaefer_dataset_from_config(args.config, split="heldout")

    train_sampler: DistributedSampler[int] | None
    if is_distributed_run():
        train_sampler = DistributedSampler(train_ds, shuffle=True, seed=int(cfg.seed))
    else:
        train_sampler = None

    batch_size = int(cfg.train.batch_size_per_gpu)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg.eval.batch_size_per_gpu),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    horizons = tuple(int(h) for h in cfg.train.forecasting_horizons)
    if is_rank_zero():
        print(f"[day6] horizons={horizons}, building BaselineSchaefer400 on {device} ...")
    _model = BaselineSchaefer400(
        d_in=1,
        d_model=128,
        n_layers=4,
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=adjacency,
        horizons=horizons,
        use_checkpoint=True,
    ).to(device)
    n_params = sum(p.numel() for p in _model.parameters())
    if is_rank_zero():
        print(f"[day6]   params: {n_params/1e6:.3f} M")
    model = setup_model_for_ddp(
        _model,
        find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
    )

    optimizer = build_optimizer(
        model,
        lr=float(cfg.train.lr),
        weight_decay=float(cfg.train.weight_decay),
        betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
    )
    scheduler = build_scheduler(
        optimizer,
        schedule=str(cfg.train.schedule),  # type: ignore[arg-type]
        warmup_steps=int(cfg.train.warmup_steps),
        max_steps=max_steps,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        horizons=horizons,
        grad_clip_norm=float(cfg.train.grad_clip_norm),
        precision=str(cfg.train.precision),  # type: ignore[arg-type]
        log_every=int(cfg.train.log_every),
        ckpt_every=int(cfg.train.ckpt_every),
        out_dir=out_dir,
    )

    if is_rank_zero():
        print(f"[day6] fitting for {max_steps} steps, val_every={args.val_every} ...")
    history = trainer.fit(
        train_loader,
        max_steps=max_steps,
        sampler=train_sampler,
        val_loader=val_loader,
        val_every=int(args.val_every),
    )

    if is_rank_zero():
        raw_model = getattr(model, "module", model)
        ckpt_path = out_dir / "ckpt_final.pt"
        save_checkpoint(raw_model, optimizer, step=max_steps, path=ckpt_path)
        print(f"[day6] final checkpoint -> {ckpt_path}")
        print(
            f"[day6] train loss steps: {len(history['loss'])}, "
            f"val measurements: {len(history['val_loss'])}"
        )

    # Acceptance criterion #2: held-out val loss beats strongest trivial
    # baseline by ≥15% (Cohen's large-effect R²; see
    # docs/superpowers/specs/2026-05-24-acceptance-gate-baseline-relative-design.md).
    if is_rank_zero():
        if not history["val_loss"]:
            print(
                "[day6] baseline acceptance FAILED: "
                "no val measurements recorded "
                "(history['val_loss'] is empty — check val_every config)."
            )
            if is_distributed_run():
                cleanup_distributed()
            raise SystemExit(1)
        model_val_loss = history["val_loss"][-1]
        baselines = compute_trivial_baselines(
            val_loader, horizons, device, model=None
        )
        best_name = min(baselines, key=lambda k: baselines[k])
        best_val = baselines[best_name]
        improvement_pct = (best_val - model_val_loss) / best_val * 100.0
        gate_frac = 0.15
        gate_pct = gate_frac * 100.0
        if not beats_best_baseline(model_val_loss, baselines, frac=gate_frac):
            print(
                f"[day6] baseline acceptance FAILED: "
                f"model={model_val_loss:.6f}, "
                f"best baseline={best_val:.6f} ({best_name}), "
                f"improvement={improvement_pct:+.2f}% over best baseline, "
                f"gate={gate_pct:.1f}%."
            )
            if is_distributed_run():
                cleanup_distributed()
            raise SystemExit(1)
        print(
            f"[day6] baseline acceptance PASSED: "
            f"model={model_val_loss:.6f}, "
            f"best baseline={best_val:.6f} ({best_name}), "
            f"improvement={improvement_pct:+.2f}% over best baseline, "
            f"gate={gate_pct:.1f}%."
        )

    if is_distributed_run():
        cleanup_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
