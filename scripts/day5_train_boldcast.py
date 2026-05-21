"""Day-5 BOLDcast DDP training entry point.

NOTE: requires CUDA + mamba-ssm + real HCP dtseries access. Run on an
ORCD GPU compute node under the micromamba env via:
    torchrun --standalone --nproc-per-node=2 --nnodes=1 \\
        scripts/day5_train_boldcast.py --config configs/demo.yaml \\
        --out-dir results/day5_train
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# sys.path bootstrap (schist ID 197 — same as day4_overfit.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from boldcast.training import (  # noqa: E402
    Trainer,
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    get_local_rank,
    heldout_decreased_by,
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


def _read_subject_list(path: str) -> list[str]:
    """Read a one-subject-per-line text file, dropping comments and blanks."""
    out: list[str] = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override cfg.train.max_steps. Default uses config.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Override cfg.train.out_dir. Default uses config.",
    )
    p.add_argument(
        "--val-every",
        type=int,
        default=300,
        help="Held-out val cadence (steps).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Construct all objects but don't run any training step. "
            "Use for argparse + import-path smoke test."
        ),
    )
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    # CUDA gate — match day4_overfit.py pattern
    if not torch.cuda.is_available() and not args.dry_run:
        raise SystemExit("[day5] CUDA not available - run on a GPU node.")

    # DDP init (must happen BEFORE Trainer construction per Trainer.Notes).
    if is_distributed_run():
        init_distributed()
        local_rank = get_local_rank()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    max_steps = (
        args.max_steps if args.max_steps is not None else int(cfg.train.max_steps)
    )
    out_dir = Path(
        args.out_dir if args.out_dir is not None else cfg.train.out_dir
    )
    if is_rank_zero():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Seed: same model init on all ranks; samplers/augs diverge via DistributedSampler.
    seed_everything(int(cfg.seed))

    # Tokenization caches: only rank-0 builds them; other ranks wait.
    # In practice the cache files are typically pre-built before this script
    # runs (Day-2). We still ALWAYS load — build_or_load_patches and
    # build_or_load_knn are cache-hit fast paths if the files exist.
    train_subjects = _read_subject_list(str(cfg.data.subjects_train_file))
    heldout_subjects = _read_subject_list(str(cfg.data.subjects_heldout_file))
    ref_subject = train_subjects[0]
    ref_run = cfg.data.runs[0]

    if is_rank_zero():
        print(f"[day5] reference subject={ref_subject}, run={ref_run}")
        print(
            f"[day5] train subjects: {len(train_subjects)}, "
            f"heldout: {len(heldout_subjects)}"
        )

    if args.dry_run:
        # Skip actual data loads + heavy construction; just confirm imports
        # and argparse. Useful for CPU-side smoke tests.
        if is_rank_zero():
            print(
                f"[day5] dry-run mode: skipping data load, max_steps={max_steps}, "
                f"out_dir={out_dir}, val_every={args.val_every}"
            )
        if is_distributed_run():
            cleanup_distributed()
        return 0

    # --- Heavy imports deferred to here so --dry-run works in CPU/uv env ---
    from boldcast._upstream.cifti_io import (  # noqa: E402
        cortex_grayordinate_indices,
        load_dtseries,
    )
    from boldcast.data.hcp_rest import HCPRestingDataset  # noqa: E402
    from boldcast.models.boldcast_demo import BOLDcastDemo  # noqa: E402
    from boldcast.tokenize.geodesic import build_or_load_patches  # noqa: E402
    from boldcast.tokenize.knn import build_or_load_knn  # noqa: E402

    ref_path = str(cfg.data.dtseries_pattern).format(
        subject=ref_subject, run=ref_run,
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
        print(f"[day5] loading patch assignment from {cfg.tokenize.patch_cache}")
    assignment = build_or_load_patches(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(cfg.tokenize.patch_cache),
        n_patches=int(cfg.tokenize.n_patches_cortex),
    )

    if is_rank_zero():
        print(f"[day5] loading kNN from {cfg.tokenize.knn_cache}")
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

    # Datasets: full train list (all 4 runs), heldout for val.
    train_ds = HCPRestingDataset(
        subjects=train_subjects,
        runs=list(cfg.data.runs),
        dtseries_pattern=str(cfg.data.dtseries_pattern),
        cache_dir=str(cfg.tokenize.cache_dir),
        patch_assignment=assignment,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        window_size=int(cfg.window.size),
        stride=int(cfg.window.stride),
    )
    val_ds = HCPRestingDataset(
        subjects=heldout_subjects,
        runs=list(cfg.data.runs),
        dtseries_pattern=str(cfg.data.dtseries_pattern),
        cache_dir=str(cfg.tokenize.cache_dir),
        patch_assignment=assignment,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        window_size=int(cfg.window.size),
        stride=int(cfg.window.stride),
        subject_id_offset=len(train_subjects),
    )

    # DistributedSampler on TRAIN (shuffle + per-rank partition).
    # NO sampler on val — held-out is small and rank-0 consumes the whole set.
    train_sampler: DistributedSampler[int] | None
    if is_distributed_run():
        train_sampler = DistributedSampler(
            train_ds, shuffle=True, seed=int(cfg.seed),
        )
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

    # Model + DDP wrap.
    horizons = tuple(int(h) for h in cfg.train.forecasting_horizons)
    if is_rank_zero():
        print(f"[day5] horizons={horizons}, building BOLDcastDemo on {device} ...")
    _demo_model = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=adjacency,
        horizons=horizons,
        use_checkpoint=True,
    ).to(device)
    n_params = sum(p.numel() for p in _demo_model.parameters())
    if is_rank_zero():
        print(f"[day5]   params: {n_params/1e6:.3f} M")
    model = setup_model_for_ddp(
        _demo_model,
        find_unused_parameters=bool(cfg.ddp.find_unused_parameters),
    )

    # Optimizer + scheduler (full config — no Day-4 overrides).
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
        print(f"[day5] fitting for {max_steps} steps, val_every={args.val_every} ...")
    history = trainer.fit(
        train_loader,
        max_steps=max_steps,
        sampler=train_sampler,
        val_loader=val_loader,
        val_every=int(args.val_every),
    )

    # Final checkpoint (rank-0 only; unwrap DDP via .module if wrapped).
    if is_rank_zero():
        raw_model = getattr(model, "module", model)
        ckpt_path = out_dir / "ckpt_final.pt"
        save_checkpoint(raw_model, optimizer, step=max_steps, path=ckpt_path)
        print(f"[day5] final checkpoint -> {ckpt_path}")
        print(
            f"[day5] train loss steps: {len(history['loss'])}, "
            f"val measurements: {len(history['val_loss'])}"
        )

    # Acceptance criterion #2: held-out val loss decreased ≥30%.
    if is_rank_zero():
        if not heldout_decreased_by(history, frac=0.30, window=3):
            val_loss = history["val_loss"]
            first_3 = val_loss[:3] if len(val_loss) >= 3 else val_loss
            last_3 = val_loss[-3:] if len(val_loss) >= 3 else val_loss
            print(
                f"[day5] heldout acceptance FAILED: "
                f"first-3-mean={sum(first_3)/max(len(first_3),1):.6f}, "
                f"last-3-mean={sum(last_3)/max(len(last_3),1):.6f} "
                f"(need ≥30% drop)."
            )
            # cleanup before exit
            if is_distributed_run():
                cleanup_distributed()
            raise SystemExit(1)
        val_loss = history["val_loss"]
        print(
            f"[day5] heldout acceptance PASSED: "
            f"first-3-mean={sum(val_loss[:3])/3:.6f}, "
            f"last-3-mean={sum(val_loss[-3:])/3:.6f}."
        )

    if is_distributed_run():
        cleanup_distributed()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
