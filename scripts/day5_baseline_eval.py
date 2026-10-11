"""Day-5 trivial-baseline val-loss evaluation.

Computes forecasting MSE on the held-out val_loader for three reference
predictors that require no learning, alongside (optionally) the trained
``ckpt_final.pt`` model evaluated on the same batches:

- ``predict-zero``        pred = 0 everywhere
- ``predict-input``       pred[:, t, p, h, :] = tokens[:, t, p, :]
                          (temporal persistence: "future TR = current TR")
- ``predict-window-mean`` pred[:, t, p, h, :] = tokens[:, :T_valid, p, :].mean(dim=1)
                          (constant per-window per-channel mean)
- ``model``               pred = ckpt(tokens) at the first ``T_valid`` output
                          positions (matches Trainer._eval)

All four use the same ``boldcast.training.loss.forecasting_loss`` reduction
(``mean`` over (B, T_valid, P, H, d_in)) so the numbers are directly
comparable to the ``val_loss=0.21861`` reported in
``results/day5_train/loss_log.jsonl``.

Output is sanitized: scalar losses only, no batch contents, no subject IDs.

Run via sbatch wrapper (HCP DUA: data loads on the compute node only)::

    sbatch scripts/day5_baseline_eval.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from boldcast.eval.baselines import compute_trivial_baselines  # noqa: E402


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument(
        "--ckpt",
        default="results/day5_train/ckpt_final.pt",
        help="Path to trained checkpoint. Pass '' to skip the model arm.",
    )
    p.add_argument(
        "--out-json",
        default="results/day5_train/baseline_eval.json",
        help="Where to write the summary JSON.",
    )
    args = p.parse_args()

    from boldcast.utils.env import load_repo_dotenv

    load_repo_dotenv(_REPO_ROOT)

    import numpy as np
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available():
        raise SystemExit("[baseline_eval] CUDA not available — needs a GPU node.")
    device = torch.device("cuda")

    # Heavy imports deferred so import errors don't crash --help.
    from boldcast._upstream.cifti_io import (
        cortex_grayordinate_indices,
        load_dtseries,
    )
    from boldcast.data.hcp_rest import HCPRestingDataset
    from boldcast.models.boldcast_demo import BOLDcastDemo
    from boldcast.tokenize.geodesic import build_or_load_patches
    from boldcast.tokenize.knn import build_or_load_knn

    heldout_subjects: list[str] = []
    with open(str(cfg.data.subjects_heldout_file)) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                heldout_subjects.append(s)

    train_subjects: list[str] = []
    with open(str(cfg.data.subjects_train_file)) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                train_subjects.append(s)

    ref_subject = train_subjects[0]
    ref_run = cfg.data.runs[0]
    ref_path = str(cfg.data.dtseries_pattern).format(subject=ref_subject, run=ref_run)
    _, header = load_dtseries(ref_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)

    surface_dir = str(cfg.data.surface_dir_template).format(subject=ref_subject)
    lh_mesh = f"{surface_dir}/{ref_subject}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    rh_mesh = f"{surface_dir}/{ref_subject}.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

    assignment = build_or_load_patches(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(cfg.tokenize.patch_cache),
        n_patches=int(cfg.tokenize.n_patches_cortex),
    )
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
    val_loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        val_ds,
        batch_size=int(cfg.eval.batch_size_per_gpu),
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    horizons = tuple(int(h) for h in cfg.train.forecasting_horizons)

    model: torch.nn.Module | None = None
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
        if not ckpt_path.exists():
            raise SystemExit(f"[baseline_eval] ckpt not found: {ckpt_path}")
        model = BOLDcastDemo(
            d_in=1,
            d_model=128,
            n_layers=4,
            n_patches=int(cfg.tokenize.n_patches_cortex),
            k_neighbors=int(cfg.tokenize.knn_k),
            adjacency=adjacency,
            horizons=horizons,
            use_checkpoint=False,  # eval-only — no need
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        # save_checkpoint (boldcast/training/utils.py:120) stores under "model".
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state)
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[baseline_eval] loaded ckpt from {ckpt_path} "
            f"(step={ckpt.get('step', '?')}); params={n_params / 1e6:.3f}M"
        )

    print(
        f"[baseline_eval] heldout subjects={len(heldout_subjects)}, "
        f"horizons={horizons}, val batches will be counted below."
    )

    results = compute_trivial_baselines(val_loader, horizons, device, model=model)

    print("[baseline_eval] results (lower = better):")
    for key in ("zero", "input", "window_mean", "model"):
        if key in results:
            print(f"  {key:>14s}: {results[key]:.6f}")
    if "model" in results:
        rel_zero = (results["zero"] - results["model"]) / results["zero"]
        rel_input = (results["input"] - results["model"]) / results["input"]
        rel_wmean = (results["window_mean"] - results["model"]) / results["window_mean"]
        print("[baseline_eval] model improvement over baselines:")
        print(f"  vs predict-zero       : {rel_zero * 100:+.2f}%")
        print(f"  vs predict-input      : {rel_input * 100:+.2f}%")
        print(f"  vs predict-window-mean: {rel_wmean * 100:+.2f}%")

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"[baseline_eval] wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
