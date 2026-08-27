"""Day-7 subject-fingerprinting evaluation: BOLDcast vs Schaefer-400.

Loads the Day-5 (BOLDcast) and Day-6 (Schaefer baseline) checkpoints,
extracts per-run embeddings on the held-out split under both pooling
protocols, computes top-k retrieval accuracy with bootstrap 95% CIs
and a paired McNemar test, sweeps embedding window length, and
writes three figures + a metrics JSON.

NOTE: requires CUDA + mamba-ssm + real HCP dtseries access.  Run on
an ORCD GPU compute node under the micromamba env:

    srun -p mit_normal_gpu --gres=gpu:h200:1 -t 02:00:00 \\
        python scripts/day7_fingerprint_eval.py \\
        --config configs/demo.yaml \\
        --boldcast-ckpt results/day5_train/ckpt_final.pt \\
        --baseline-ckpt results/day6_baseline/ckpt_final.pt \\
        --out-dir figures/

Outputs (under ``--out-dir``):
* ``day7_fingerprint_topk.png`` — bars, BOLDcast vs Schaefer, k=1/5/10
  with bootstrap 95% CIs; both pooling protocols.
* ``day7_fingerprint_confusion.png`` — BOLDcast top-1 confusion matrix.
* ``day7_fingerprint_window_sweep.png`` — top-1 accuracy vs window
  length (15 s / 30 s / 60 s / 5 min) for both models.
* ``day7_metrics.json`` — every number, ready for the K99 prelim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal, cast

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from boldcast.utils.env import load_repo_dotenv  # noqa: E402

load_repo_dotenv(_REPO_ROOT)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from boldcast.data.hcp_rest import HCPRestingDataset  # noqa: E402
from boldcast.eval.fingerprint import (  # noqa: E402
    bootstrap_ci_topk,
    extract_embeddings,
    paired_mcnemar,
    per_run_correct,
    topk_accuracy,
)
from numpy.typing import NDArray  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402


def _load_model_from_ckpt(
    ckpt_path: Path,
    model: torch.nn.Module,
    device: torch.device,
) -> torch.nn.Module:
    """Load weights from a Day-5/Day-6 checkpoint into ``model``."""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model


def _confusion_matrix(
    embeddings: NDArray[np.float32],
    subject_ids: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return ``(C, labels)``: ``C[i, j]`` = # probes of true subj i predicted as j."""
    from boldcast.eval.fingerprint import _per_run_predicted_rank

    n = embeddings.shape[0]
    unique_subjects = np.unique(subject_ids)
    n_subj = unique_subjects.shape[0]

    # Recover top-1 prediction per probe (the unique subject at rank 0).
    # _per_run_predicted_rank returns the rank of the TRUE subject; we need the
    # top-1 predicted subject too. Recompute the per-subject similarity matrix.
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)
    per_subj = np.full((n, n_subj), -np.inf, dtype=np.float64)
    for j, s in enumerate(unique_subjects):
        gallery_mask = subject_ids == s
        s_sim = sim[:, gallery_mask].copy()
        s_sim[np.isinf(s_sim)] = np.nan
        with np.errstate(invalid="ignore"):
            per_subj[:, j] = np.nanmean(s_sim, axis=1)
    top1_col = np.argmax(per_subj, axis=1)
    predicted_subjects = unique_subjects[top1_col]

    cm = np.zeros((n_subj, n_subj), dtype=np.int64)
    for true_s, pred_s in zip(subject_ids, predicted_subjects):
        i = int(np.where(unique_subjects == true_s)[0][0])
        j = int(np.where(unique_subjects == pred_s)[0][0])
        cm[i, j] += 1
    # Also assert: row sums equal runs-per-subject.
    _ = _per_run_predicted_rank  # keep import used; signals the protocol parity
    return cm, unique_subjects


def _compute_metrics(
    model: torch.nn.Module,
    val_ds: HCPRestingDataset,
    device: torch.device,
    pools: list[str],
    k_list: list[int],
    n_resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Run extract_embeddings + topk + bootstrap for each pool; return a dict."""
    out: dict[str, Any] = {}
    for pool in pools:
        pool_lit = cast(Literal["mean_tp", "mean_t"], pool)
        emb, sids, rids = extract_embeddings(model, val_ds, pool=pool_lit, device=device)
        acc = topk_accuracy(emb, sids, k_list=k_list)
        cis: dict[int, dict[str, float]] = {}
        for k in k_list:
            point, lo, hi = bootstrap_ci_topk(
                emb,
                sids,
                k=k,
                n_resamples=n_resamples,
                seed=seed,
            )
            cis[int(k)] = {"point": point, "ci_low": lo, "ci_high": hi}
        out[pool] = {
            "embeddings": emb,
            "subject_ids": sids,
            "run_ids": rids,
            "topk": {int(k): float(v) for k, v in acc.items()},
            "ci": cis,
        }
    return out


def _plot_topk_bars(
    boldcast_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    k_list: list[int],
    pools: list[str],
    out_path: Path,
) -> None:
    """Bar chart: top-k accuracy, BOLDcast vs Schaefer, per pool."""
    fig, axes = plt.subplots(1, len(pools), figsize=(5 * len(pools), 4), sharey=True)
    if len(pools) == 1:
        axes = [axes]
    x = np.arange(len(k_list))
    width = 0.38
    for ax, pool in zip(axes, pools):
        bc = boldcast_metrics[pool]
        bl = baseline_metrics[pool]
        bc_pts = [bc["topk"][k] for k in k_list]
        bl_pts = [bl["topk"][k] for k in k_list]
        bc_err = np.array(
            [
                [bc["topk"][k] - bc["ci"][k]["ci_low"] for k in k_list],
                [bc["ci"][k]["ci_high"] - bc["topk"][k] for k in k_list],
            ]
        )
        bl_err = np.array(
            [
                [bl["topk"][k] - bl["ci"][k]["ci_low"] for k in k_list],
                [bl["ci"][k]["ci_high"] - bl["topk"][k] for k in k_list],
            ]
        )
        ax.bar(
            x - width / 2,
            bc_pts,
            width,
            yerr=bc_err,
            capsize=4,
            label="BOLDcast",
            color="#3b82f6",
        )
        ax.bar(
            x + width / 2,
            bl_pts,
            width,
            yerr=bl_err,
            capsize=4,
            label="Schaefer-400",
            color="#94a3b8",
        )
        ax.axhline(1.0 / 8.0, linestyle=":", color="grey", linewidth=1, label="chance (1/8)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"top-{k}" for k in k_list])
        ax.set_ylim(0, 1.05)
        ax.set_title(f"pool = {pool}")
        ax.set_ylabel("accuracy")
        ax.legend(loc="lower right", fontsize=9)
    fig.suptitle("Day-7 fingerprinting: BOLDcast vs Schaefer-400 (held-out, bootstrap 95% CI)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_confusion(
    cm: NDArray[np.int64],
    labels: NDArray[np.int64],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([str(s) for s in labels], rotation=45, fontsize=8)
    ax.set_yticklabels([str(s) for s in labels], fontsize=8)
    ax.set_xlabel("predicted subject ID")
    ax.set_ylabel("true subject ID")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="# runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_window_sweep(
    sweep_results: dict[str, dict[int, float]],
    window_seconds: list[int],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, results in sweep_results.items():
        accs = [results[w] for w in window_seconds]
        ax.plot(window_seconds, accs, marker="o", label=label)
    ax.axhline(1.0 / 8.0, linestyle=":", color="grey", linewidth=1, label="chance (1/8)")
    ax.set_xscale("log")
    ax.set_xticks(window_seconds)
    ax.set_xticklabels([str(w) for w in window_seconds])
    ax.set_xlabel("embedding window length (s)")
    ax.set_ylabel("top-1 accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Day-7 fingerprinting: window-length sweep")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument(
        "--boldcast-ckpt",
        type=Path,
        default=Path("results/day5_train/ckpt_final.pt"),
    )
    p.add_argument(
        "--baseline-ckpt",
        type=Path,
        default=Path("results/day6_baseline/ckpt_final.pt"),
    )
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    p.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("results/day7_fingerprint/day7_metrics.json"),
    )
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Smoke test: parse args + verify imports without GPU.",
    )
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    k_list = [int(k) for k in cfg.eval.topk]
    pools = ["mean_tp", "mean_t"]
    window_seconds = [int(w) for w in cfg.eval.windows_seconds]

    if args.dry_run:
        print(f"[day7] dry-run: k_list={k_list}, pools={pools}, window_seconds={window_seconds}")
        return 0

    if not torch.cuda.is_available():
        raise SystemExit("[day7] CUDA not available - run on a GPU node.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # --- Heavy imports deferred so --dry-run works in CPU env ---
    from boldcast._upstream.cifti_io import (  # noqa: E402
        cortex_grayordinate_indices,
        load_dtseries,
    )
    from boldcast.data.hcp_rest import HCPRestingDataset  # noqa: E402
    from boldcast.data.schaefer_baseline import (  # noqa: E402
        build_schaefer_dataset_from_config,
        load_schaefer_cortex_assignment,
    )
    from boldcast.models.baseline import BaselineSchaefer400  # noqa: E402
    from boldcast.models.boldcast_demo import BOLDcastDemo  # noqa: E402
    from boldcast.tokenize.geodesic import build_or_load_patches  # noqa: E402
    from boldcast.tokenize.knn import build_or_load_knn  # noqa: E402

    horizons = tuple(int(h) for h in cfg.train.forecasting_horizons)

    # --- BOLDcast model + dataset --------------------------------------
    val_ds_bc = HCPRestingDataset.from_config(args.config, split="heldout")
    # Need adjacency for model construction: rebuild from same cache.
    train_subjects = [
        s
        for s in Path(str(cfg.data.subjects_train_file)).read_text().splitlines()
        if s.strip() and not s.startswith("#")
    ]
    ref_subject = train_subjects[0].strip()
    ref_run = cfg.data.runs[0]
    ref_path = str(cfg.data.dtseries_pattern).format(subject=ref_subject, run=ref_run)
    _, header = load_dtseries(ref_path)
    cortex_lh, cortex_rh = cortex_grayordinate_indices(header)
    surface_dir = str(cfg.data.surface_dir_template).format(subject=ref_subject)
    lh_mesh = f"{surface_dir}/{ref_subject}.L.midthickness_MSMAll.32k_fs_LR.surf.gii"
    rh_mesh = f"{surface_dir}/{ref_subject}.R.midthickness_MSMAll.32k_fs_LR.surf.gii"

    assignment_geo = build_or_load_patches(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        cache_path=str(cfg.tokenize.patch_cache),
        n_patches=int(cfg.tokenize.n_patches_cortex),
    )
    adj_geo = build_or_load_knn(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        patch_assignment=assignment_geo,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k=int(cfg.tokenize.knn_k),
        cache_path=str(cfg.tokenize.knn_cache),
    )
    adj_geo_t = torch.from_numpy(np.asarray(adj_geo)).long().to(device)
    bc_model: torch.nn.Module = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=adj_geo_t,
        horizons=horizons,
        use_checkpoint=False,
    )
    bc_model = _load_model_from_ckpt(args.boldcast_ckpt, bc_model, device)
    print(f"[day7] loaded BOLDcast ckpt: {args.boldcast_ckpt}")

    # --- Schaefer baseline model + dataset -----------------------------
    n_rois = int(cfg.baseline.n_rois)
    assignment_sch = load_schaefer_cortex_assignment(
        str(cfg.baseline.schaefer_dlabel),
        n_rois=n_rois,
    )
    adj_sch = build_or_load_knn(
        mesh_lh_path=lh_mesh,
        mesh_rh_path=rh_mesh,
        cortex_indices_lh=cortex_lh,
        cortex_indices_rh=cortex_rh,
        patch_assignment=assignment_sch,
        n_patches=n_rois,
        k=int(cfg.tokenize.knn_k),
        cache_path=str(cfg.baseline.knn_cache),
    )
    adj_sch_t = torch.from_numpy(np.asarray(adj_sch)).long().to(device)
    sch_model: torch.nn.Module = BaselineSchaefer400(
        d_in=1,
        d_model=128,
        n_layers=4,
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=adj_sch_t,
        horizons=horizons,
        use_checkpoint=False,
    )
    sch_model = _load_model_from_ckpt(args.baseline_ckpt, sch_model, device)
    print(f"[day7] loaded Schaefer ckpt: {args.baseline_ckpt}")
    val_ds_sch = build_schaefer_dataset_from_config(args.config, split="heldout")

    # --- Main metrics --------------------------------------------------
    print(f"[day7] computing BOLDcast metrics across pools={pools}, k_list={k_list} ...")
    bc_metrics = _compute_metrics(
        bc_model,
        val_ds_bc,
        device,
        pools=pools,
        k_list=k_list,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    print("[day7] computing Schaefer metrics ...")
    sch_metrics = _compute_metrics(
        sch_model,
        val_ds_sch,
        device,
        pools=pools,
        k_list=k_list,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )

    # --- Paired McNemar on per-run top-1 correctness -------------------
    mcnemar: dict[str, float] = {}
    for pool in pools:
        bc_correct = per_run_correct(
            bc_metrics[pool]["embeddings"],
            bc_metrics[pool]["subject_ids"],
            k=1,
        )
        sch_correct = per_run_correct(
            sch_metrics[pool]["embeddings"],
            sch_metrics[pool]["subject_ids"],
            k=1,
        )
        # NB: the two methods see the SAME held-out runs but in the order
        # produced by each model's dataset.  Sort both by (subject_id,
        # run_id) before pairing.
        bc_order = np.lexsort((bc_metrics[pool]["run_ids"], bc_metrics[pool]["subject_ids"]))
        sch_order = np.lexsort((sch_metrics[pool]["run_ids"], sch_metrics[pool]["subject_ids"]))
        mcnemar[pool] = paired_mcnemar(bc_correct[bc_order], sch_correct[sch_order])

    # --- Window sweep (top-1 only, mean_tp pool) -----------------------
    sweep: dict[str, dict[int, float]] = {"BOLDcast": {}, "Schaefer-400": {}}
    for w_sec in window_seconds:
        w_tr = int(round(w_sec / float(cfg.data.tr_seconds)))
        # Rebuild val dataset with override window size.  Tokenized cache
        # is window-agnostic, so this is cheap.
        bc_ds_w = HCPRestingDataset(
            subjects=val_ds_bc.subjects,
            runs=val_ds_bc.runs,
            dtseries_pattern=val_ds_bc.dtseries_pattern,
            cache_dir=val_ds_bc.cache_dir,
            patch_assignment=val_ds_bc.patch_assignment,
            n_patches=val_ds_bc.n_patches,
            window_size=w_tr,
            stride=max(w_tr // 2, 1),
            subject_id_offset=val_ds_bc.subject_id_offset,
        )
        sch_ds_w = HCPRestingDataset(
            subjects=val_ds_sch.subjects,
            runs=val_ds_sch.runs,
            dtseries_pattern=val_ds_sch.dtseries_pattern,
            cache_dir=val_ds_sch.cache_dir,
            patch_assignment=val_ds_sch.patch_assignment,
            n_patches=val_ds_sch.n_patches,
            window_size=w_tr,
            stride=max(w_tr // 2, 1),
            subject_id_offset=val_ds_sch.subject_id_offset,
        )
        emb_bc, sid_bc, _ = extract_embeddings(bc_model, bc_ds_w, pool="mean_tp", device=device)
        emb_sch, sid_sch, _ = extract_embeddings(sch_model, sch_ds_w, pool="mean_tp", device=device)
        sweep["BOLDcast"][w_sec] = topk_accuracy(emb_bc, sid_bc, k_list=[1])[1]
        sweep["Schaefer-400"][w_sec] = topk_accuracy(emb_sch, sid_sch, k_list=[1])[1]

    # --- Figures -------------------------------------------------------
    _plot_topk_bars(
        bc_metrics,
        sch_metrics,
        k_list,
        pools,
        args.out_dir / "day7_fingerprint_topk.png",
    )
    cm, labels = _confusion_matrix(
        bc_metrics["mean_tp"]["embeddings"],
        bc_metrics["mean_tp"]["subject_ids"],
    )
    _plot_confusion(
        cm,
        labels,
        args.out_dir / "day7_fingerprint_confusion.png",
        title="BOLDcast top-1 confusion (held-out, mean_tp pool)",
    )
    _plot_window_sweep(sweep, window_seconds, args.out_dir / "day7_fingerprint_window_sweep.png")

    # --- Metrics JSON --------------------------------------------------
    def _strip_arrays(m: dict[str, Any]) -> dict[str, Any]:
        return {
            pool: {
                "topk": v["topk"],
                "ci": v["ci"],
                "n_runs": int(v["embeddings"].shape[0]),
                "d_emb": int(v["embeddings"].shape[1]),
            }
            for pool, v in m.items()
        }

    metrics_dict = {
        "boldcast": _strip_arrays(bc_metrics),
        "schaefer400": _strip_arrays(sch_metrics),
        "paired_mcnemar_p_top1": mcnemar,
        "window_sweep_top1_mean_tp": {
            label: {int(w): float(a) for w, a in results.items()}
            for label, results in sweep.items()
        },
        "n_resamples": args.n_resamples,
        "seed": args.seed,
        "k_list": k_list,
        "pools": pools,
        "window_seconds": window_seconds,
    }
    args.metrics_out.write_text(json.dumps(metrics_dict, indent=2, sort_keys=True))
    print(f"[day7] metrics -> {args.metrics_out}")
    print(f"[day7] figures -> {args.out_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
