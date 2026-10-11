"""Day-7 BOLDcast-only fingerprinting (interim: Schaefer baseline TBD).

Computes leave-one-run-out subject retrieval on the held-out 8 subjects
using BOLDcast's frozen ``ckpt_final.pt``.  Reports:

- top-1 / top-5 / top-10 accuracy for both pooling protocols
  (``mean_tp`` = mean over T and P; ``mean_t`` = mean over T, flatten P×d_model)
- Clopper-Pearson exact binomial 95% CI on each top-k (see
  :func:`boldcast.eval.fingerprint.binomial_ci_topk`)

We use the binomial CI rather than the subject-resample bootstrap because
the n=8 held-out cohort + strong subject-discriminative embeddings hit
the documented bootstrap cluster-collapse regime: resampled duplicates
become indistinguishable in retrieval and the CI collapses below the
point estimate. The binomial CI sidesteps that by inverting the binomial
distribution around the observed per-probe correctness counts (n=32
trials = 8 subjects × 4 runs).

K99 headline: top-1 with binomial CI on the ``mean_tp`` pool. Day-7 full
eval (``scripts/day7_fingerprint_eval.py``) supersedes this once the
Schaefer Day-6 ckpt lands.

Output is sanitized: scalar metrics only, no subject IDs in stdout.

Run::

    sbatch scripts/day7_fingerprint_boldcast.sh
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from boldcast.eval.fingerprint import (  # noqa: E402
    binomial_ci_topk,
    extract_embeddings,
    topk_accuracy,
)
from boldcast.utils.env import load_repo_dotenv  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

load_repo_dotenv(_REPO_ROOT)


def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument(
        "--ckpt",
        type=Path,
        default=Path("results/day5_train/ckpt_final.pt"),
        help="Path to BOLDcast trained checkpoint.",
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/day7_fingerprint/boldcast_metrics.json"),
    )
    p.add_argument(
        "--ci",
        type=float,
        default=0.95,
        help="Confidence level for the Clopper-Pearson binomial CI.",
    )
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    if not torch.cuda.is_available():
        raise SystemExit("[day7-bc] CUDA not available — run on a GPU node.")
    device = torch.device("cuda")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    # Heavy imports deferred so --help works in CPU env.
    from boldcast._upstream.cifti_io import (
        cortex_grayordinate_indices,
        load_dtseries,
    )
    from boldcast.data.hcp_rest import HCPRestingDataset
    from boldcast.models.boldcast_demo import BOLDcastDemo
    from boldcast.tokenize.geodesic import build_or_load_patches
    from boldcast.tokenize.knn import build_or_load_knn

    horizons = tuple(int(h) for h in cfg.train.forecasting_horizons)

    train_subjects = [
        s.strip()
        for s in Path(str(cfg.data.subjects_train_file)).read_text().splitlines()
        if s.strip() and not s.startswith("#")
    ]
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

    val_ds = HCPRestingDataset.from_config(args.config, split="heldout")
    print(
        f"[day7-bc] heldout dataset: {len(val_ds.subjects)} subjects × "
        f"{len(val_ds.runs)} runs = {len(val_ds.subjects) * len(val_ds.runs)} expected runs"
    )

    model = BOLDcastDemo(
        d_in=1,
        d_model=128,
        n_layers=4,
        n_patches=int(cfg.tokenize.n_patches_cortex),
        k_neighbors=int(cfg.tokenize.knn_k),
        adjacency=adjacency,
        horizons=horizons,
        use_checkpoint=False,  # eval-only
    ).to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(
        f"[day7-bc] loaded ckpt from {args.ckpt} "
        f"(step={ckpt.get('step', '?') if isinstance(ckpt, dict) else '?'}); "
        f"params={sum(p.numel() for p in model.parameters()) / 1e6:.3f}M"
    )

    pools: list[Literal["mean_tp", "mean_t"]] = ["mean_tp", "mean_t"]
    k_list = [int(k) for k in cfg.eval.topk]

    results: dict[str, dict[str, Any]] = {}
    last_sids: np.ndarray | None = None
    for pool in pools:
        print(f"[day7-bc] extracting embeddings: pool={pool}")
        emb, sids, _rids = extract_embeddings(model, val_ds, pool=pool, device=device)
        last_sids = sids
        n_runs, d_emb = int(emb.shape[0]), int(emb.shape[1])
        print(f"[day7-bc]   n_runs={n_runs}, d_emb={d_emb}")

        topk = topk_accuracy(emb, sids, k_list=k_list)
        cis: dict[int, dict[str, float]] = {}
        for k in k_list:
            point, lo, hi = binomial_ci_topk(emb, sids, k=k, ci=args.ci)
            cis[int(k)] = {"point": point, "ci_low": lo, "ci_high": hi}
            print(
                f"[day7-bc]   top-{k}: {point:.4f}  {int(args.ci * 100)}% CI=[{lo:.4f}, {hi:.4f}]"
            )
        results[pool] = {
            "topk": {int(k): float(v) for k, v in topk.items()},
            "ci": cis,
            "n_runs": n_runs,
            "d_emb": d_emb,
        }

    assert last_sids is not None
    n_heldout_subjects = int(np.unique(last_sids).size)
    summary: dict[str, Any] = {
        "ckpt": str(args.ckpt),
        "config": args.config,
        "n_heldout_subjects": n_heldout_subjects,
        "ci_method": "clopper_pearson",
        "ci_level": args.ci,
        "k_list": k_list,
        "pools": list(pools),
        "results": results,
    }
    args.out_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[day7-bc] wrote {args.out_json}")

    # K99 headline line
    chance = 1.0 / float(n_heldout_subjects)
    headline_topk = results["mean_tp"]["topk"][1]
    headline_ci = results["mean_tp"]["ci"][1]
    pct = int(args.ci * 100)
    print(
        f"[day7-bc] HEADLINE: BOLDcast top-1 (mean_tp pool) = "
        f"{headline_topk:.3f} [{pct}% CP CI: "
        f"{headline_ci['ci_low']:.3f}, "
        f"{headline_ci['ci_high']:.3f}]; "
        f"chance = {chance:.3f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
