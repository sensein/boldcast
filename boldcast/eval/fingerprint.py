"""Frozen-backbone subject fingerprinting on held-out HCP runs.

Day-7 of the 10-day plan / Section X.3 of the K99 prelim. Implements
the leave-one-run-out retrieval protocol with two pooling variants
(``mean_tp`` and ``mean_t``), bootstrap CIs on top-k, and a paired
McNemar test between two models' per-run correctness vectors.

Conventions
-----------
* Embeddings are L2-normalized so the cosine similarity reduces to
  the dot product.
* Retrieval is **subject-level**: for each probe run, the per-subject
  gallery score is the mean similarity to all other runs of that
  subject; predictions are ranked over unique subjects.
* Bootstrap CIs resample SUBJECTS with replacement (not runs), to
  account for the within-subject correlation between a subject's
  multiple runs.

This module is GPU-agnostic: ``extract_embeddings`` accepts a device
arg; everything downstream is numpy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "bootstrap_ci_topk",
    "extract_embeddings",
    "paired_mcnemar",
    "per_run_correct",
    "topk_accuracy",
]


def extract_embeddings(
    model: nn.Module,
    dataset: Dataset[dict[str, torch.Tensor]],
    pool: Literal["mean_tp", "mean_t"] = "mean_tp",
    device: torch.device | str = "cuda",
    batch_size: int = 4,
    num_workers: int = 2,
) -> tuple[NDArray[np.float32], NDArray[np.int64], NDArray[np.int64]]:
    """Extract per-(subject, run) L2-normalized embeddings.

    Iterates the dataset window-by-window, calls ``model.embed(...)``,
    pools each window per the ``pool`` protocol, then averages all
    windows belonging to the same (subject_id, run_id) into a single
    per-run vector. Finally L2-normalizes.

    Parameters
    ----------
    model : nn.Module
        Must expose ``embed(tokens: (B, T, P, d_in)) -> (B, T, P, d_model)``.
        Set to ``model.eval()`` inside this function and restored on exit.
    dataset : Dataset
        Yields dicts with keys ``"tokens" (T, P)``, ``"subject_id" int``,
        ``"run_id" int``. Both ``HCPRestingDataset`` and the Schaefer
        variant satisfy this contract.
    pool : ``"mean_tp"`` or ``"mean_t"``
        * ``mean_tp`` — mean over T AND P; ``d_emb = d_model``.
        * ``mean_t`` — mean over T only, flatten P × d_model; ``d_emb = P * d_model``.
    device : torch.device or str
        Where to place the model + batches.
    batch_size, num_workers : int
        Passed to ``DataLoader``.

    Returns
    -------
    embeddings : ndarray of shape ``(N_runs, d_emb)`` float32, L2-normalized
    subject_ids : ndarray of shape ``(N_runs,)`` int64
    run_ids : ndarray of shape ``(N_runs,)`` int64
    """
    if pool not in ("mean_tp", "mean_t"):
        raise ValueError(f"pool must be 'mean_tp' or 'mean_t', got {pool!r}")

    was_training = model.training
    model.eval()
    try:
        loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        windows_by_run: dict[tuple[int, int], list[torch.Tensor]] = {}
        with torch.no_grad():
            for batch in loader:
                tokens = batch["tokens"].to(device).unsqueeze(-1)  # (B, T, P, 1)
                # nn.Module.__getattr__ types `.embed` as Tensor | Module;
                # the cast pins the return type, the ignore silences the
                # operator complaint at the call site itself.
                h = cast(
                    torch.Tensor,
                    model.embed(tokens),  # type: ignore[operator]
                )  # (B, T, P, d_model)
                if pool == "mean_tp":
                    emb_batch = h.mean(dim=(1, 2))  # (B, d_model)
                else:  # mean_t
                    h_pooled = h.mean(dim=1)  # (B, P, d_model)
                    emb_batch = h_pooled.reshape(h_pooled.shape[0], -1)  # (B, P*d_model)

                subj = batch["subject_id"].tolist()
                run = batch["run_id"].tolist()
                for i in range(emb_batch.shape[0]):
                    key = (int(subj[i]), int(run[i]))
                    windows_by_run.setdefault(key, []).append(emb_batch[i].detach().cpu().float())

        if not windows_by_run:
            raise ValueError("extract_embeddings: dataset is empty")

        keys_sorted = sorted(windows_by_run.keys())
        d_emb = windows_by_run[keys_sorted[0]][0].numel()
        embeddings_arr: NDArray[np.float32] = np.zeros((len(keys_sorted), d_emb), dtype=np.float32)
        subject_ids: NDArray[np.int64] = np.zeros(len(keys_sorted), dtype=np.int64)
        run_ids: NDArray[np.int64] = np.zeros(len(keys_sorted), dtype=np.int64)
        for i, key in enumerate(keys_sorted):
            stacked = torch.stack(windows_by_run[key], dim=0)  # (n_windows, d_emb)
            embeddings_arr[i] = stacked.mean(dim=0).numpy().astype(np.float32, copy=False)
            subject_ids[i] = key[0]
            run_ids[i] = key[1]

        norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
        embeddings_norm: NDArray[np.float32] = (
            embeddings_arr / np.maximum(norms, 1e-12)
        ).astype(np.float32, copy=False)
        return embeddings_norm, subject_ids, run_ids
    finally:
        model.train(mode=was_training)


def _per_run_predicted_rank(
    embeddings: NDArray[np.float32],
    subject_ids: NDArray[np.int64],
) -> NDArray[np.int64]:
    """For each row, return the rank position of its true subject in the
    predicted-subject ranking (0 = top hit).

    Predictions are formed by averaging per-subject gallery similarities
    (excluding the probe itself), then ranking unique subjects by
    descending mean similarity.
    """
    n = embeddings.shape[0]
    sim = embeddings @ embeddings.T
    np.fill_diagonal(sim, -np.inf)

    unique_subjects = np.unique(subject_ids)
    n_subjects = unique_subjects.shape[0]
    # Per-subject mean similarity for every probe.  Shape (n, n_subjects).
    per_subj = np.full((n, n_subjects), -np.inf, dtype=np.float64)
    for j, s in enumerate(unique_subjects):
        # Vectorize: gallery mask is (n,), True for runs of subject s.
        # For probes of subject s themselves, diagonal -inf already
        # excludes the self-similarity, so mean is well-defined as long
        # as the subject has >= 2 runs.
        gallery_mask = subject_ids == s
        # sim[probe, gallery]: ignore -inf entries (diag) when computing
        # the mean. Implementation: mask off the probe-self entry by
        # setting those sim values to NaN, then nanmean over the gallery.
        s_sim = sim[:, gallery_mask].copy()
        # Replace -inf with NaN so nanmean ignores it.
        s_sim[np.isinf(s_sim)] = np.nan
        with np.errstate(invalid="ignore"):
            per_subj[:, j] = np.nanmean(s_sim, axis=1)

    # Rank subjects descending by similarity per probe.
    order = np.argsort(-per_subj, axis=1, kind="stable")  # (n, n_subjects)
    ranked_subjects = unique_subjects[order]  # (n, n_subjects)
    # True-subject column index per probe.
    true_col = (subject_ids[:, None] == ranked_subjects).argmax(axis=1)
    return np.asarray(true_col, dtype=np.int64)


def topk_accuracy(
    embeddings: NDArray[np.float32],
    subject_ids: NDArray[np.int64],
    k_list: Sequence[int] = (1, 5, 10),
) -> dict[int, float]:
    """Leave-one-run-out top-k subject-retrieval accuracy.

    Returns
    -------
    dict mapping k -> accuracy in [0, 1].
    """
    ranks = _per_run_predicted_rank(embeddings, subject_ids)
    n = ranks.shape[0]
    return {int(k): float((ranks < k).sum()) / n for k in k_list}


def per_run_correct(
    embeddings: NDArray[np.float32],
    subject_ids: NDArray[np.int64],
    k: int = 1,
) -> NDArray[np.bool_]:
    """Return ``(N_runs,) bool``: True iff true subject is in top-k retrievals."""
    ranks = _per_run_predicted_rank(embeddings, subject_ids)
    return (ranks < k).astype(bool)


def bootstrap_ci_topk(
    embeddings: NDArray[np.float32],
    subject_ids: NDArray[np.int64],
    k: int = 1,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI on top-k accuracy by resampling SUBJECTS.

    Resampling subjects (not individual runs) propagates within-subject
    correlation correctly: if subject 17 happens to be over-represented
    in a resample, all four of its runs come with it.

    Returns
    -------
    (point_estimate, ci_low, ci_high)
    """
    if not (0 < ci < 1):
        raise ValueError(f"ci must be in (0, 1), got {ci}")
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)
    n_subj = unique_subjects.shape[0]

    point = topk_accuracy(embeddings, subject_ids, k_list=[k])[k]

    boot_accs = np.zeros(n_resamples, dtype=np.float64)
    for b in range(n_resamples):
        sampled = rng.choice(unique_subjects, size=n_subj, replace=True)
        # Rebuild embeddings + relabel subject IDs to the resampled index.
        # If the same subject is drawn twice, its runs appear twice with
        # DIFFERENT new subject IDs — that's the standard subject-bootstrap.
        chunks_emb: list[NDArray[np.float32]] = []
        chunks_ids: list[NDArray[np.int64]] = []
        for new_sid, old_sid in enumerate(sampled):
            mask = subject_ids == old_sid
            chunks_emb.append(embeddings[mask])
            chunks_ids.append(np.full(int(mask.sum()), new_sid, dtype=np.int64))
        boot_emb = np.concatenate(chunks_emb, axis=0)
        boot_ids = np.concatenate(chunks_ids, axis=0)
        boot_accs[b] = topk_accuracy(boot_emb, boot_ids, k_list=[k])[k]

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(boot_accs, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(point), float(lo), float(hi)


def paired_mcnemar(
    correct_a: NDArray[np.bool_],
    correct_b: NDArray[np.bool_],
) -> float:
    """Exact two-sided McNemar's test on paired binary correctness vectors.

    ``correct_a[i]`` and ``correct_b[i]`` are the two methods' results
    on the *same* probe (run) ``i``. Discordant pairs (one method right,
    the other wrong) carry the signal; concordant pairs are uninformative.

    Returns
    -------
    p : float
        Two-sided exact binomial p-value. Returns 1.0 when there are no
        discordant pairs (both methods are identical on this set).
    """
    if correct_a.shape != correct_b.shape:
        raise ValueError(
            f"shape mismatch: correct_a={correct_a.shape}, "
            f"correct_b={correct_b.shape}"
        )
    b_count = int((correct_a & ~correct_b).sum())
    c_count = int((~correct_a & correct_b).sum())
    n_disc = b_count + c_count
    if n_disc == 0:
        return 1.0
    # Exact binomial McNemar (two-sided).  scipy.stats.binomtest under
    # the null p=0.5 gives the standard exact form.
    from scipy.stats import binomtest

    result = binomtest(min(b_count, c_count), n_disc, p=0.5, alternative="two-sided")
    return float(result.pvalue)
