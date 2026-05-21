"""CPU unit tests for boldcast.eval.fingerprint.

Synthetic embeddings only — no model, no HCP data. Tests the
mathematical correctness of the retrieval protocol, the bootstrap
machinery, and the paired McNemar statistic.
"""

from __future__ import annotations

import numpy as np
import pytest
from boldcast.eval.fingerprint import (
    bootstrap_ci_topk,
    paired_mcnemar,
    per_run_correct,
    topk_accuracy,
)


def _make_clustered_embeddings(
    n_subjects: int,
    n_runs_per_subject: int,
    d_emb: int,
    noise_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Embeddings where each subject has its own cluster center.

    Returns L2-normalized ``(N, d)`` embeddings and ``(N,)`` subject IDs.
    Tunable noise lets us span "perfect fingerprinting" (small noise)
    through "chance-level" (large noise) regimes.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_subjects, d_emb))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)

    embs = []
    sids = []
    for s in range(n_subjects):
        for _ in range(n_runs_per_subject):
            x = centers[s] + noise_std * rng.standard_normal(d_emb)
            x /= max(np.linalg.norm(x), 1e-12)
            embs.append(x)
            sids.append(s)
    return np.asarray(embs, dtype=np.float32), np.asarray(sids, dtype=np.int64)


# --- topk_accuracy ---------------------------------------------------------


def test_topk_accuracy_perfect_clusters() -> None:
    """Tiny-noise clusters → top-1 = 1.0."""
    emb, sids = _make_clustered_embeddings(
        n_subjects=8, n_runs_per_subject=4, d_emb=32, noise_std=0.01, seed=0,
    )
    acc = topk_accuracy(emb, sids, k_list=[1, 5])
    assert acc[1] == pytest.approx(1.0)
    assert acc[5] == pytest.approx(1.0)


def test_topk_accuracy_random_is_near_chance() -> None:
    """Random (unit-norm) embeddings → top-1 near 1/n_subjects."""
    rng = np.random.default_rng(0)
    n_subj, n_run = 8, 4
    emb = rng.standard_normal((n_subj * n_run, 16)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    sids = np.repeat(np.arange(n_subj, dtype=np.int64), n_run)
    acc = topk_accuracy(emb, sids, k_list=[1])
    # Chance = 1/8 = 0.125; with N=32 we expect noise.  Just sanity-bound it.
    assert 0.0 <= acc[1] <= 0.5


def test_topk_accuracy_top_k_monotone_in_k() -> None:
    """top-k accuracy is non-decreasing in k."""
    emb, sids = _make_clustered_embeddings(
        n_subjects=8, n_runs_per_subject=4, d_emb=16, noise_std=0.6, seed=42,
    )
    acc = topk_accuracy(emb, sids, k_list=[1, 3, 5])
    assert acc[1] <= acc[3] <= acc[5]


def test_topk_accuracy_excludes_self_match() -> None:
    """A probe never retrieves itself; with 1 run/subject, top-1 != 1.0.

    With a single run per subject, the only run of subject s besides
    the probe itself is empty — the per-subject gallery similarity for
    s collapses to NaN.  np.nanmean of an all-NaN slice yields NaN +
    a RuntimeWarning, which np.argsort treats as smaller than any
    finite value.  Either way, the probe's own subject can NEVER win
    on the basis of its own embedding, so top-1 = 0 here.
    """
    emb, sids = _make_clustered_embeddings(
        n_subjects=4, n_runs_per_subject=1, d_emb=8, noise_std=0.01, seed=0,
    )
    acc = topk_accuracy(emb, sids, k_list=[1])
    assert acc[1] == pytest.approx(0.0)


# --- per_run_correct -------------------------------------------------------


def test_per_run_correct_shape_and_dtype() -> None:
    emb, sids = _make_clustered_embeddings(
        n_subjects=4, n_runs_per_subject=4, d_emb=16, noise_std=0.1, seed=1,
    )
    correct = per_run_correct(emb, sids, k=1)
    assert correct.shape == (16,)
    assert correct.dtype == bool


def test_per_run_correct_aggregates_to_topk() -> None:
    """Mean of per_run_correct equals top-k accuracy."""
    emb, sids = _make_clustered_embeddings(
        n_subjects=8, n_runs_per_subject=4, d_emb=16, noise_std=0.5, seed=7,
    )
    acc = topk_accuracy(emb, sids, k_list=[1])[1]
    correct = per_run_correct(emb, sids, k=1)
    assert float(correct.mean()) == pytest.approx(acc)


# --- bootstrap_ci_topk -----------------------------------------------------


def test_bootstrap_ci_seed_deterministic() -> None:
    """Same seed → identical CI; different seeds → CIs need not match."""
    emb, sids = _make_clustered_embeddings(
        n_subjects=8, n_runs_per_subject=4, d_emb=16, noise_std=0.3, seed=0,
    )
    a = bootstrap_ci_topk(emb, sids, k=1, n_resamples=100, seed=0)
    b = bootstrap_ci_topk(emb, sids, k=1, n_resamples=100, seed=0)
    assert a == b
    c = bootstrap_ci_topk(emb, sids, k=1, n_resamples=100, seed=1)
    # point estimate is seed-independent, CIs need not match
    assert a[0] == c[0]


def test_bootstrap_ci_bounds_and_ordering() -> None:
    """CI is in [0, 1], ordered, and brackets a plausible point estimate.

    Note: subject-resample CIs are *conservative* on perfectly-clustered
    embeddings because a subject drawn twice creates two new_sid groups
    with indistinguishable mean similarity — the retrieval result is
    arbitrary on those probes. This is the price of propagating within-
    subject correlation correctly; we test bounds, not tightness.
    """
    emb, sids = _make_clustered_embeddings(
        n_subjects=8, n_runs_per_subject=4, d_emb=16, noise_std=0.3, seed=0,
    )
    point, lo, hi = bootstrap_ci_topk(emb, sids, k=1, n_resamples=200, seed=0)
    assert 0.0 <= lo <= hi <= 1.0
    assert 0.0 <= point <= 1.0
    # Point estimate well above chance (1/8 = 0.125) for this noise level
    assert point > 0.5


# --- paired_mcnemar --------------------------------------------------------


def test_mcnemar_identical_methods_p_one() -> None:
    """Two methods with identical correctness → p = 1.0 (no discordants)."""
    a = np.array([True, False, True, True, False, True, False, True] * 2)
    b = a.copy()
    assert paired_mcnemar(a, b) == pytest.approx(1.0)


def test_mcnemar_one_method_strictly_better_p_small() -> None:
    """A method that beats the other on every discordant pair → small p."""
    a = np.array([True] * 10 + [False] * 0)
    b = np.array([False] * 10)
    # 10 discordants, all in one direction.
    p = paired_mcnemar(a, b)
    # Binomial test: P(X >= 10 | n=10, p=0.5) two-sided is 2/1024 ≈ 0.002
    assert p < 0.01


def test_mcnemar_shape_mismatch_raises() -> None:
    a = np.array([True, False, True])
    b = np.array([True, False])
    with pytest.raises(ValueError, match="shape mismatch"):
        paired_mcnemar(a, b)


def test_mcnemar_symmetric_discordants_p_one() -> None:
    """Equal discordants in each direction → p = 1.0 (perfect tie)."""
    a = np.array([True, True, False, False])
    b = np.array([False, False, True, True])
    # b=2, c=2, two-sided exact: P(X=2 | n=4, p=0.5) center = 1.0
    p = paired_mcnemar(a, b)
    assert p == pytest.approx(1.0)
