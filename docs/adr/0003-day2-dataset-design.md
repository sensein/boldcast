# 0003 — Day 2 Dataset and Dataloader Design

**Status:** Accepted
**Date:** 2026-05-08
**Supersedes:** N/A
**Related:** ADR 0002 (Day 1 tokenizer)

## Context

Day 2 of the 10-day demo plan calls for a PyTorch `Dataset` that yields
tokenized rsfMRI windows from HCP 7T REST runs. The Day 1 work shipped
the per-grayordinate-to-per-patch tokenizer (`Patcher`) plus a shared
patch-assignment cache. Day 2 wraps that in a per-subject-per-run
caching dataloader and exposes a map-style PyTorch interface for Day 4
(overfit) and Day 5 (DDP training).

Several decisions are not specified by the demo plan and are
locked here.

## Decisions

### D1. One shared patch assignment, built from a reference subject's mesh.

`fs_LR_32k` mesh topology (vertex count, face connectivity, cortex /
medial-wall grayordinate partition) is identical across HCP subjects;
only vertex *coordinates* differ slightly. We build the patch
assignment once via FPS+Lloyd on the first training subject's
midthickness MSMAll mesh and apply that same `(V_cortex,) -> (P,)`
mapping to every other subject. This biases patch boundaries toward
the reference subject's geometry by the small inter-subject coordinate
delta. Per-subject FPS and a Conte69 group-average template mesh are
tracked as Day-7+ follow-ups; the demo headline (subject
fingerprinting) does not require true group-level patch comparability.

### D2. Run-wise per-grayordinate Z-score standardization BEFORE the Patcher.

For each (subject, run) dtseries, each of the 59,412 cortex
grayordinates is centered (mean 0) and scaled (std 1) over the run's
TRs, with `eps=1e-8` against zero-variance channels. Patcher then
averages the standardized timeseries within each patch. This matches
`docs/methods.md` line 110 ("expected to drop to ~1e-5 once data is
run-wise standardized in the dataloader") and is the conventional fMRI
pre-tokenization step. Standardizing post-pooling instead would lose
per-grayordinate amplitude information across patch members.

### D3. Per-(subject, run) cache files with metadata-keyed invalidation.

Each cache file is `{cache_dir}/{subject}_{run}.npz`, written via
`numpy.savez_compressed`, holding `tokens: (T_full, P) float32` plus
metadata: `dtseries_sha`, `assignment_sha`, `n_patches`,
`standardize_method`. On read, mismatch raises `ValueError` (mirrors
Day 1's `build_or_load_patches`). `dtseries_sha` and `assignment_sha`
are first-16-hex of SHA-256 over the source bytes; this catches
re-preprocessed dtseries (e.g., a datalad re-pull picking up a
corrected file) and re-built patch assignments without silent reuse.

### D4. Map-style `Dataset`, not `IterableDataset`.

Day 5 DDP training uses `DistributedSampler` for deterministic
shuffling, which requires random access (`__getitem__(int)`). Window
enumeration is eager at `__init__`: a list of
`(subject_idx, run_idx, window_start)` triples. `len(dataset)` is the
length of that list.

### D5. Train and heldout share disjoint integer subject-ID ranges.

`subject_id` returned by `__getitem__` is `subjects.index(subject) +
id_offset`, where `id_offset` is `0` for train and `n_train` for
heldout. The model never sees the same subject int across splits, so
fingerprint evaluation cannot leak via integer collisions.

### D6. `from_config(config_path, split)` is the canonical constructor.

The bare `__init__` takes pre-resolved primitives (subject list, run
list, dtseries-pattern, cache-dir, patch-assignment array, n_patches).
Tests can instantiate without disk I/O for cortex indices or patch
assignments. `from_config` reads `configs/demo.yaml`, resolves the
patch-assignment cache via Day 1's `build_or_load_patches`, picks
cortex indices from the first dtseries header, and forwards primitives
into `__init__`.

## Consequences

- **Positive:** Tests run on synthetic data without touching the
  patcher cache. Real-data validation, which needs DUA access, is a single
  `from_config` call. Cache invalidation surface is explicit.
- **Negative:** D1 introduces a small per-subject geometry bias.
  Mitigated by tracking Conte69 template-mesh as a follow-up.
  D2 implies the cached `(T, P)` is patch-mean-of-standardized-
  grayordinates, **not** per-patch standardized — downstream loss
  computation should expect approximately-zero-mean but
  patch-size-dependent variance.
