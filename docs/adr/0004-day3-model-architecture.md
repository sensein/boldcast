# 0004 — Day 3 BOLDcast-Demo Model Architecture

**Status:** Accepted
**Date:** 2026-05-10
**Supersedes:** N/A
**Related:** ADR 0002 (Day 1 tokenizer), ADR 0003 (Day 2 dataset)

## Context

Day 3 of the 10-day demo plan builds the minimal BOLDcast backbone: a
4-layer interleaved Mamba + kNN-attention stack at d_model=128 over
P=1024 cortical patches, ~0.7M params. The 10-day plan and methods.md
together specify shapes and component identities but leave seven
implementation choices open. This ADR locks them.

## Decisions

### D1. Day 3 head is single-horizon.

methods.md describes a multi-head regression emitting
`{ŷ_{t+1}, ŷ_{t+5}, ŷ_{t+10}}` in parallel; the demo configuration
uses `{1, 5}`. The 10-day plan's Day 3 forward signature is
`(B, T, P, d_in) → (B, T, P, d_in)` — single horizon. We treat the
multi-horizon head as a Day 4 (training loop) concern, not a Day 3
(backbone) concern. The Day-3 head is a single `nn.Linear(d_model,
d_in)`; Day 4 swaps it for `nn.Linear(d_model, H * d_in)` plus a
reshape, with no backbone change.

### D2. KNNAttention is single-head.

Multi-head attention adds params (one Q/K/V/O linear set per head)
without changing the inductive bias for `k=8` neighbors. Single-head
fits the demo param budget (~66k per attention block × 4 layers =
~265k of the ~680k total) and keeps the math auditable. We revisit if
held-out forecasting saturates and ablations point at spatial mixing
as the bottleneck.

### D3. kNN includes self.

`k=8 = 1 self-link + 7 spatial neighbors`. The self-link ensures the
attention output for token `i` includes its own value contribution,
making the residual `x + KNNAttention(x)` a strict refinement rather
than a substitution. Without self in the neighbor set, a token's
attention output is purely a function of its neighbors' values, which
behaves like a low-pass spatial filter at every layer — fine in
principle but a hidden inductive bias we'd rather avoid.

### D4. MambaBlock is a thin pre-LN wrapper.

`MambaBlock(x) = x + Mamba(LayerNorm(x))`. No FFN, no second
LayerNorm, no dropout. The methods.md description ("stack of causal
Mamba blocks") supports the minimal form; adding FFN would inflate
params past the ~0.7M demo budget and is recoverable in the scaled
config. Pre-LN matches the recommended placement for SSM stacks
[Jamba; Hymba].

### D5. Embed and head are patch-shared Linear layers.

A single `nn.Linear(d_in=1, d_model=128)` applied per-token, broadcast
across `(B, T, P)`. Similarly for the head: `nn.Linear(d_model=128,
d_in=1)` shared across patches. No per-patch parameters. Two reasons:
(a) the atlas-free framing treats patches as interchangeable tokens,
so per-patch weights would inject an arbitrary spatial prior; (b) it
keeps the param count linear in `d_model`, not `P × d_model`.

### D6. kNN precompute lives in `boldcast/tokenize/knn.py`.

Same module-layout convention as `boldcast/tokenize/geodesic.py`: a
project-side wrapper around a pure-numpy adjacency builder. Returns
`(P, k) int` indices; metadata key on the cache is
`{patch_assignment_sha, k, n_patches}` so a re-built patch assignment
invalidates the kNN cache automatically (mirrors Day 1 / Day 2's
mismatch-raises pattern).

### D7. mamba-ssm tests carry `@pytest.mark.gpu`.

`mamba-ssm` and `causal-conv1d` ship CUDA kernels that don't build on
a CPU-only login node and don't run on CPU at inference time. We mark
every test that imports `mamba_ssm` with `@pytest.mark.gpu`. Default
`pytest` invocations on the login node (uv `.venv`) skip them via a
`-m "not gpu"` collector override in `pyproject.toml`; explicit
`pytest -m gpu` on a GPU compute node (micromamba env) runs them.

## Consequences

- **Positive:** Param count is predictable (~680k, comfortably inside
  `[0.5e6, 1.5e6]`). Day 4 head swap is a one-line change. Daily uv
  dev gates remain CUDA-free.
- **Negative:** Multi-head attention and FFN are deferred — if held-
  out forecasting underperforms, those are obvious recovery levers
  but require revisiting this ADR. The `@pytest.mark.gpu` marker
  means CI on the login node provides only partial coverage for the
  model layer; the GPU test path is run-by-Yibei rather than
  CI-enforced.
