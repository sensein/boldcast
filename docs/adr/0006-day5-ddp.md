# 0006 — Day 5 DDP Multi-GPU Training

**Status:** Accepted
**Date:** 2026-05-13
**Supersedes:** N/A
**Related:** ADR 0005 (Day 4 training loop), `docs/methods.md` "Training Phases",
[`docs/superpowers/specs/at-the-same-time-resilient-clock.md`](../superpowers/specs/at-the-same-time-resilient-clock.md)

## Context

Day 5 of the 10-day demo plan scales Day-4's single-GPU `Trainer` to 2× H200
via `torchrun` + `torch.distributed.DataParallel` (DDP) and introduces held-out
validation loss tracking. The 10-day plan and ADR 0005 specify the hardware
target and hyperparameters, but leave nine critical distributed-training design
choices open. Two real risks shape this ADR:

1. **Silent epoch-shuffle freeze.** Day-4's `_infinite_loader` calls `iter(dataloader)`
   each cycle but does NOT call `sampler.set_epoch(epoch)`. With `DistributedSampler`
   this silently locks all ranks to the same window order every epoch — no crash,
   no NaN, just quietly reduced data diversity and early loss plateau.

2. **Non-rank-aware logging and checkpoint.** Current `print()`, JSONL writes,
   and checkpoint saves are unconditional. On DDP, all ranks racing the same file
   corrupts JSONL and creates disk contention on checkpoint.

This ADR locks the nine distributed design decisions (D1–D9) and adds one more
(D10) that guards against the silent-shuffle-freeze trap.

## Decisions

### D1. Use `torchrun --standalone --nproc-per-node=2 --nnodes=1`

Invoking training via `torchrun` (PyTorch's standard launcher) instead of
hand-rolled `torch.multiprocessing.spawn`:
- Automatically sets `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`,
  `MASTER_PORT` for all ranks.
- Single-node (`--nnodes=1`) with two ranks per node (`--nproc-per-node=2`).
- `--standalone` disables parent process tracking (OK for SLURM via sbatch).
- The launcher lives in `scripts/day5_train_boldcast.sh` as a `torchrun` invocation
  before the Python entry point.

### D2. NCCL on CUDA, Gloo fallback for CPU tests

`torch.distributed` backend selection is context-dependent:
- **Training (H200, CUDA):** use NCCL. Designed for GPU-to-GPU communication
  at datacenter scale; benchmarks show ≥95% efficiency on modern hardware
  (RTX, H100, H200).
- **Testing (CPU login node):** use Gloo. Pure CPU-based communication; allows
  testing DDP logic without a GPU. Tests spawn via `torch.multiprocessing.spawn`
  over Gloo + TCP.
- Both backends initialized via `init_process_group(backend, init_method="env://")`,
  which reads the env vars set by `torchrun`.

### D3. Preserve `_infinite_loader`, thread `sampler` through via optional kwarg

Day-4's `_infinite_loader` generator pattern (schist ID 227) stays unchanged
to avoid re-architecting the training loop on Day 5. Instead:
- Add a `sampler: Sampler | None = None` kwarg to `Trainer.fit()`.
- Modify `_infinite_loader` to call `sampler.set_epoch(epoch)` between cycles
  (before each `iter(dataloader)` call).
- When sampler is `None`, the hook is a no-op; existing Day-4 code paths work
  unchanged.
- On Day 5, wrap `DistributedSampler` and pass it to `fit()`.

### D4. Validation loop inside `Trainer.fit()`; rank-0-only measurement with broadcast

Validation introduces:
- `val_loader: DataLoader | None = None` and `val_every: int | None = None`
  kwargs to `Trainer.fit()`.
- A private `_eval(val_loader)` method that runs on rank 0 only (ranks > 0
  sleep at a `dist.barrier()`). Rank 0 computes mean loss over the entire
  validation set and broadcasts it to all ranks so all logging is consistent.
- Rank-0-only val avoids redundant computation and simplifies distributed
  reduction; the held-out set (~78 windows across 8 subjects) is small enough
  that broadcasting the scalar loss is negligible.
- `model.training` is explicitly restored to `True` after each validation call
  so the training loop sees the model in training mode.

### D5. All-reduce logged train loss on each step; average across ranks

Per-step training loss is synchronized across all ranks:
- After backward pass, call `loss_value = _maybe_all_reduce_mean(loss_value)`.
- This computes the average loss across all ranks using `dist.all_reduce(..., op=AVG)`.
- The averaged loss is logged (JSONL, stdout, wandb) on rank 0 only.
- When `torch.distributed` is not initialized, `_maybe_all_reduce_mean` is a no-op;
  Day-4 single-GPU mode is unaffected.

### D6. Rank-0 gates on all logging and checkpoint operations

Prevent file-write races:
- `print()` calls (line 164) wrapped in `if is_rank_zero(): print(...)`.
- `JsonlLogger.write()` only executes on rank 0; initialize the logger only on
  rank 0.
- `save_checkpoint()` only executes on rank 0.
- **Unwrap DDP wrapper on save:** When saving, extract the original model state
  dict via `model.module.state_dict()` (or `getattr(model, "module", model).state_dict()`
  for non-DDP-wrapped models). This ensures checkpoints are loadable as single-GPU
  models on Day 7 fingerprint evaluation, avoiding `DDP`-wrapper artifacts.

### D7. Separate `scripts/day5_bench_ddp_scaling.py` for scaling-efficiency measurement

DDP scaling efficiency (accepting criterion #4: ≥70% on 2× H200) is measured
separately:
- `scripts/day5_bench_ddp_scaling.py` is a one-shot benchmark, not part of the
  training run itself.
- It can be invoked with `--world-size 1` (single GPU) or `--world-size 2`
  (two GPUs via torchrun). Measures throughput in `tokens_per_second` over 50
  timed forward+backward steps (with 3+ warmup steps per schist ID 202).
- Efficiency = `throughput_2gpu / (2 × throughput_1gpu)`. Raises `SystemExit` if
  `< 0.70`.

### D8. Held-out validation acceptance: "mean(last_3) ≤ 0.7 × mean(first_3)"

Held-out loss must decrease by at least 30% from the start of training. Measured via
the `heldout_decreased_by(history, frac=0.30)` helper in `boldcast.training.utils`:
- `history` is a dict with `"val_loss"` list (one entry per `val_every` interval).
- Helper computes mean of the first 3 validation measurements and mean of the last 3.
- Returns `True` if `mean(last_3) ≤ 0.7 × mean(first_3)`, else raises `AssertionError`.
- The Day-5 training script asserts this at the end, blocking success if validation
  did not decrease sufficiently. This guards against a training run that appears
  to complete but learns nothing.

### D9. Per-rank seeding for sampler/augmentation; identical model init via `cfg.seed` only

Ensure sampler shuffles differ across ranks (necessary for DDP data coverage) while
model initialization is bit-identical:
- Sampler seed: `cfg.seed + rank`. Each rank's `DistributedSampler` gets a different
  seed, so shuffling order differs across ranks, guaranteeing each rank sees different
  data in each epoch.
- Augmentation seed (if any): `cfg.seed + rank`.
- **Model init seed:** `cfg.seed` only (applied via `seed_everything(cfg.seed)` before
  model construction). No rank offset. Both ranks initialize weights identically, so
  gradient averaging during `all_reduce` remains mathematically sensible.

### D10. The `set_epoch` trap guard

**Load-bearing correctness invariant.** The `_infinite_loader` MUST call
`sampler.set_epoch(epoch)` between successive `iter(dataloader)` calls when using
`DistributedSampler`. This is the only mechanism that guarantees rank-specific shuffle
orders across epochs. Without it, all ranks see the same window order every epoch —
not a crash, but silent data loss.

The trainer implementation embeds this as the primary guard. Tests verify that
`sampler.set_epoch` is called exactly once per epoch cycle (via a spy callback
in `tests/test_trainer_ddp.py`).

## Consequences

- **Positive:** DDP work is a ~150-line addition to the same `Trainer` from Day 4.
  `torchrun` avoids hand-rolled spawn logic. Rank-0-only validation + broadcast
  keeps distributed logic minimal. Separate scaling benchmark is a clean one-shot
  measurement. The silent-shuffle-freeze trap is explicitly guarded by name (D10).
  Tests for rank-awareness run on CPU via Gloo, keeping the test suite fast on
  login nodes.
- **Negative:** Day-5 training depends on Yibei to execute on ORCD compute with
  `torchrun` (HCP DUA; mamba-ssm). The Day-4 horizon-mismatch guard (ADR 0005
  lines 113–118) must remain byte-identical; any refactor that changes it breaks
  gradient flow silently. Validation is rank-0-only; if rank 0's held-out data
  distribution differs systematically from other ranks, the decision may be slightly
  optimistic. (This risk is negligible for a fixed held-out set split by subject,
  but worth noting for future work with run-level shuffled validation splits.)
