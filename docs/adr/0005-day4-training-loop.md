# 0005 — Day 4 Training Loop

**Status:** Accepted
**Date:** 2026-05-11
**Supersedes:** N/A
**Related:** ADR 0004 (Day 3 model), `docs/methods.md` "Training Objectives",
[`docs/superpowers/specs/2026-05-11-day4-training-loop-design.md`](../superpowers/specs/2026-05-11-day4-training-loop-design.md)

## Context

Day 4 of the 10-day demo plan wires the forecasting head, loss,
optimizer, and trainer for `BOLDcastDemo` and demonstrates that the
model overfits a tiny batch — proving the gradient path through the
multi-horizon head and MSE-over-horizons loss is correct before Day-5
DDP scale-up. The 10-day plan and ADR 0004 specify the shapes and
component identities but leave seven implementation choices open;
this ADR locks them.

## Decisions

### D1. Multi-horizon head ships on Day 4, not Day 5

ADR 0004 D1 explicitly defers the multi-horizon head from Day 3 to
Day 4. The Day-4 plan text in `docs/10_day_plan.md` line 322
("Single-step forecasting only on day 4 … Multi-step is day 5")
predates ADR 0004 and is stale. `configs/demo.yaml` already lists
`forecasting_horizons: [1, 5]`. Day 4 builds and exercises the
multi-horizon head + loss + targets. Same shape change either way;
catching a horizon-reshape bug now beats debugging it on Day 5 under
DDP.

### D2. Forward returns `(B, T, P, H, d_in)`; H axis always present

`BOLDcastDemo.forward` reshapes the head output to
`(B, T, P, H, d_in)` with H always materialized, including at H=1.
Shape-stable consumer contract is worth more than the dim-2 special
case. `embed()` is unchanged — Day-7 fingerprint eval keeps working.

### D3. Trainer is raw PyTorch, no Lightning

A ~150-line `Trainer` class handles forward, autocast, loss,
backward, grad clip, optimizer step, schedule step, and per-step
logging. Day-5 DDP adds `DistributedDataParallel` wrapping +
`DistributedSampler` + a `torchrun` launcher in ~10 lines on top of
this same class — no abstraction layer between the model and the
loop. Zero indirection beats five fewer lines on Day 5 for a 10-day
demo where debug time dominates.

### D4. Forecast targets + loss live in `boldcast/training/loss.py`

Two pure functions:

- `build_forecast_targets(tokens, horizons)`:
  `(B, T, P, d_in)` + `Sequence[int]` → `(B, T_valid, P, H, d_in)`
  with `T_valid = T - max(horizons)`.
- `forecasting_loss(pred, target)`: `F.mse_loss(pred, target)` with
  default `reduction='mean'`.

Mean reduction over `(B, T_valid, P, H, d_in)` is identically
equal-per-horizon weight when `T_valid` and `d_in` are uniform across
horizons (which they are by construction). No learnable per-horizon
scalar (`docs/methods.md`: "we explicitly do not learn a balancing
scalar"). Locating these in `loss.py` keeps the math unit-testable on
CPU without touching mamba-ssm or CUDA.

### D5. Overfit subset is 4 subjects × 1 run × 1 window = 4-window batch

Dataset construction: subjects = `cfg.data.subjects_train[:4]`,
runs = `[cfg.data.runs[0]]`, `window_size = cfg.window.size = 256`,
`stride = window_size + 10_000` (forces single window-start per run;
cardinality assertion `len(ds) == 4` at script entry). DataLoader
`batch_size=4`, `shuffle=False`, `num_workers=0`; one batch per
"epoch", a `_infinite_loader` generator provides an infinite stream of
the same batch to the trainer. Exercises the dataloader + collate +
model + loss path on the same shape Day-5 will see, not a synthetic
detour.

### D6. Day-4 script overrides `weight_decay=0` and `schedule="constant"`

The Trainer reads its hyperparameters from `cfg.train.*` so Day-4
exercises the same Trainer code Day-5 will run. The Day-4 script
overrides two fields inline before constructing the optimizer and
scheduler factories:

- `weight_decay=0.0` (overrides `cfg.train.weight_decay=0.05`).
  Regularization on an overfit subset adds noise to the gradient-
  flow signal.
- `schedule="constant"` (overrides `cfg.train.schedule="cosine"`).
  No warmup, no decay. Constant LR makes a loss flatline
  interpretable ("the optimizer isn't moving") rather than ambiguous
  ("is the LR zero from cosine decay or is the gradient zero?").

`grad_clip_norm` stays at `cfg.train.grad_clip_norm=1.0` — clipping
on a 4-window overfit shouldn't bite, and we want to exercise the
clip path.

### D7. Logging is stdout + JSONL on Day 4; wandb deferred to Day 5

Each step appends `{"step": int, "loss": float, "lr": float}` to
`{out_dir}/loss_log.jsonl`. Stdout prints every `log_every=10` steps.
A small plot helper (script-local in `scripts/day4_overfit.py`) reads
the JSONL and writes `figures/day4_overfit_curve.png` at end of run.
wandb setup is a pre-Day-5 task (depends on `wandb login` on the
ORCD compute node + API key in `~/.netrc`).

## Consequences

- **Positive:** Day-5 DDP work is a thin wrapper on the same Trainer.
  The Day-4 acceptance run exercises the same shapes Day-5 will see,
  not a synthetic detour. CPU dev gates (`.venv/bin/pytest -m "not
  gpu"`) cover loss, optim, utils, and a 0-layer Trainer end-to-end
  — meaningful coverage without a GPU runner.
- **Negative:** The Day-4 overfit run still depends on Yibei to
  execute on a GPU compute node under the micromamba env (HCP DUA;
  mamba-ssm). Plan-text in `docs/10_day_plan.md` line 322 is now
  stale; correcting it is a docs follow-up, not a code blocker.
