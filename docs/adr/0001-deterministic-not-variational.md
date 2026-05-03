# 0001. Deterministic forecasting, not variational

**Status:** accepted
**Date:** 2026-05-03

## Context

Brain-state modeling work occasionally proposes variational latent-state
formulations (e.g., variational dynamical models, VAE-style encoders for
fMRI) on the grounds that brain states are "uncertain" and uncertainty
should be modeled explicitly.

We need to choose: do we train BOLDcast as a deterministic forecaster
(`p(x_{t+k} | x_{≤t}, s)` parameterized as a point estimate), or do we add
variational latent variables?

## Decision

BOLDcast is a **deterministic** multi-step forecaster. Loss is MSE on the
predicted next-TR token activations. No variational lower bound, no
explicit per-step latent variable, no learned posterior.

Uncertainty estimates, if needed for downstream consumers, are obtained
post-hoc via temperature-calibrated ensembling.

## Consequences

- Training is simpler: one objective (MSE forecasting) plus the InfoNCE
  contrastive head in phase 2. No KL term to balance, no posterior
  collapse failure mode, no reparameterization tricks to debug.
- Reproducibility is higher: deterministic forward pass in eval mode given
  the same input.
- We forfeit the ability to interpret latent activations as *probabilistic*
  brain states. We treat learned representations as features whose
  usefulness is judged on downstream task performance, not as a
  probabilistic state estimator.
- Memory and compute footprint is lower (no encoder–decoder split, no
  sampling at training time).

## Alternatives considered

- **Full variational latent dynamics** (rSLDS-style or VAE-style): rejected
  because (a) the literature is unclear on whether brain-state
  uncertainty estimates from these models are well-calibrated, and (b)
  the additional complexity is not justified by the evaluations we plan
  (forecasting MSE, retrieval accuracy, downstream linear probing).
- **Deterministic forecaster with MC-dropout uncertainty:** considered as
  a fallback for downstream tasks that need uncertainty. Cheap to add
  later if needed; not part of the core training objective.
- **Diffusion forecaster:** out of scope for the seed-grant timeline. May
  be revisited if forecasting quality is the bottleneck for downstream
  tasks.

## References

- See `docs/methods.md` "Scope and Honest Limitations" for the public
  framing of this decision.
