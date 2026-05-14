# BOLDcast docs

Atlas-free, surface-based hybrid-Mamba foundation model for joint
stimulus–brain latent state tracking from naturalistic fMRI.

## Quickstart

- [Methods](methods.md) — architecture spec, training objectives, eval protocols
- [Architecture](architecture.md) — model details
- [10-day plan](10_day_plan.md) — demo deliverable plan
- [Data preparation](data_preparation.md) — HCP / CNeuroMod loaders + caching
- [ORCD benchmarks](orcd_benchmarks.md) — Day-3 measurements on H200

## Architecture Decision Records

The [ADR series](adr/README.md) records implementation decisions and their
rationale.

| ADR | Topic |
|-----|-------|
| [0001](adr/0001-deterministic-not-variational.md) | Deterministic, not variational |
| [0002](adr/0002-day1-tokenizer-implementation.md) | Day-1 tokenizer implementation |
| [0003](adr/0003-day2-dataset-design.md) | Day-2 dataset design |
| [0004](adr/0004-day3-model-architecture.md) | Day-3 model architecture |
| [0005](adr/0005-day4-training-loop.md) | Day-4 training loop |
| [0006](adr/0006-day5-ddp.md) | Day-5 DDP |

## Talks

The seed-grant pitch deck and future talks live at [../talks/](../talks/).

## Source

- GitHub: <https://github.com/sensein/boldcast>
- License: Apache 2.0
- Authors: Yibei Chen &amp; Satra Ghosh · MIT Senseable Intelligence Group
