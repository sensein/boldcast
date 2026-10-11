# Architecture Decision Records

Whenever an architectural decision is made (variational vs deterministic,
kNN attention vs spherical conv, frozen vs end-to-end CLIP), record it
here as `NNNN-short-title.md` using the template below.

## Template

```markdown
# NNNN. <decision>

**Status:** accepted | superseded by NNNN | deprecated
**Date:** YYYY-MM-DD

## Context
What problem is this decision solving?

## Decision
What was decided?

## Consequences
What does this mean for the project? Tradeoffs?

## Alternatives considered
What was rejected, and why?
```

## Index

- [0001 — Deterministic forecasting, not variational](0001-deterministic-not-variational.md)
- [0002 — Day-1 tokenizer implementation](0002-day1-tokenizer-implementation.md)
- [0003 — Day-2 dataset design](0003-day2-dataset-design.md)
- [0004 — Day-3 model architecture](0004-day3-model-architecture.md)
- [0005 — Day-4 training loop](0005-day4-training-loop.md)
- [0006 — Day-5 DDP](0006-day5-ddp.md)
