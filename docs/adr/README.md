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
