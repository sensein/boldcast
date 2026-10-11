# `experiments/` — exploratory work

Code here is not part of the `boldcast` package and is not held to the same
standards. Use it for:

- One-off scripts, sketches, or analyses that aren't ready to be tested.
- Sanity checks during development that won't be re-run as part of the demo
  or paper.
- Anything you'd otherwise be tempted to leave in a notebook.

When a piece of code becomes a deliverable (produces a paper figure, gets
re-used by another script, or you find yourself re-running it), promote it:

1. Move it into the appropriate `boldcast/` submodule.
2. Add a test in `tests/`.
3. Add a `scripts/` entry point if it's user-facing.

Nothing in `experiments/` should be imported from `boldcast/`.
