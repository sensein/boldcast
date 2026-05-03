# `_upstream/` — code earmarked for nobrainer PRs

Modules in this directory are destined to be contributed upstream to
[nobrainer](https://github.com/neuronets/nobrainer). They are held to
nobrainer-acceptable standards even while they live in this repo:

- **No imports from any other `boldcast/` submodule.** This code must be
  importable in isolation.
- **No project-specific magic constants.** Everything configurable is a
  function/class parameter with a clear default.
- **Full type hints** on every public function and class.
- **NumPy-style docstrings** on every module and public symbol.
- **Tested in isolation** under `tests/_upstream/`.

When a module is accepted into nobrainer:

1. Delete it from `boldcast/_upstream/`.
2. Add `nobrainer >= X.Y` to `pyproject.toml`.
3. Update the project-side wrapper in `boldcast/io/`, `boldcast/tokenize/`,
   etc. to re-export from `nobrainer`.

Per `docs/methods.md`, the planned upstream contributions are:

- **CIFTI I/O** — `boldcast/_upstream/cifti_io.py` → `nobrainer.io.cifti`
- **Surface dataset** — `boldcast/_upstream/surface_dataset.py` → `nobrainer.dataset`
- **Geodesic patcher** — `boldcast/_upstream/geodesic_patcher.py` → `nobrainer.layers`

Anything else in this repo that turns out to be reusable can be promoted
into `_upstream/` later — but only after it works in `boldcast/` first.
