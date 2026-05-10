# BOLDcast

Atlas-free, surface-based hybrid-Mamba foundation model for joint
stimulus–brain latent state tracking from naturalistic fMRI.

## Development setup

The project uses **two Python environments by design**:

- **uv-managed `.venv`** — for dev gates (`pytest`, `mypy`, `ruff`).
  Fast, no env activation, runs from any worktree.
- **micromamba env** — for training and any code path that imports
  `mamba-ssm` or `causal-conv1d` (Day 3+). These wheels need a CUDA
  toolchain that uv can't supply on a CPU-only login node.

### One-time per checkout (or per worktree)

```bash
# In the worktree root (or repo root)
uv venv .venv --python 3.11
uv pip install \
    "torch>=2.1" "nibabel>=5.0" "nilearn>=0.10" \
    "numpy>=1.26" "scipy>=1.12" "scikit-learn>=1.4" \
    "trimesh>=4.0" "omegaconf>=2.3" "hydra-core>=1.3" \
    "transformers>=4.40" "open-clip-torch>=2.24" \
    "wandb>=0.16" "matplotlib>=3.8" "pandas>=2.2" \
    "pytest>=8.0" "pytest-cov>=5.0" "ruff>=0.4" "mypy>=1.9" \
    "pyyaml" --python .venv/bin/python
uv pip install -e . --no-deps --python .venv/bin/python
```

The dep list above is the Day 1 + Day 2 + downstream-eval surface;
`mamba-ssm` and `causal-conv1d` are intentionally omitted (Day 3+
training uses the micromamba env instead).

### Daily dev gates

```bash
.venv/bin/pytest tests/
.venv/bin/mypy --strict boldcast/
.venv/bin/ruff check boldcast/ tests/ scripts/
```

No env activation. mypy comments use
`# type: ignore[attr-defined,unused-ignore]` so the same source passes
mypy under both nibabel-stub generations (older in micromamba, newer
in uv).

### Training & GPU runtime (Day 5+)

```bash
micromamba activate $BOLDCAST_ENV
python scripts/day5_train_boldcast.py --config configs/demo.yaml
```

The micromamba env at `$BOLDCAST_ENV`
is the authoritative runtime on ORCD; `pyproject.toml` `dependencies`
are kept in sync with `environment.yml`.

## License

Apache 2.0 — see `LICENSE`.
