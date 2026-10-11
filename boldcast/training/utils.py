"""Training utilities: seed setup, checkpoint I/O, JSONL logger.

All CPU-runnable. Used by the Day-4 Trainer and the Day-4 overfit script.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO

import numpy as np
import torch
from torch import nn

__all__ = [
    "JsonlLogger",
    "beats_best_baseline",
    "save_checkpoint",
    "seed_everything",
]

_LOGGER = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs. CUDA seed is set if available.

    Note: ``mamba-ssm``'s selective-scan CUDA kernel is not bit-deterministic
    on GPU (methods.md "Reproducibility caveats"); this seeds the PyTorch
    initializer state but does not produce bit-identical training runs
    across hardware. Logs ``CUBLAS_WORKSPACE_CONFIG`` at DEBUG level for
    diagnosability when reproducibility breaks on a CUDA host.

    Parameters
    ----------
    seed : int
        Integer seed to apply to all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  — global seed required for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    _LOGGER.debug(
        "CUBLAS_WORKSPACE_CONFIG=%s",
        os.environ.get("CUBLAS_WORKSPACE_CONFIG", "<not set>"),
    )


def beats_best_baseline(
    model_val_loss: float,
    baselines: dict[str, float],
    frac: float = 0.15,
) -> bool:
    """Return True iff the model beats the strongest trivial baseline by ``frac``.

    Specifically: returns ``model_val_loss <= (1 - frac) * min(baselines.values())``.

    The metric ``(best_baseline - model_val_loss) / best_baseline`` is the
    fraction of the baseline's residual variance the model explains —
    i.e., R² against the trivial baseline as the null model. Two
    independent literatures anchor the default threshold ``frac=0.15``:

    1. Cohen's effect-size conventions for variance explained:
       small=0.01, medium=0.06, **large=0.14**. A 15% improvement sits
       just above Cohen's "large effect" threshold.
    2. Neuroimaging encoding/decoding norms: fMRI encoding models
       typically report out-of-sample R² in the 0.05-0.20 range;
       R² = 0.10 is considered a strong effect. 15% places the gate in
       the upper-middle of this range.

    See ``docs/superpowers/specs/2026-05-24-acceptance-gate-baseline-relative-design.md``
    (internal design note, not published) for the full rationale and
    migration history.

    Parameters
    ----------
    model_val_loss
        The trained model's mean val MSE on the same loader the
        baselines are computed against.
    baselines
        Mapping from baseline name to mean val MSE. The gate compares
        against ``min(baselines.values())`` so the strongest baseline
        sets the bar.
    frac
        Required improvement fraction over the strongest baseline.
        Default 0.15 (15%, ≈ Cohen's large-effect R²).

    Returns
    -------
    bool
        True iff ``model_val_loss <= (1 - frac) * min(baselines.values())``.

    Raises
    ------
    ValueError
        If ``baselines`` is empty.
    """
    if not baselines:
        raise ValueError("baselines must be non-empty")
    best = min(baselines.values())
    return model_val_loss <= (1.0 - frac) * best


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    path: Path,
) -> None:
    """Save model + optimizer state dicts and the step counter to ``path``.

    Creates parent directories as needed. Uses ``torch.save`` defaults
    (pickled state dict, no weights-only restriction).

    Parameters
    ----------
    model : nn.Module
        The model whose ``state_dict`` will be saved.
    optimizer : torch.optim.Optimizer
        The optimizer whose ``state_dict`` will be saved.
    step : int
        Current training step counter, stored under key ``"step"``.
    path : Path
        Destination file path. Parent directories are created if absent.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )


class JsonlLogger:
    """Append-only JSON-Lines logger. One row per ``write`` call.

    Opens the file in append mode so multiple Trainer.fit calls into the
    same path concatenate cleanly.

    Parameters
    ----------
    path : Path
        Destination file path. Parent directories are created if absent.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = open(self.path, "a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Append ``record`` as a single JSON line, then flush.

        Parameters
        ----------
        record : dict[str, Any]
            Mapping of metric names to values (must be JSON-serialisable).
        """
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlLogger:
        """Return self for use as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Close on context exit regardless of exception."""
        self.close()
