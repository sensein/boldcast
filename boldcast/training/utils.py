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

__all__ = ["JsonlLogger", "heldout_decreased_by", "save_checkpoint", "seed_everything"]

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


def heldout_decreased_by(
    history: dict[str, list[float]],
    frac: float = 0.30,
    window: int = 3,
) -> bool:
    """Return True if held-out val loss dropped by at least ``frac``.

    Compares the mean of the last ``window`` ``history['val_loss']``
    values to the mean of the first ``window``. Returns True iff
    ``mean(last) <= (1 - frac) * mean(first)``.

    Used at end of Day-5 training to assert acceptance criterion #2
    (≥30% drop) without binding to a single-step threshold (seed-
    independent per schist ID 229).

    Parameters
    ----------
    history
        Trainer.fit return dict; must have key ``'val_loss'``.
    frac
        Required drop fraction (0.30 = 30%).
    window
        Number of measurements to average at each end.

    Returns
    -------
    bool
        True if the drop criterion holds, False otherwise (including
        if val_loss has fewer than ``2 * window`` entries).

    Raises
    ------
    KeyError
        If history does not have key ``'val_loss'``.
    """
    losses = history["val_loss"]
    if len(losses) < 2 * window:
        return False
    first_mean = sum(losses[:window]) / window
    last_mean = sum(losses[-window:]) / window
    return last_mean <= (1.0 - frac) * first_mean


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
