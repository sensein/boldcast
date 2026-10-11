"""Distributed Data Parallel (DDP) environment detection and lifecycle utilities.

Provides two layers of functionality:

1. **Pure env-detection helpers** (no torch.distributed calls): read os.environ
   directly and have zero side effects.  Used in Day-4 training scripts to
   conditionally log and checkpoint on rank-0 only.

2. **torch.distributed lifecycle** (``init_distributed``, ``cleanup_distributed``,
   ``setup_model_for_ddp``): initialize / tear down the process group and wrap
   a model in ``DistributedDataParallel``.  These require the calling process
   to have RANK / WORLD_SIZE set (torchrun contract).

All env-detection functions read os.environ directly (no caching).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def is_distributed_run() -> bool:
    """Return True if both RANK and WORLD_SIZE environment variables are set.

    This is the torchrun contract: torchrun always sets both RANK and WORLD_SIZE
    on each process. A process is considered distributed if both are present.

    Returns
    -------
    bool
        True if RANK and WORLD_SIZE are both set, False otherwise.
    """
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def get_rank() -> int:
    """Return the global rank of this process.

    Returns int(os.environ['RANK']) if set, else 0 (non-distributed default).
    In non-distributed mode, all processes have rank 0.

    Returns
    -------
    int
        Global process rank; 0 if RANK is not set.
    """
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    """Return the total number of processes in the distributed run.

    Returns int(os.environ['WORLD_SIZE']) if set, else 1 (non-distributed default).
    In non-distributed mode, world size is 1.

    Returns
    -------
    int
        Total number of processes; 1 if WORLD_SIZE is not set.
    """
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    """Return the local rank (rank within the current node).

    Returns int(os.environ['LOCAL_RANK']) if set, else 0.
    LOCAL_RANK is the rank among processes on the same physical node,
    ranging from 0 to (# GPUs per node - 1).

    Returns
    -------
    int
        Local process rank within this node; 0 if LOCAL_RANK is not set.
    """
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_rank_zero() -> bool:
    """Return True if this process is global rank 0 (the master process).

    In non-distributed mode (RANK not set), returns True.
    In distributed mode, only the process with RANK=0 returns True.

    Always True when get_rank() == 0, which is always true in non-distributed.
    Used to conditionally log metrics, save checkpoints, etc. to avoid
    duplication and filesystem contention in multi-GPU training.

    Returns
    -------
    bool
        True if rank is 0, False otherwise.
    """
    return get_rank() == 0


# ---------------------------------------------------------------------------
# torch.distributed lifecycle
# ---------------------------------------------------------------------------


def init_distributed(backend: str | None = None) -> None:
    """Initialize torch.distributed via env:// init.

    Reads RANK / WORLD_SIZE / LOCAL_RANK / MASTER_ADDR / MASTER_PORT from
    the environment (set by torchrun).  Idempotent: returns immediately if
    ``torch.distributed.is_initialized()``.

    Backend default: ``'nccl'`` if CUDA is available, else ``'gloo'``.
    When CUDA is available, also calls
    ``torch.cuda.set_device(get_local_rank())``.

    Parameters
    ----------
    backend:
        Explicit backend name (``'nccl'``, ``'gloo'``, …).  If *None*, picks
        ``'nccl'`` when CUDA is available, ``'gloo'`` otherwise.

    Raises
    ------
    RuntimeError
        If called when ``is_distributed_run()`` is *False* (RANK / WORLD_SIZE
        not set in the environment) — that is a calling-code bug.
    """
    if dist.is_initialized():
        return

    if not is_distributed_run():
        raise RuntimeError(
            "init_distributed() called but RANK and/or WORLD_SIZE are not set "
            "in the environment.  This function must only be called from a "
            "torchrun-launched process."
        )

    resolved_backend: str
    if backend is not None:
        resolved_backend = backend
    elif torch.cuda.is_available():
        resolved_backend = "nccl"
    else:
        resolved_backend = "gloo"

    if torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())

    dist.init_process_group(backend=resolved_backend, init_method="env://")


def cleanup_distributed() -> None:
    """Destroy the process group if initialized.

    Idempotent: no-op if ``torch.distributed.is_initialized()`` is *False*.
    """
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_model_for_ddp(
    model: nn.Module,
    *,
    find_unused_parameters: bool = False,
) -> nn.Module:
    """Wrap *model* in ``DistributedDataParallel`` if distributed is initialized.

    Returns the model unchanged when ``torch.distributed.is_initialized()`` is
    *False* — this is the "dry-run" / single-GPU code path.

    Parameters
    ----------
    model:
        The ``nn.Module`` to wrap.
    find_unused_parameters:
        Passed directly to ``DistributedDataParallel``.  Set to *True* only
        when the model has parameters that do not receive gradients on every
        forward pass (uncommon; incurs overhead).

    Returns
    -------
    nn.Module
        A ``DistributedDataParallel``-wrapped module when distributed is active,
        or the original *model* otherwise.  The wrapped module exposes the
        original as ``.module``.
    """
    if not dist.is_initialized():
        return model

    if torch.cuda.is_available():
        return DistributedDataParallel(
            model,
            device_ids=[get_local_rank()],
            find_unused_parameters=find_unused_parameters,
        )
    # CPU / gloo test path: no device_ids
    return DistributedDataParallel(
        model,
        find_unused_parameters=find_unused_parameters,
    )
