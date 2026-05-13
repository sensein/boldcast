"""Distributed Data Parallel (DDP) environment detection utilities.

Pure os.environ-based detection of torchrun/torch.distributed setup.
No torch.distributed initialization or GPU code in this module.
Intended for use in Day-5+ training scripts to conditionally log,
checkpoint, and reduce outputs on rank-0 only.

All functions read os.environ directly (no caching).
"""

from __future__ import annotations

import os


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
