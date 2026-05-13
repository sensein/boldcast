"""Tests for torch.distributed lifecycle and DDP model wrapping in ddp.py.

Spawn-based tests use gloo backend over CPU so they run on the login
node without GPUs.  Two categories:
  1. Non-distributed (no spawn needed) — passthrough and error-raise checks.
  2. Multi-process (mp.spawn, 2 workers) — real init / cleanup / DDP wrap.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from boldcast.training.ddp import (
    cleanup_distributed,
    init_distributed,
    setup_model_for_ddp,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _free_port() -> int:
    """Find a free TCP port on localhost."""
    s = socket.socket()
    s.bind(("", 0))
    port: int = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# Skip guard for spawn-based tests
# ---------------------------------------------------------------------------

_gloo_available = dist.is_available() and dist.is_gloo_available()  # type: ignore[attr-defined,unused-ignore]
_skip_no_gloo = pytest.mark.skipif(
    not _gloo_available,
    reason="gloo backend not available",
)

# ---------------------------------------------------------------------------
# Worker functions (module-level so mp.spawn can pickle them)
# ---------------------------------------------------------------------------


def _worker_init_cleanup(rank: int, port: int) -> None:
    """Worker: init → assert state → cleanup."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        init_distributed(backend="gloo")
        assert dist.is_initialized(), "dist should be initialized after init_distributed"
        assert dist.get_rank() == rank
        assert dist.get_world_size() == 2
    finally:
        cleanup_distributed()
        # Verify cleanup
        assert not dist.is_initialized(), "dist should not be initialized after cleanup"
        # Clean env so other tests don't see stale values
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


def _worker_ddp_wrap(rank: int, port: int) -> None:
    """Worker: init → wrap nn.Linear in DDP → verify → cleanup."""
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        init_distributed(backend="gloo")

        linear = nn.Linear(4, 4)
        wrapped = setup_model_for_ddp(linear, find_unused_parameters=False)

        # Must be DDP-wrapped
        assert isinstance(wrapped, DistributedDataParallel), (
            f"Expected DistributedDataParallel, got {type(wrapped)}"
        )
        # .module must be the original Linear
        assert wrapped.module is linear, ".module must be the original nn.Linear"

        # Smoke: forward + backward (DDP auto-syncs grads)
        x = torch.randn(2, 4)
        out = wrapped(x)
        loss = out.sum()
        loss.backward()

        # After backward, gradients should be non-None
        for p in linear.parameters():
            assert p.grad is not None, "gradient should be set after backward"

    finally:
        cleanup_distributed()
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Non-distributed tests (no spawn)
# ---------------------------------------------------------------------------


def test_setup_model_for_ddp_passthrough_when_not_initialized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without calling init_distributed, setup_model_for_ddp returns the
    original module unchanged (no DDP wrapping)."""
    # Ensure distributed env vars absent so dist is definitely not initialized
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        monkeypatch.delenv(key, raising=False)

    assert not dist.is_initialized(), "pre-condition: dist must not be initialized"

    model = nn.Linear(4, 4)
    result = setup_model_for_ddp(model)
    assert result is model, (
        "setup_model_for_ddp should return the original model when dist is not initialized"
    )
    assert not isinstance(result, DistributedDataParallel)


def test_init_distributed_raises_when_not_distributed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calling init_distributed() without RANK/WORLD_SIZE set must raise
    RuntimeError — guards against accidental single-process calls."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    with pytest.raises(RuntimeError, match="RANK"):
        init_distributed()


# ---------------------------------------------------------------------------
# Multi-process spawn tests (gloo, CPU)
# ---------------------------------------------------------------------------


@_skip_no_gloo
def test_init_and_cleanup_2procs_gloo() -> None:
    """Two gloo workers: init_distributed → assert state → cleanup."""
    port = _free_port()
    mp.spawn(_worker_init_cleanup, args=(port,), nprocs=2, join=True)  # type: ignore[attr-defined,no-untyped-call,unused-ignore]


@_skip_no_gloo
def test_setup_model_for_ddp_2procs_gloo() -> None:
    """Two gloo workers: init → DDP-wrap Linear → verify .module → backward."""
    port = _free_port()
    mp.spawn(_worker_ddp_wrap, args=(port,), nprocs=2, join=True)  # type: ignore[attr-defined,no-untyped-call,unused-ignore]
