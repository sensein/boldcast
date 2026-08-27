"""Regression test: DDP forward+backward smoke for BOLDcastDemo.

Three invariants verified across 2 gloo/CPU workers:
  1. ``setup_model_for_ddp(find_unused_parameters=False)`` emits no
     ``find_unused_parameters`` UserWarning.
  2. DDP gradient sync: after backward, all_reduce of any parameter's grad
     is a no-op (DDP already averaged it).
  3. No NaN in loss or gradients at step 1+.

Uses BOLDcastDemo(n_layers=0) so the test path does not import mamba_ssm
and runs under uv on the login node (CPU only, gloo backend).
"""

from __future__ import annotations

import os
import socket
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from boldcast.models.boldcast_demo import BOLDcastDemo
from boldcast.training.ddp import (
    cleanup_distributed,
    init_distributed,
    setup_model_for_ddp,
)

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


def _identity_adjacency(n_patches: int, k: int) -> torch.Tensor:
    """Cyclic identity-ish adjacency: neighbor j of patch i is (i+j) % n."""
    import numpy as np

    adj = np.zeros((n_patches, k), dtype=np.int64)
    for i in range(n_patches):
        adj[i, 0] = i
        for j in range(1, k):
            adj[i, j] = (i + j) % n_patches
    return torch.from_numpy(adj)


# ---------------------------------------------------------------------------
# Skip guard
# ---------------------------------------------------------------------------

_gloo_available = dist.is_available() and dist.is_gloo_available()  # type: ignore[attr-defined,unused-ignore]
_skip_no_gloo = pytest.mark.skipif(
    not _gloo_available,
    reason="gloo backend not available",
)

# ---------------------------------------------------------------------------
# Worker function (module-level so mp.spawn can pickle it)
# ---------------------------------------------------------------------------


def _worker_forward_smoke(rank: int, port: int) -> None:
    """Two-step forward+backward smoke with DDP-wrapped BOLDcastDemo.

    Checks (all three must pass on every rank):
    * No ``find_unused_parameters`` UserWarning from DDP setup.
    * Loss is finite at every step.
    * After DDP-synced backward, manual all_reduce of a grad tensor is a
      no-op within float epsilon (DDP averaged it already).
    """
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    try:
        init_distributed(backend="gloo")

        n_patches, k = 8, 4
        adj = _identity_adjacency(n_patches, k)

        # Same seed on both ranks → identical model init (required for DDP
        # correctness; ranks must start with the same weights).
        torch.manual_seed(42)
        model = BOLDcastDemo(
            d_in=1,
            d_model=8,
            n_layers=0,
            n_patches=n_patches,
            k_neighbors=k,
            adjacency=adj,
            horizons=(1,),
            use_checkpoint=False,
        )

        # Catch any warnings emitted during DDP setup + training loop.
        with warnings.catch_warnings(record=True) as w_list:
            warnings.simplefilter("always")
            wrapped = setup_model_for_ddp(model, find_unused_parameters=False)
            optimizer = torch.optim.SGD(wrapped.parameters(), lr=1e-3)

            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                # Ranks intentionally differ (+rank) to exercise gradient sync.
                tokens = torch.randn(2, 5, n_patches, 1) + float(rank)
                pred = wrapped(tokens)  # (B, T-H, P, H, d_in)
                target = torch.randn_like(pred)
                loss = ((pred - target) ** 2).mean()

                # Invariant 3: finite loss.
                assert torch.isfinite(loss).item(), f"non-finite loss at step {step}, rank {rank}"

                loss.backward()
                optimizer.step()

            # Invariant 1: no find_unused_parameters warning.
            for warning in w_list:
                msg = str(warning.message).lower()
                assert "find_unused_parameters" not in msg, (
                    f"Unexpected find_unused_parameters warning on rank {rank}: {warning.message}"
                )

        # Invariant 2: DDP gradient sync.
        # DDP averages gradients across ranks during backward via reducer hooks.
        # So: all_reduce(SUM) / world_size == local_grad (within float eps).
        any_param = next(p for p in wrapped.parameters() if p.grad is not None)
        assert any_param.grad is not None  # narrow for mypy
        local_grad = any_param.grad.clone()

        # Invariant 3 (grad): finite gradients.
        assert torch.isfinite(local_grad).all().item(), f"non-finite gradient on rank {rank}"

        synced = local_grad.clone()
        dist.all_reduce(synced, op=dist.ReduceOp.SUM)
        expected = synced / 2.0  # world_size == 2
        diff = (local_grad - expected).abs().max().item()
        assert diff < 1e-5, (
            f"DDP gradients not synced across ranks "
            f"(max |local - all_reduce/2| = {diff:.2e}) on rank {rank}"
        )

    finally:
        cleanup_distributed()
        for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            os.environ.pop(key, None)


# ---------------------------------------------------------------------------
# Test wrapper
# ---------------------------------------------------------------------------


@_skip_no_gloo
def test_ddp_forward_smoke_2procs_gloo() -> None:
    """Two gloo workers: DDP-wrap BOLDcastDemo(n_layers=0), 2 forward+backward,
    no find_unused_parameters warning, loss and gradients finite, gradients
    synced across ranks."""
    port = _free_port()
    mp.spawn(_worker_forward_smoke, args=(port,), nprocs=2, join=True)  # type: ignore[attr-defined,no-untyped-call,unused-ignore]
