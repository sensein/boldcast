"""Day-5 Task 3 DDP-aware Trainer tests.

Tests:
  A - THE TRAP GUARD: sampler.set_epoch is called per epoch cycle
  B - rank-0-only IO: JsonlLogger + checkpoints suppressed on non-rank-0
  C - all_reduce no-op in single-process: regression test
  D - existing test_trainer.py tests still pass (via this file importing
      and running them; see conftest.py — actually we just assert no
      regressions by running the full suite in CI; here we just run the
      4 new tests)

CPU-only; no mamba-ssm import.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from boldcast.models.boldcast_demo import BOLDcastDemo
from boldcast.training.optim import build_optimizer
from boldcast.training.trainer import Trainer
from boldcast.training.utils import seed_everything
from torch.utils.data import DataLoader, Dataset, Sampler

# ---------------------------------------------------------------------------
# Shared helpers (same pattern as tests/test_trainer.py)
# ---------------------------------------------------------------------------


def _identity_adjacency(n_patches: int, k: int) -> torch.Tensor:
    import numpy as np

    adj = np.zeros((n_patches, k), dtype=np.int64)
    for i in range(n_patches):
        adj[i, 0] = i
        for j in range(1, k):
            adj[i, j] = (i + j) % n_patches
    return torch.from_numpy(adj)


class _SyntheticWindows(Dataset[dict[str, torch.Tensor]]):
    """Tiny AR(1) dataset; matches test_trainer.py's fixture."""

    def __init__(
        self,
        n: int,
        T: int,
        P: int,
        rho: float = 0.9,
        seed: int = 0,
    ) -> None:
        g = torch.Generator().manual_seed(seed)
        x = torch.zeros(n, T, P)
        x[:, 0, :] = torch.randn(n, P, generator=g)
        noise_scale = (1.0 - rho**2) ** 0.5
        for t in range(1, T):
            x[:, t, :] = rho * x[:, t - 1, :] + noise_scale * torch.randn(n, P, generator=g)
        self.windows = x

    def __len__(self) -> int:
        return self.windows.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"tokens": self.windows[idx]}


def _build_tiny_setup(
    tmp_path: Path,
    horizons: tuple[int, ...] = (1, 5),
    n_data: int = 5,
) -> tuple[BOLDcastDemo, torch.optim.AdamW, DataLoader[dict[str, torch.Tensor]], Trainer]:
    seed_everything(0)
    n_patches, k = 8, 4
    adj = _identity_adjacency(n_patches, k)
    model = BOLDcastDemo(
        d_in=1,
        d_model=8,
        n_layers=0,
        n_patches=n_patches,
        k_neighbors=k,
        adjacency=adj,
        horizons=horizons,
    )
    opt = build_optimizer(model, lr=3e-3, weight_decay=0.0, betas=(0.9, 0.95))
    ds = _SyntheticWindows(n=n_data, T=16, P=n_patches, seed=0)
    loader: DataLoader[dict[str, torch.Tensor]] = DataLoader(
        ds, batch_size=1, shuffle=False, num_workers=0
    )
    trainer = Trainer(
        model=model,
        optimizer=opt,
        scheduler=None,
        device=torch.device("cpu"),
        horizons=horizons,
        grad_clip_norm=1.0,
        precision="fp32",
        log_every=100,
        out_dir=tmp_path,
    )
    return model, opt, loader, trainer


# ---------------------------------------------------------------------------
# Spy helper
# ---------------------------------------------------------------------------


class _SamplerSpy(Sampler[int]):
    """Records every set_epoch call.  Mimics DistributedSampler's contract.

    Inherits from ``Sampler[int]`` so mypy accepts it where ``Sampler[int]``
    is expected.  The ``__iter__`` / ``__len__`` stubs are never called by
    the Trainer (it iterates the DataLoader, not the sampler directly).
    """

    def __init__(self) -> None:
        super().__init__()
        self.epochs_seen: list[int] = []

    def set_epoch(self, e: int) -> None:
        self.epochs_seen.append(e)

    def __iter__(self) -> Iterator[int]:  # pragma: no cover
        return iter([])

    def __len__(self) -> int:  # pragma: no cover
        return 0


# ---------------------------------------------------------------------------
# Test A — THE TRAP GUARD
# ---------------------------------------------------------------------------


def test_sampler_set_epoch_called_per_cycle(tmp_path: Path) -> None:
    """_infinite_loader_with_epoch must call sampler.set_epoch(epoch) at the
    start of every epoch cycle.

    Loader has 5 batches; max_steps=12 → 3 cycles (batches 0-4, 5-9, 10-11).
    Expects set_epoch called with [0, 1, 2].
    """
    _, _, loader, trainer = _build_tiny_setup(tmp_path, n_data=5)
    spy = _SamplerSpy()
    trainer.fit(loader, max_steps=12, sampler=spy)
    # At minimum 2 distinct epochs must be observed (proves cycling happened)
    assert len(spy.epochs_seen) >= 2, (
        f"Expected at least 2 set_epoch calls but got: {spy.epochs_seen}"
    )
    # Epochs must start at 0
    assert spy.epochs_seen[0] == 0, f"First set_epoch call should be 0, got {spy.epochs_seen[0]}"
    # Epochs must strictly increase by 1
    for i in range(1, len(spy.epochs_seen)):
        assert spy.epochs_seen[i] == spy.epochs_seen[i - 1] + 1, (
            f"Non-sequential epoch call: {spy.epochs_seen}"
        )
    # With 5-batch loader and 12 steps: cycles at steps 0, 5, 10 → epochs [0,1,2]
    assert spy.epochs_seen == [0, 1, 2], f"Expected [0, 1, 2], got {spy.epochs_seen}"


# ---------------------------------------------------------------------------
# Test B — rank-0-only IO
# ---------------------------------------------------------------------------


def test_rank_zero_only_io_suppressed_on_non_rank_zero(tmp_path: Path) -> None:
    """When is_rank_zero() returns False, no JSONL log or .pt checkpoint
    should be created under out_dir.
    """
    with patch("boldcast.training.trainer.is_rank_zero", return_value=False):
        _, _, loader, _ = _build_tiny_setup(tmp_path)
        # Rebuild trainer INSIDE the patch so __init__ sees is_rank_zero=False
        n_patches, k = 8, 4
        adj = _identity_adjacency(n_patches, k)
        seed_everything(0)
        model = BOLDcastDemo(
            d_in=1,
            d_model=8,
            n_layers=0,
            n_patches=n_patches,
            k_neighbors=k,
            adjacency=adj,
            horizons=(1, 5),
        )
        opt = build_optimizer(model, lr=3e-3, weight_decay=0.0, betas=(0.9, 0.95))
        trainer = Trainer(
            model=model,
            optimizer=opt,
            scheduler=None,
            device=torch.device("cpu"),
            horizons=(1, 5),
            grad_clip_norm=1.0,
            precision="fp32",
            log_every=1,  # would log every step if rank-0
            ckpt_every=2,  # would checkpoint at step 2, 4, ... if rank-0
            out_dir=tmp_path,
        )
        trainer.fit(loader, max_steps=5)

    jsonl_path = tmp_path / "loss_log.jsonl"
    assert not jsonl_path.exists(), (
        "loss_log.jsonl was created on non-rank-0 process — IO not suppressed"
    )
    pt_files = list(tmp_path.glob("*.pt"))
    assert len(pt_files) == 0, f"Checkpoint files found on non-rank-0 process: {pt_files}"


# ---------------------------------------------------------------------------
# Test C — all_reduce no-op in single-process
# ---------------------------------------------------------------------------


def test_all_reduce_no_op_single_process(tmp_path: Path) -> None:
    """Without dist.init_process_group, _all_reduce_mean must be a no-op.

    Verifies that history["loss"] still has the expected length and
    all values are finite (no NaN / Inf from a broken all_reduce path).
    """
    _, _, loader, trainer = _build_tiny_setup(tmp_path, n_data=5)
    history = trainer.fit(loader, max_steps=10)
    assert len(history["loss"]) == 10, f"Expected 10 loss values, got {len(history['loss'])}"
    for i, v in enumerate(history["loss"]):
        assert torch.isfinite(torch.tensor(v)), f"Non-finite loss at step {i}: {v}"


# ---------------------------------------------------------------------------
# Test D — existing tests unaffected (documentation; actual run in full suite)
# ---------------------------------------------------------------------------


def test_existing_trainer_tests_still_importable() -> None:
    """Smoke-test that test_trainer module imports without error.

    The full 6 existing tests are verified by running pytest on
    tests/test_trainer.py separately; this just confirms no import-time
    breakage from the new trainer.py changes.
    """
    import importlib

    mod = importlib.import_module("tests.test_trainer")
    assert hasattr(mod, "_build_tiny_setup")
    assert hasattr(mod, "test_trainer_overfit_reduces_loss_on_cpu")


# ---------------------------------------------------------------------------
# Task 4 — _eval + fit() val loop
# ---------------------------------------------------------------------------


def test_fit_runs_val_loop_with_val_loader_and_val_every(tmp_path: Path) -> None:
    """val_loader + val_every: history accumulates val_loss at the right cadence."""
    _, _, train_loader, trainer = _build_tiny_setup(tmp_path, n_data=5)
    _, _, val_loader, _ = _build_tiny_setup(tmp_path, n_data=4)  # 4-window val set
    history = trainer.fit(
        train_loader,
        max_steps=30,
        val_loader=val_loader,
        val_every=10,
    )
    # max_steps=30, val_every=10 → val at steps 9, 19, 29 (1-indexed eval boundary)
    assert len(history["val_step"]) == 3, (
        f"Expected 3 val measurements, got {len(history['val_step'])}: {history['val_step']}"
    )
    assert len(history["val_loss"]) == 3
    for v in history["val_loss"]:
        assert torch.isfinite(torch.tensor(v))
    # Model must be in training mode after fit() returns
    assert trainer.model.training is True


def test_fit_no_val_loader_leaves_history_val_empty(tmp_path: Path) -> None:
    """Without val_loader, history['val_step'] / history['val_loss'] are present but empty."""
    _, _, loader, trainer = _build_tiny_setup(tmp_path, n_data=5)
    history = trainer.fit(loader, max_steps=10)
    assert "val_step" in history and history["val_step"] == []
    assert "val_loss" in history and history["val_loss"] == []


def test_eval_runs_under_no_grad(tmp_path: Path) -> None:
    """_eval should not leave any gradients on model parameters."""
    _, _, train_loader, trainer = _build_tiny_setup(tmp_path, n_data=5)
    _, _, val_loader, _ = _build_tiny_setup(tmp_path, n_data=4)
    # Zero existing grads (none yet, but be defensive)
    for p in trainer.model.parameters():
        if p.grad is not None:
            p.grad = None
    trainer._eval(val_loader)
    # _eval must not accumulate gradients
    for p in trainer.model.parameters():
        assert p.grad is None, "gradients should not be accumulated by _eval"


def test_eval_uses_unwrapped_module_under_ddp(tmp_path: Path) -> None:
    """Regression for Day-5 NCCL deadlock (sbatch 14282858 / Issue #16):
    ``_eval``'s rank-0-only forward MUST go through the unwrapped module
    (``model.module``), NOT the DDP wrapper.

    The DDP wrapper calls ``_sync_buffers()`` at the start of forward
    whenever ``require_forward_param_sync=True`` (which it is immediately
    after each ``loss.backward()``). ``_sync_buffers`` issues a broadcast
    collective on rank 0 with no matching call on rank 1 (rank 1 is
    already waiting at the explicit ``dist.broadcast(result, src=0)``
    further down). The mismatched broadcast shifts NCCL SeqNum alignment
    by 1 across both ranks → the subsequent ``dist.broadcast(result, …)``
    and ``dist.barrier()`` land on different op types between ranks →
    NCCL deadlocks at the next ALLREDUCE.

    BOLDcastDemo has 4 ``KNNAttention.adjacency`` buffers (``spatial.py:53``)
    plus any future BN/LN running stats — any non-empty buffer set triggers
    this collective. The fix forwards through ``getattr(self.model,
    "module", self.model)`` on the rank-0 eval path so DDP's forward-time
    machinery is bypassed entirely.

    Test discrimination: a fake DDP-like wrap exposes ``.module`` and
    increments ``wrapper_forward_calls`` when its own forward runs. If
    ``_eval`` forwards through the wrapper, the counter goes up and the
    assertion fails. If ``_eval`` forwards through ``.module`` directly,
    the wrapper's forward is never entered.
    """
    _, _, _, trainer = _build_tiny_setup(tmp_path, n_data=5)
    real_model = trainer.model

    wrapper_forward_calls: list[int] = []
    inner_forward_calls: list[int] = []

    class _FakeDDPWrap(torch.nn.Module):
        """Mimics DDP's surface: exposes ``.module`` and a forward that
        would be the place where DDP's ``_sync_buffers`` runs."""

        def __init__(self, inner: torch.nn.Module) -> None:
            super().__init__()
            self.module = inner

        def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
            wrapper_forward_calls.append(1)
            out: torch.Tensor = self.module(*args, **kwargs)
            return out

    # Also instrument the inner forward so the test fails loudly if _eval
    # somehow skips the forward pass entirely (which would trivially keep
    # wrapper_forward_calls at zero).
    orig_inner_forward = real_model.forward

    def _spy_inner_forward(*args: object, **kwargs: object) -> torch.Tensor:
        inner_forward_calls.append(1)
        return orig_inner_forward(*args, **kwargs)  # type: ignore[no-any-return]

    real_model.forward = _spy_inner_forward  # type: ignore[method-assign]
    trainer.model = _FakeDDPWrap(real_model)

    val_batches: list[dict[str, torch.Tensor]] = [
        {"tokens": torch.randn(1, 16, 8)} for _ in range(3)
    ]
    with (
        patch("boldcast.training.trainer.dist.is_initialized", return_value=True),
        patch("boldcast.training.trainer.is_rank_zero", return_value=True),
        patch("boldcast.training.trainer.dist.broadcast"),
        patch("boldcast.training.trainer.dist.barrier"),
    ):
        trainer._eval(val_batches)  # type: ignore[arg-type]

    assert inner_forward_calls, (
        "_eval did not forward through the inner model at all — test cannot "
        "discriminate. Expected len(val_batches)=3 inner forwards."
    )
    assert not wrapper_forward_calls, (
        f"_eval forwarded through the DDP wrapper {len(wrapper_forward_calls)} "
        "time(s) on rank-0. DDP.forward() triggers _sync_buffers() which "
        "issues a broadcast collective with no matching call on rank 1, "
        "deadlocking NCCL. _eval must forward through model.module on the "
        "rank-0-only branch."
    )


@pytest.mark.parametrize("rank_zero", [True, False])
def test_eval_broadcast_tensor_on_trainer_device(tmp_path: Path, rank_zero: bool) -> None:
    """Regression for sbatch 14220441: dist.broadcast must receive a tensor on
    trainer.device, not the default CPU — on BOTH the rank-0 and non-rank-0
    construction paths.

    Under NCCL, broadcasting a CPU tensor raises
    ``RuntimeError: No backend type associated with device type cpu``.
    The fix is to construct the broadcast tensor with ``device=self.device``
    in both ``_eval`` branches (rank-0 uses ``torch.tensor(mean_loss, …)``,
    other ranks use ``torch.zeros((), …)``).

    On a CPU-only test runner, ``self.device == cpu`` and the default
    placement is also CPU — a tensor-location check alone would pass even
    with the bug. Instead, we wrap ``torch.tensor`` / ``torch.zeros`` and
    assert that the call site passed an explicit ``device=`` kwarg matching
    ``self.device``. That is the invariant that breaks the NCCL path.
    """
    _, _, _, trainer = _build_tiny_setup(tmp_path, n_data=5)
    # Bare list iterable — DataLoader probes dist.get_world_size when
    # is_initialized() is True, which would fail under our mock.
    val_batches: list[dict[str, torch.Tensor]] = [
        {"tokens": torch.randn(1, 16, 8)} for _ in range(3)
    ]
    with (
        patch("boldcast.training.trainer.dist.is_initialized", return_value=True),
        patch("boldcast.training.trainer.is_rank_zero", return_value=rank_zero),
        patch("boldcast.training.trainer.dist.broadcast"),
        patch("boldcast.training.trainer.dist.barrier"),
        patch.object(torch, "tensor", wraps=torch.tensor) as mock_tensor,
        patch.object(torch, "zeros", wraps=torch.zeros) as mock_zeros,
    ):
        trainer._eval(val_batches)  # type: ignore[arg-type]

    if rank_zero:
        # Rank-0 path constructs the result via torch.tensor(mean_loss, …).
        calls = [
            call for call in mock_tensor.call_args_list if call.kwargs.get("dtype") is torch.float64
        ]
        ctor = "torch.tensor"
    else:
        # Non-rank-0 path uses torch.zeros((), …).
        calls = [
            call for call in mock_zeros.call_args_list if call.kwargs.get("dtype") is torch.float64
        ]
        ctor = "torch.zeros"
    assert calls, f"expected a {ctor}(…, dtype=float64, …) call in _eval on rank_zero={rank_zero}"
    for call in calls:
        assert call.kwargs.get("device") == trainer.device, (
            f"_eval result tensor constructed via {ctor} without "
            f"device=self.device on rank_zero={rank_zero} "
            f"(got device={call.kwargs.get('device')!r}, "
            f"expected {trainer.device!r}). NCCL will reject a CPU tensor."
        )
