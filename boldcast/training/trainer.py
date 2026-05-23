"""Single-GPU / DDP Trainer class (Day 4 + Day 5, ADR 0005 D3).

Raw PyTorch — no Lightning. Day-5 DDP wraps the model in
``DistributedDataParallel`` and passes a ``DistributedSampler``-driven
DataLoader together with the ``sampler`` kwarg to ``fit()``.

Inner loop per step:
    1. Pull next batch from _infinite_loader_with_epoch(dataloader, sampler).
    2. Move tokens to device; add singleton d_in axis -> (B, T, P, 1).
    3. Build (B, T_valid, P, H, d_in) target via build_forecast_targets.
    4. Forward under BF16 autocast (when precision='bf16' on CUDA).
    5. Slice prediction to T_valid positions; compute MSE.
    6. NaN guard: raise RuntimeError if loss is not finite.
    7. Backward, optional grad-clip, optimizer.step, optional scheduler.step.
    8. All-reduce loss across ranks (no-op when not distributed).
    9. Append {step, loss, lr} to stdout (every log_every, rank-0 only) + JSONL.
    10. Optional checkpoint every ckpt_every steps (rank-0 only).

BF16 does NOT require GradScaler — that's FP16 only. CPU precision='fp32'
disables autocast entirely (PyTorch's BF16 CPU support is incomplete and
not needed for the n_layers=0 trainer test).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Sampler

from boldcast.training.ddp import is_rank_zero
from boldcast.training.loss import build_forecast_targets, forecasting_loss
from boldcast.training.utils import JsonlLogger, save_checkpoint


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all ranks. No-op when not distributed."""
    if dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def _infinite_loader_with_epoch(
    dataloader: DataLoader[dict[str, torch.Tensor]] | Iterable[dict[str, torch.Tensor]],
    sampler: Sampler[int] | None,
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever, calling ``sampler.set_epoch(epoch)`` each cycle.

    Unlike ``itertools.cycle`` (which caches every yielded element on the
    first pass), this calls ``iter(dataloader)`` afresh each epoch, keeping
    memory flat across long runs.

    If ``sampler`` is provided and has a ``set_epoch`` method (as
    ``DistributedSampler`` does), it is called with the current epoch index
    before each cycle — required so each rank sees a different shuffle order
    every epoch in DDP training.

    Parameters
    ----------
    dataloader
        Any iterable of ``{str: Tensor}`` dicts.
    sampler
        Optional sampler to call ``set_epoch`` on each cycle.  Pass
        ``None`` for single-GPU / no-shuffle training.
    """
    epoch = 0
    while True:
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        yield from dataloader
        epoch += 1


__all__ = ["Trainer"]

# Note: the prior `_infinite_loader(dataloader)` (Day-4) was renamed to
# `_infinite_loader_with_epoch(dataloader, sampler)` for the Day-5 DDP
# set_epoch hook. No external callers — the rename is breaking but
# contained.

# torch.amp.autocast isn't in torch's typed exports; alias with the same
# `unused-ignore` form as scripts/day3_validate_model.py so the file passes
# under both older and newer torch stubs.
autocast = torch.amp.autocast  # type: ignore[attr-defined,unused-ignore]


class Trainer:
    """Minimal single-GPU forecasting trainer.

    Parameters
    ----------
    model
        Trained module; must accept ``(B, T, P, d_in)`` and return
        ``(B, T, P, H, d_in)`` (BOLDcastDemo / BaselineSchaefer400 do).
    optimizer
    scheduler
        ``None`` for constant LR. Otherwise stepped after every
        ``optimizer.step()``.
    device
        Target device; tensors are moved here per-batch.
    horizons
        Forecast offsets — must match the model's head config.
    grad_clip_norm
        ``None`` disables clipping.
    precision
        ``'bf16'`` enables ``torch.amp.autocast(device_type='cuda',
        dtype=torch.bfloat16)`` around forward + loss. ``'fp32'`` runs
        forward in fp32 (used for the CPU test path).
    log_every
        Stdout log frequency (steps). JSONL is written every step.
    ckpt_every
        ``None`` to disable periodic checkpointing.
    out_dir
        ``None`` disables JSONL + checkpoint output. Otherwise the JSONL
        is at ``out_dir / "loss_log.jsonl"`` and periodic ckpts at
        ``out_dir / "ckpt_step{step}.pt"``.

    Notes
    -----
    For DDP training, ``boldcast.training.ddp.init_distributed()`` must be
    called BEFORE constructing the Trainer. Rank-0 status is captured in
    ``__init__`` and used to gate stdout, JSONL writes, and checkpoint
    saves. Constructing the Trainer before init means all ranks will
    behave as rank-0 (and corrupt the shared JSONL / ckpt files).
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        device: torch.device,
        horizons: Sequence[int],
        grad_clip_norm: float | None = 1.0,
        precision: Literal["fp32", "bf16"] = "bf16",
        log_every: int = 10,
        ckpt_every: int | None = None,
        out_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.horizons = tuple(int(h) for h in horizons)
        # Catch a common config-sweep bug: passing different `horizons` to
        # Trainer than the model was constructed with would silently broadcast
        # through F.mse_loss with a UserWarning. Fail loudly instead.
        model_horizons = getattr(model, "horizons", None)
        if model_horizons is not None and tuple(model_horizons) != self.horizons:
            raise ValueError(
                f"Trainer.horizons={self.horizons!r} must match "
                f"model.horizons={tuple(model_horizons)!r}"
            )
        self.grad_clip_norm = grad_clip_norm
        self.precision = precision
        self.log_every = log_every
        self.ckpt_every = ckpt_every
        self.out_dir = Path(out_dir) if out_dir is not None else None
        # Capture rank at init time so monkeypatching is straightforward.
        self._is_rank_zero: bool = is_rank_zero()
        self._logger: JsonlLogger | None = None
        if self.out_dir is not None and self._is_rank_zero:
            self._logger = JsonlLogger(self.out_dir / "loss_log.jsonl")

    def fit(
        self,
        dataloader: DataLoader[dict[str, torch.Tensor]] | Iterable[dict[str, torch.Tensor]],
        max_steps: int,
        *,
        sampler: Sampler[int] | None = None,
        val_loader: DataLoader[dict[str, torch.Tensor]] | None = None,  # Task 4
        val_every: int | None = None,  # Task 4
    ) -> dict[str, list[float]]:
        """Run ``max_steps`` training iterations.

        Cycles the dataloader so a single-batch dataset still yields
        ``max_steps`` updates (Day-4 overfit pattern). Returns step / loss
        / lr history.

        Parameters
        ----------
        dataloader
            Any iterable of ``{str: Tensor}`` dicts. ``DataLoader`` is the
            canonical source; bare iterables work for tests.
        max_steps
            Number of gradient steps to take.
        sampler
            Optional ``DistributedSampler`` (or any object with
            ``set_epoch(int)``). When provided, ``set_epoch`` is called at
            the start of every epoch cycle so each DDP rank sees a
            different shuffle order. Ignored in single-GPU mode.
        val_loader
            Optional held-out validation DataLoader. When provided
            together with ``val_every``, ``_eval`` runs every
            ``val_every`` steps; results land in ``history['val_step']``
            and ``history['val_loss']``.
        val_every
            Validation cadence (steps). Pair with ``val_loader``.

        Returns
        -------
        dict[str, list[float]]
            Keys ``"step"``, ``"loss"``, ``"lr"``; each list has length
            ``max_steps``.
        """
        self.model.train()
        history: dict[str, list[float]] = {
            "step": [],
            "loss": [],
            "lr": [],
            "val_step": [],
            "val_loss": [],
        }
        data_iter = _infinite_loader_with_epoch(dataloader, sampler)
        try:
            for step in range(max_steps):
                batch = next(data_iter)
                loss_value, lr_value = self._train_step(batch, step)
                history["step"].append(float(step))
                history["loss"].append(loss_value)
                history["lr"].append(lr_value)
                if step % self.log_every == 0 or step == max_steps - 1:
                    if self._is_rank_zero:
                        print(
                            f"[trainer] step={step:>5d}  "
                            f"loss={loss_value:.6f}  lr={lr_value:.2e}"
                        )
                if (
                    self._is_rank_zero
                    and self.ckpt_every is not None
                    and self.out_dir is not None
                    and (step + 1) % self.ckpt_every == 0
                ):
                    # Unwrap DDP module before checkpointing.
                    raw_model = getattr(self.model, "module", self.model)
                    save_checkpoint(
                        raw_model,
                        self.optimizer,
                        step=step + 1,
                        path=self.out_dir / f"ckpt_step{step + 1}.pt",
                    )
                if val_loader is not None and val_every is not None:
                    if (step + 1) % val_every == 0:
                        val_loss = self._eval(val_loader)
                        history["val_step"].append(float(step))
                        history["val_loss"].append(val_loss)
                        if self._is_rank_zero:
                            print(
                                f"[trainer] val_step={step:>5d}  "
                                f"val_loss={val_loss:.6f}"
                            )
                            if self._logger is not None:
                                self._logger.write(
                                    {"step": step, "val_loss": val_loss}
                                )
        finally:
            if self._logger is not None:
                self._logger.close()
        return history

    def _eval(
        self,
        val_loader: DataLoader[dict[str, torch.Tensor]],
    ) -> float:
        """Run model.eval() over val_loader, return mean MSE forecasting loss.

        Iterates val_loader fully (no infinite-loader). Runs under
        autocast(bf16) on CUDA. No gradients accumulated. Restores
        model.training=True before returning.

        In DDP mode, only rank-0 actually iterates val data; other ranks
        wait at dist.barrier(). The rank-0 mean is broadcast to all ranks
        so logging is consistent.

        Returns
        -------
        float
            Mean validation MSE loss (averaged over batches).
        """
        was_training = self.model.training
        self.model.eval()
        try:
            if dist.is_initialized():
                if is_rank_zero():
                    # Forward through the UNWRAPPED module. DDP.forward() calls
                    # _sync_buffers() at the start of every forward (whenever
                    # require_forward_param_sync=True, which it is right after
                    # backward), issuing a broadcast collective from rank 0 to
                    # all other ranks. Other ranks are NOT here — they fall
                    # straight to dist.broadcast(result, src=0) below. The
                    # unmatched buffer broadcast shifts NCCL SeqNum alignment
                    # by 1 across ranks → subsequent collectives mismatch in
                    # op type → deadlock. Issue #16 / sbatch 14282858.
                    eval_module = getattr(self.model, "module", self.model)
                    total_loss = 0.0
                    n_batches = 0
                    with torch.no_grad():
                        for batch in val_loader:
                            tokens = batch["tokens"].to(self.device).unsqueeze(-1)
                            targets = build_forecast_targets(tokens, self.horizons)
                            with autocast(
                                device_type="cuda",
                                dtype=torch.bfloat16,
                                enabled=(
                                    self.precision == "bf16"
                                    and self.device.type == "cuda"
                                ),
                            ):
                                pred_full: torch.Tensor = eval_module(tokens)
                                pred = pred_full[:, : targets.shape[1]]
                                loss = forecasting_loss(pred, targets)
                            total_loss += float(loss.item())
                            n_batches += 1
                    mean_loss = total_loss / max(n_batches, 1)
                    result = torch.tensor(
                        mean_loss, dtype=torch.float64, device=self.device
                    )
                else:
                    result = torch.zeros((), dtype=torch.float64, device=self.device)
                # NCCL has no CPU backend — the broadcast tensor must live on
                # the device backing the process group (CUDA under NCCL).
                dist.broadcast(result, src=0)
                dist.barrier()
                return float(result.item())
            else:
                total_loss = 0.0
                n_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        tokens = batch["tokens"].to(self.device).unsqueeze(-1)
                        targets = build_forecast_targets(tokens, self.horizons)
                        with autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=(
                                self.precision == "bf16"
                                and self.device.type == "cuda"
                            ),
                        ):
                            pred_full = self.model(tokens)
                            pred = pred_full[:, : targets.shape[1]]
                            loss = forecasting_loss(pred, targets)
                        total_loss += float(loss.item())
                        n_batches += 1
                return total_loss / max(n_batches, 1)
        finally:
            self.model.train(mode=was_training)

    def _train_step(
        self,
        batch: dict[str, torch.Tensor],
        step: int,
    ) -> tuple[float, float]:
        """Execute one gradient step.

        Parameters
        ----------
        batch
            Dict with key ``"tokens"`` of shape ``(B, T, P)``.
        step
            Current step index (for error messages and logging).

        Returns
        -------
        tuple[float, float]
            ``(loss_value, current_lr)``
        """
        tokens = batch["tokens"].to(self.device).unsqueeze(-1)  # (B, T, P, 1)
        targets = build_forecast_targets(tokens, self.horizons)
        self.optimizer.zero_grad(set_to_none=True)
        with autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(self.precision == "bf16" and self.device.type == "cuda"),
        ):
            pred_full: torch.Tensor = self.model(tokens)
            pred = pred_full[:, : targets.shape[1]]
            loss = forecasting_loss(pred, targets)

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss at step {step}: {loss.item()}"
            )

        loss.backward()  # type: ignore[no-untyped-call,unused-ignore]
        if self.grad_clip_norm is not None:
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_clip_norm
            )
        else:
            # Compute the total norm without clipping (for logging only).
            grad_norm_t = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), float("inf")
            )
        grad_norm = float(grad_norm_t)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        lr = float(self.optimizer.param_groups[0]["lr"])
        # Average loss across ranks for consistent logging; no-op when not
        # distributed.  Detached clone avoids touching the gradient graph.
        reduced = _all_reduce_mean(loss.detach().clone())
        loss_value = float(reduced.item())
        if self._logger is not None:
            self._logger.write({
                "step": step,
                "loss": loss_value,
                "lr": lr,
                "grad_norm": grad_norm,
            })
        return loss_value, lr
