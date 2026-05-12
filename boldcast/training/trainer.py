"""Single-GPU Trainer class for Day 4 (ADR 0005 D3).

Raw PyTorch — no Lightning. Day-5 DDP will wrap the model in
``DistributedDataParallel`` and pass a ``DistributedSampler``-driven
DataLoader; nothing else about this class changes.

Inner loop per step:
    1. Pull next batch from _infinite_loader(dataloader).
    2. Move tokens to device; add singleton d_in axis -> (B, T, P, 1).
    3. Build (B, T_valid, P, H, d_in) target via build_forecast_targets.
    4. Forward under BF16 autocast (when precision='bf16' on CUDA).
    5. Slice prediction to T_valid positions; compute MSE.
    6. NaN guard: raise RuntimeError if loss is not finite.
    7. Backward, optional grad-clip, optimizer.step, optional scheduler.step.
    8. Append {step, loss, lr} to stdout (every log_every) + JSONL.
    9. Optional checkpoint every ckpt_every steps.

BF16 does NOT require GradScaler — that's FP16 only. CPU precision='fp32'
disables autocast entirely (PyTorch's BF16 CPU support is incomplete and
not needed for the n_layers=0 trainer test).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader

from boldcast.training.loss import build_forecast_targets, forecasting_loss
from boldcast.training.utils import JsonlLogger, save_checkpoint


def _infinite_loader(
    dataloader: DataLoader[dict[str, torch.Tensor]] | Iterable[dict[str, torch.Tensor]],
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever, re-iterating ``dataloader`` each pass.

    Unlike ``itertools.cycle`` (which caches every yielded element on the
    first pass), this calls ``iter(dataloader)`` afresh each epoch, keeping
    memory flat across long runs. Note: a Day-5 DDP caller must explicitly
    call ``sampler.set_epoch(epoch)`` before each pass — this loop has no
    epoch counter and does NOT trigger ``DistributedSampler``'s re-seed.
    """
    while True:
        yield from dataloader


__all__ = ["Trainer"]

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
        self._logger: JsonlLogger | None = None
        if self.out_dir is not None:
            self._logger = JsonlLogger(self.out_dir / "loss_log.jsonl")

    def fit(
        self,
        dataloader: DataLoader[dict[str, torch.Tensor]] | Iterable[dict[str, torch.Tensor]],
        max_steps: int,
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

        Returns
        -------
        dict[str, list[float]]
            Keys ``"step"``, ``"loss"``, ``"lr"``; each list has length
            ``max_steps``.
        """
        self.model.train()
        history: dict[str, list[float]] = {"step": [], "loss": [], "lr": []}
        data_iter = _infinite_loader(dataloader)
        try:
            for step in range(max_steps):
                batch = next(data_iter)
                loss_value, lr_value = self._train_step(batch, step)
                history["step"].append(float(step))
                history["loss"].append(loss_value)
                history["lr"].append(lr_value)
                if step % self.log_every == 0 or step == max_steps - 1:
                    print(
                        f"[trainer] step={step:>5d}  "
                        f"loss={loss_value:.6f}  lr={lr_value:.2e}"
                    )
                if (
                    self.ckpt_every is not None
                    and self.out_dir is not None
                    and (step + 1) % self.ckpt_every == 0
                ):
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        step=step + 1,
                        path=self.out_dir / f"ckpt_step{step + 1}.pt",
                    )
        finally:
            if self._logger is not None:
                self._logger.close()
        return history

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

        loss.backward()  # type: ignore[no-untyped-call]
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
        loss_value = float(loss.item())
        if self._logger is not None:
            self._logger.write({
                "step": step,
                "loss": loss_value,
                "lr": lr,
                "grad_norm": grad_norm,
            })
        return loss_value, lr
