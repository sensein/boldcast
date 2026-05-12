"""Optimizer + scheduler factories for Day-4 training (ADR 0005 D3).

``build_optimizer`` returns AdamW with the standard transformer-style param
group split (weight decay on 2D+ tensors, none on biases and LayerNorm).
``build_scheduler`` returns ``None`` for ``schedule='constant'`` (Day-4
overfit, per ADR 0005 D6) or a linear-warmup→cosine ``SequentialLR`` for
``schedule='cosine'`` (Day-5 full training).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

__all__ = ["build_optimizer", "build_scheduler"]


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> torch.optim.AdamW:
    """AdamW with two param groups: weight decay on 2D+ tensors, zero decay
    on 1D tensors (biases, LayerNorm weight/bias).

    Parameters
    ----------
    model:
        The model whose parameters will be optimized.
    lr:
        Learning rate passed to AdamW.
    weight_decay:
        Weight-decay coefficient applied to the decay param group (2D+
        tensors). The no-decay group always receives 0.0.
    betas:
        AdamW ``(beta1, beta2)`` tuple.

    Returns
    -------
    torch.optim.AdamW
        Optimizer with two param groups.
    """
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for _name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=betas,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: Literal["constant", "cosine"],
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Return ``None`` for ``constant`` (constant LR, no step needed); return
    a linear-warmup→cosine ``SequentialLR`` for ``cosine`` (matches Day-5
    training config).

    Parameters
    ----------
    optimizer:
        The optimizer whose LR will be scheduled.
    schedule:
        ``'constant'`` — no scheduler (returns ``None``).
        ``'cosine'`` — linear warmup followed by cosine annealing.
    warmup_steps:
        Number of steps for linear warm-up (only used when
        ``schedule='cosine'``).
    max_steps:
        Total training steps (only used when ``schedule='cosine'``).

    Returns
    -------
    torch.optim.lr_scheduler.LRScheduler or None
        ``None`` for ``'constant'``; a ``SequentialLR`` for ``'cosine'``.

    Raises
    ------
    ValueError
        If ``schedule`` is not ``'constant'`` or ``'cosine'``.
    """
    if schedule == "constant":
        return None
    if schedule == "cosine":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / max(warmup_steps, 1),
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(max_steps - warmup_steps, 1),
            eta_min=0.0,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )
    raise ValueError(
        f"schedule must be 'constant' or 'cosine', got {schedule!r}"
    )
