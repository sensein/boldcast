"""CPU unit tests for boldcast.training.optim factories."""

from __future__ import annotations

import pytest
import torch
from boldcast.training.optim import build_optimizer, build_scheduler
from torch import nn


def _toy_model() -> nn.Module:
    """A model with one 2D Linear (decay-eligible), one LayerNorm (1D),
    and an explicit standalone bias parameter."""
    return nn.Sequential(
        nn.Linear(4, 8, bias=True),     # weight (2D) -> decay, bias (1D) -> no_decay
        nn.LayerNorm(8),                # weight (1D), bias (1D) -> both no_decay
        nn.Linear(8, 2, bias=False),    # weight (2D) -> decay
    )


def test_build_optimizer_returns_adamw() -> None:
    opt = build_optimizer(
        _toy_model(), lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95)
    )
    assert isinstance(opt, torch.optim.AdamW)


def test_build_optimizer_splits_decay_and_no_decay_groups() -> None:
    model = _toy_model()
    opt = build_optimizer(
        model, lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95)
    )
    assert len(opt.param_groups) == 2
    decay_group = next(g for g in opt.param_groups if g["weight_decay"] > 0)
    no_decay_group = next(g for g in opt.param_groups if g["weight_decay"] == 0)
    assert decay_group["weight_decay"] == 0.05
    assert no_decay_group["weight_decay"] == 0.0

    # Decay group: 2D weights only.
    for p in decay_group["params"]:
        assert p.ndim >= 2
    # No-decay group: 1D params (biases, LayerNorm weight/bias).
    for p in no_decay_group["params"]:
        assert p.ndim < 2


def test_build_optimizer_count_matches_total_trainable() -> None:
    model = _toy_model()
    opt = build_optimizer(
        model, lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95)
    )
    n_in_opt = sum(len(g["params"]) for g in opt.param_groups)
    n_in_model = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_in_opt == n_in_model


def test_build_optimizer_passes_lr_and_betas() -> None:
    opt = build_optimizer(
        _toy_model(), lr=1e-3, weight_decay=0.0, betas=(0.8, 0.99)
    )
    for g in opt.param_groups:
        assert g["lr"] == 1e-3
        assert g["betas"] == (0.8, 0.99)


def test_build_scheduler_constant_returns_none() -> None:
    opt = build_optimizer(
        _toy_model(), lr=3e-4, weight_decay=0.0, betas=(0.9, 0.95)
    )
    sched = build_scheduler(
        opt, schedule="constant", warmup_steps=100, max_steps=1000
    )
    assert sched is None


def test_build_scheduler_cosine_returns_scheduler() -> None:
    opt = build_optimizer(
        _toy_model(), lr=3e-4, weight_decay=0.0, betas=(0.9, 0.95)
    )
    sched = build_scheduler(
        opt, schedule="cosine", warmup_steps=10, max_steps=100
    )
    assert isinstance(sched, torch.optim.lr_scheduler.LRScheduler)
    # Warmup: LR at step 0 < target after warmup steps.
    lr_step0 = opt.param_groups[0]["lr"]
    for _ in range(10):
        opt.step()
        sched.step()
    lr_post_warmup = opt.param_groups[0]["lr"]
    assert lr_post_warmup > lr_step0


def test_build_scheduler_rejects_unknown() -> None:
    opt = build_optimizer(
        _toy_model(), lr=3e-4, weight_decay=0.0, betas=(0.9, 0.95)
    )
    with pytest.raises(ValueError, match="schedule"):
        build_scheduler(
            opt, schedule="exponential", warmup_steps=10, max_steps=100
        )


def test_build_optimizer_excludes_frozen_params() -> None:
    """Params with requires_grad=False must not appear in any optimizer group."""
    model = _toy_model()
    # Freeze the first Linear's weight (a 2D param that would otherwise go to decay).
    first_linear = list(model.children())[0]
    first_linear.weight.requires_grad_(False)

    opt = build_optimizer(
        model, lr=3e-4, weight_decay=0.05, betas=(0.9, 0.95)
    )
    n_in_opt = sum(len(g["params"]) for g in opt.param_groups)
    n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
    assert n_in_opt == n_trainable

    # Verify the frozen param is not in any group.
    all_opt_params = [p for g in opt.param_groups for p in g["params"]]
    assert all(p is not first_linear.weight for p in all_opt_params)
