"""Forecasting targets + MSE loss for Day-4 training (ADR 0005 D4).

Two pure functions:

- ``build_forecast_targets`` slices ``(B, T, P, d_in)`` input tokens into
  ``(B, T_valid, P, H, d_in)`` targets, where ``T_valid = T - max(horizons)``
  and ``targets[..., i, :] = tokens[:, horizons[i]:horizons[i] + T_valid]``.
- ``forecasting_loss`` is ``F.mse_loss`` with default ``reduction='mean'``;
  uniform T_valid and d_in across horizons make mean reduction identically
  equal weight per horizon (methods.md prescription, no learned scalar).

Both are CPU-runnable and have no dependency on ``mamba_ssm``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

__all__ = ["build_forecast_targets", "forecasting_loss"]


def build_forecast_targets(
    tokens: torch.Tensor,
    horizons: Sequence[int],
) -> torch.Tensor:
    """Stack future-horizon target slices for non-autoregressive multi-horizon
    forecasting.

    Parameters
    ----------
    tokens
        ``(B, T, P, d_in)`` input window.
    horizons
        Positive integer offsets to predict. ``max(horizons)`` must be ``< T``.
        Duplicate values are allowed but will produce identical H slices; the
        caller is responsible for uniqueness if that matters.

    Returns
    -------
    targets
        ``(B, T_valid, P, H, d_in)`` where ``T_valid = T - max(horizons)``
        and ``H = len(horizons)``. ``targets[..., i, :]`` equals
        ``tokens[:, horizons[i]:horizons[i] + T_valid]``.

    Raises
    ------
    ValueError
        If ``horizons`` is empty, contains a non-positive value, or
        ``max(horizons) >= T``.
    """
    if len(horizons) == 0:
        raise ValueError("horizons must be non-empty")
    if any(h <= 0 for h in horizons):
        raise ValueError("horizons must be positive (got at least one h<=0)")
    max_h = max(horizons)
    t_full = tokens.shape[1]
    if max_h >= t_full:
        raise ValueError(f"max(horizons)={max_h} must be < T={t_full}")
    t_valid = t_full - max_h
    return torch.stack(
        [tokens[:, h:h + t_valid] for h in horizons], dim=3
    )


def forecasting_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE with ``reduction='mean'`` over all elements of ``(B, T_valid, P, H, d_in)``.

    Equal weight per element; with uniform T_valid and d_in across horizons,
    that is identically equal weight per horizon (methods.md, no learned
    balancing scalar).

    Parameters
    ----------
    pred
        Predicted tensor; shape must broadcast with ``target``.
    target
        Ground-truth tensor; shape must broadcast with ``pred``.

    Returns
    -------
    loss
        Scalar mean-squared error.
    """
    return F.mse_loss(pred, target)
