"""CPU unit tests for boldcast.training.loss.

build_forecast_targets is a pure tensor slice op; forecasting_loss is
F.mse_loss with reduction='mean'. Both are CPU-runnable under uv.
"""

from __future__ import annotations

import pytest
import torch
from boldcast.training.loss import build_forecast_targets, forecasting_loss


def test_build_forecast_targets_shape_for_two_horizons() -> None:
    tokens = torch.randn(2, 8, 3, 1)  # (B=2, T=8, P=3, d_in=1)
    out = build_forecast_targets(tokens, horizons=(1, 5))
    # T_valid = T - max(horizons) = 8 - 5 = 3
    assert out.shape == (2, 3, 3, 2, 1)
    assert out.dtype == tokens.dtype


def test_build_forecast_targets_values_match_manual_slices() -> None:
    tokens = torch.arange(24, dtype=torch.float32).view(1, 8, 3, 1)
    out = build_forecast_targets(tokens, horizons=(1, 5))
    # T_valid = 3 ; horizon 1 -> tokens[:, 1:4] ; horizon 5 -> tokens[:, 5:8]
    expected_h1 = tokens[:, 1:4]
    expected_h5 = tokens[:, 5:8]
    assert torch.equal(out[:, :, :, 0, :], expected_h1)
    assert torch.equal(out[:, :, :, 1, :], expected_h5)


def test_build_forecast_targets_single_horizon_keeps_h_axis() -> None:
    """ADR 0005 D2: H axis materialized even at H=1."""
    tokens = torch.randn(2, 6, 4, 1)
    out = build_forecast_targets(tokens, horizons=(1,))
    assert out.shape == (2, 5, 4, 1, 1)


def test_build_forecast_targets_rejects_empty_horizons() -> None:
    tokens = torch.randn(2, 8, 3, 1)
    with pytest.raises(ValueError, match="horizons must be non-empty"):
        build_forecast_targets(tokens, horizons=())


def test_build_forecast_targets_rejects_non_positive_horizon() -> None:
    tokens = torch.randn(2, 8, 3, 1)
    with pytest.raises(ValueError, match=r"horizons must be positive"):
        build_forecast_targets(tokens, horizons=(0, 5))
    with pytest.raises(ValueError, match=r"horizons must be positive"):
        build_forecast_targets(tokens, horizons=(1, -3))


def test_build_forecast_targets_rejects_max_horizon_ge_t() -> None:
    tokens = torch.randn(2, 8, 3, 1)
    with pytest.raises(ValueError, match=r"max\(horizons\)"):
        build_forecast_targets(tokens, horizons=(1, 8))  # 8 == T
    with pytest.raises(ValueError, match=r"max\(horizons\)"):
        build_forecast_targets(tokens, horizons=(1, 9))  # 9 > T


def test_build_forecast_targets_preserves_horizon_insertion_order() -> None:
    """H axis tracks input order, not sorted order."""
    tokens = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1)
    out = build_forecast_targets(tokens, horizons=(5, 1))  # reversed order
    # T_valid = 8 - 5 = 3; out[:, :, :, 0, :] -> h=5 -> tokens[:, 5:8]
    # out[:, :, :, 1, :] -> h=1 -> tokens[:, 1:4]
    assert torch.equal(out[:, :, :, 0, :], tokens[:, 5:8])
    assert torch.equal(out[:, :, :, 1, :], tokens[:, 1:4])


def test_forecasting_loss_zero_on_identical_inputs() -> None:
    x = torch.randn(2, 3, 4, 2, 1)
    assert forecasting_loss(x, x).item() == 0.0


def test_forecasting_loss_matches_hand_computed_mse() -> None:
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 0.0, 0.0])
    # MSE = mean([0, 4, 9]) = 13/3
    loss = forecasting_loss(pred, target)
    assert loss.item() == pytest.approx(13.0 / 3.0)


def test_forecasting_loss_preserves_grad() -> None:
    pred = torch.randn(2, 3, 4, 2, 1, requires_grad=True)
    target = torch.randn(2, 3, 4, 2, 1)
    loss = forecasting_loss(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
