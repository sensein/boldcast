"""Unit tests for the Day-5 trivial-baseline computations.

The baselines compare the trained model's val MSE against three reference
predictors that require no learning:

- ``predict-zero``: ``pred = 0`` everywhere
- ``predict-input``: ``pred[:, t, p, h, :] = tokens[:, t, p, :]`` (temporal
  persistence — "the next/future TR equals the current TR")
- ``predict-window-mean``: ``pred[:, t, p, h, :] = tokens[:, :T_valid, p, :].mean(dim=1)``
  (constant per-window per-channel mean)

We pin the math on a synthetic tokens tensor whose values are arange(0, T)
broadcast over P and d_in, so all three baseline losses have closed-form
expected values.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from boldcast.eval.baselines import (
    compute_trivial_baselines,
    predict_input_loss,
    predict_window_mean_loss,
    predict_zero_loss,
)
from boldcast.training.loss import build_forecast_targets, forecasting_loss

HORIZONS = (1, 5)


def _synth_tokens() -> torch.Tensor:
    """tokens[b=0, t, p, c] = t, shape (1, 8, 4, 1)."""
    t_vals = torch.arange(8, dtype=torch.float32)
    return t_vals.view(1, 8, 1, 1).expand(1, 8, 4, 1).contiguous()


def test_predict_zero_matches_mean_target_squared() -> None:
    tokens = _synth_tokens()
    targets = build_forecast_targets(tokens, HORIZONS)  # (1, 3, 4, 2, 1)
    # h=1 target = tokens[:, 1:4] -> values [1, 2, 3]
    # h=5 target = tokens[:, 5:8] -> values [5, 6, 7]
    # mean(target**2) over (B, T_valid, P, H, d_in) = (1+4+9+25+36+49)/6 = 20.6667
    expected = (1 + 4 + 9 + 25 + 36 + 49) / 6
    got = predict_zero_loss(tokens, HORIZONS).item()
    assert abs(got - expected) < 1e-5, (got, expected)
    # Cross-check via the production loss function.
    direct = forecasting_loss(torch.zeros_like(targets), targets).item()
    assert abs(got - direct) < 1e-6


def test_predict_input_matches_temporal_persistence() -> None:
    tokens = _synth_tokens()
    # pred = tokens[:, :T_valid] broadcast over H, T_valid=3
    # h=1: pred-target = [0-1, 1-2, 2-3] = [-1,-1,-1] -> MSE 1
    # h=5: pred-target = [0-5, 1-6, 2-7] = [-5,-5,-5] -> MSE 25
    # uniform-weight mean over horizons + T_valid + P + d_in: (1*3*4*1 + 25*3*4*1)/(3*4*2*1) = 13
    expected = (1 + 25) / 2
    got = predict_input_loss(tokens, HORIZONS).item()
    assert abs(got - expected) < 1e-5, (got, expected)


def test_predict_window_mean_matches_constant_window_mean() -> None:
    tokens = _synth_tokens()
    # Window mean over T_valid=3 first TRs: tokens[:, :3] = [0,1,2] -> mean 1.0
    # h=1: pred-target = [1-1, 1-2, 1-3] = [0,-1,-2] -> MSE = (0+1+4)/3 = 5/3
    # h=5: pred-target = [1-5, 1-6, 1-7] = [-4,-5,-6] -> MSE = (16+25+36)/3 = 77/3
    # uniform-weight mean: (5/3 + 77/3)/2 = 82/6 = 13.6667
    expected = (5 / 3 + 77 / 3) / 2
    got = predict_window_mean_loss(tokens, HORIZONS).item()
    assert abs(got - expected) < 1e-5, (got, expected)


def test_baselines_handle_batch_dim() -> None:
    """Two batches with different content should give different per-batch losses
    averaged into the same scalar as separate calls — sanity check that the
    baselines reduce over (B, T_valid, P, H, d_in) uniformly."""
    tokens_a = _synth_tokens()
    tokens_b = _synth_tokens() * 2.0  # values [0, 2, 4, ..., 14]
    tokens_batched = torch.cat([tokens_a, tokens_b], dim=0)

    # predict-zero scales quadratically with input scale.
    # tokens_a: 20.6667 ; tokens_b: 4 * 20.6667 = 82.6667
    # batched mean: (20.6667 + 82.6667)/2 = 51.6667
    expected = (20.6666667 + 82.6666667) / 2
    got = predict_zero_loss(tokens_batched, HORIZONS).item()
    assert abs(got - expected) < 1e-4, (got, expected)


# ----------------------------------------------------------------------
# compute_trivial_baselines (iteration + reduction)
# ----------------------------------------------------------------------


class _SyntheticLoader:
    """Iterable yielding fixed (B, T, P) batches as ``{'tokens': ...}`` dicts.

    Used in lieu of torch.utils.data.DataLoader so tests can run on CPU
    without pulling HCP data through the loader path.
    """

    def __init__(self, batches: list[torch.Tensor]) -> None:
        self.batches = batches

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for tokens in self.batches:
            yield {"tokens": tokens}


def test_compute_trivial_baselines_returns_expected_keys() -> None:
    """Without a model, returns exactly the three baseline keys."""
    tokens = _synth_tokens().squeeze(-1)  # (B, T, P)
    loader = _SyntheticLoader([tokens])
    result = compute_trivial_baselines(
        loader, HORIZONS, torch.device("cpu"), model=None
    )
    assert set(result.keys()) == {"zero", "input", "window_mean"}
    assert all(isinstance(v, float) for v in result.values())


def test_compute_trivial_baselines_matches_primitives_on_single_batch() -> None:
    """One-batch loader: returned values equal the primitive function results."""
    tokens = _synth_tokens().squeeze(-1)  # (B, T, P) — loader path will unsqueeze
    loader = _SyntheticLoader([tokens])
    result = compute_trivial_baselines(
        loader, HORIZONS, torch.device("cpu"), model=None
    )
    # Primitives expect (B, T, P, d_in); loader adds the d_in axis via unsqueeze(-1).
    tokens_full = tokens.unsqueeze(-1)
    assert abs(result["zero"] - predict_zero_loss(tokens_full, HORIZONS).item()) < 1e-5
    assert abs(result["input"] - predict_input_loss(tokens_full, HORIZONS).item()) < 1e-5
    assert abs(
        result["window_mean"] - predict_window_mean_loss(tokens_full, HORIZONS).item()
    ) < 1e-5


def test_compute_trivial_baselines_averages_over_batches() -> None:
    """Two batches: returned value is mean of per-batch losses."""
    tokens_a = _synth_tokens().squeeze(-1)
    tokens_b = (_synth_tokens() * 2.0).squeeze(-1)
    loader = _SyntheticLoader([tokens_a, tokens_b])
    result = compute_trivial_baselines(
        loader, HORIZONS, torch.device("cpu"), model=None
    )
    # predict-zero batched: ((20.6667 from tokens_a) + (82.6667 from tokens_b)) / 2 = 51.6667
    expected_zero = (20.6666667 + 82.6666667) / 2
    assert abs(result["zero"] - expected_zero) < 1e-4

    # predict-input: tokens_a gives 13.0; tokens_b (×2 scaling) gives 52.0; batched mean = 32.5
    expected_input = (13.0 + 52.0) / 2
    assert abs(result["input"] - expected_input) < 1e-4

    # predict-window-mean: tokens_a gives ~13.6667; tokens_b gives ~54.6667; batched mean = 34.1667
    expected_window_mean = ((5 / 3 + 77 / 3) / 2 + 4 * (5 / 3 + 77 / 3) / 2) / 2
    assert abs(result["window_mean"] - expected_window_mean) < 1e-4


def test_compute_trivial_baselines_empty_loader_avoids_division_by_zero() -> None:
    """Empty loader returns 0.0 for every baseline (max(n_batches, 1) guard)."""
    loader = _SyntheticLoader([])
    result = compute_trivial_baselines(
        loader, HORIZONS, torch.device("cpu"), model=None
    )
    assert result == {"zero": 0.0, "input": 0.0, "window_mean": 0.0}
