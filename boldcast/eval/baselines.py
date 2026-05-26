"""Trivial-forecasting baselines for held-out val MSE comparison.

Used by the Day-5/Day-6 training scripts as the comparison target for
the baseline-relative acceptance gate (boldcast.training.utils.
beats_best_baseline), and by scripts/day5_baseline_eval.py as a
standalone retrospective diagnostic.

Primitives:

- :func:`predict_zero_loss`         pred = 0 everywhere
- :func:`predict_input_loss`        pred[:, t, p, h, :] = tokens[:, t, p, :]
                                    (temporal persistence: "future TR = current TR")
- :func:`predict_window_mean_loss`  pred[:, t, p, h, :] = tokens[:, :T_valid, p, :].mean(dim=1)
                                    (constant per-window per-channel mean)

Iteration:

- :func:`compute_trivial_baselines` runs all three primitives (and
  optionally a trained model) over a ``val_loader`` and returns
  per-baseline mean MSE using the same ``total_loss / n_batches``
  reduction as :class:`boldcast.training.trainer.Trainer._eval`.

All four use the same :func:`boldcast.training.loss.forecasting_loss`
reduction (``mean`` over ``(B, T_valid, P, H, d_in)``) so the numbers
are directly comparable to ``val_loss`` in the trainer's JSONL log.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch

from boldcast.training.loss import build_forecast_targets, forecasting_loss

__all__ = [
    "compute_trivial_baselines",
    "predict_input_loss",
    "predict_window_mean_loss",
    "predict_zero_loss",
]


def predict_zero_loss(
    tokens: torch.Tensor,
    horizons: Sequence[int],
) -> torch.Tensor:
    """MSE of an all-zeros prediction against forecast targets."""
    targets = build_forecast_targets(tokens, horizons)
    return forecasting_loss(torch.zeros_like(targets), targets)


def predict_input_loss(
    tokens: torch.Tensor,
    horizons: Sequence[int],
) -> torch.Tensor:
    """MSE of ``pred[:, t, p, h, :] = tokens[:, t, p, :]`` against targets.

    The temporal-persistence ("identity") baseline: at every prediction
    position ``t`` and horizon ``h``, the predictor outputs the value of
    the input at position ``t`` (no shift). Horizon-1 against this
    baseline is the lag-1 autocorrelation residual; longer horizons
    accumulate drift.
    """
    targets = build_forecast_targets(tokens, horizons)
    t_valid = targets.shape[1]
    h = targets.shape[3]
    # tokens[:, :T_valid] shape (B, T_valid, P, d_in) -> (B, T_valid, P, H, d_in)
    pred = tokens[:, :t_valid].unsqueeze(3).expand(-1, -1, -1, h, -1)
    return forecasting_loss(pred, targets)


def predict_window_mean_loss(
    tokens: torch.Tensor,
    horizons: Sequence[int],
) -> torch.Tensor:
    """MSE of a per-window per-channel constant-mean predictor against targets.

    The mean is taken over the first ``T_valid`` TRs of the input (the
    prefix where output predictions are emitted), per patch and per
    channel. Same constant value broadcast across all output positions
    and horizons.
    """
    targets = build_forecast_targets(tokens, horizons)
    t_valid = targets.shape[1]
    h = targets.shape[3]
    # tokens[:, :T_valid] shape (B, T_valid, P, d_in) -> mean over time
    window_mean = tokens[:, :t_valid].mean(dim=1, keepdim=True)
    # (B, 1, P, d_in) -> (B, T_valid, P, H, d_in)
    pred = window_mean.unsqueeze(3).expand(-1, t_valid, -1, h, -1)
    return forecasting_loss(pred, targets)


def compute_trivial_baselines(
    val_loader: Iterable[dict[str, torch.Tensor]],
    horizons: Sequence[int],
    device: torch.device,
    model: torch.nn.Module | None = None,
) -> dict[str, float]:
    """Iterate ``val_loader`` once and return per-baseline mean MSE.

    Each batch contributes one scalar per baseline (computed as the
    ``forecasting_loss`` reduction over that batch); the return value
    averages those scalars across batches with equal weight — the
    ``total_loss / n_batches`` reduction matching
    :meth:`boldcast.training.trainer.Trainer._eval`.

    If ``model`` is provided, runs it under the same BF16 autocast on
    CUDA that :meth:`boldcast.training.trainer.Trainer._eval` uses, and
    returns its loss under the ``"model"`` key. Note that the three
    trivial-baseline primitives run in the input tensor's native dtype
    (typically FP32 on CUDA) — without autocast. In practice the
    FP32-vs-BF16 difference is well below any threshold that could flip
    the acceptance gate, but reader beware. Caller is responsible for
    ``model.to(device)``; this function sets ``model.eval()``.

    Output is sanitized: scalar losses only, no batch contents.

    Returns
    -------
    dict[str, float]
        Keys ``"zero"``, ``"input"``, ``"window_mean"`` always present;
        ``"model"`` present iff ``model is not None``. Key names match the
        JSON written to ``results/day5_train/baseline_eval.json`` by
        ``scripts/day5_baseline_eval.py``; do not rename without migrating
        that artifact.
    """
    totals: dict[str, float] = {"zero": 0.0, "input": 0.0, "window_mean": 0.0}
    if model is not None:
        totals["model"] = 0.0
    n_batches = 0

    if model is not None:
        model.eval()

    with torch.no_grad():
        for batch in val_loader:
            tokens = batch["tokens"].to(device).unsqueeze(-1)
            totals["zero"] += float(predict_zero_loss(tokens, horizons).item())
            totals["input"] += float(predict_input_loss(tokens, horizons).item())
            totals["window_mean"] += float(
                predict_window_mean_loss(tokens, horizons).item()
            )
            if model is not None:
                # Recomputed here for the model branch; primitives call this
                # internally too. Share if this becomes a bottleneck.
                targets = build_forecast_targets(tokens, horizons)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    pred_full = model(tokens)
                    pred = pred_full[:, : targets.shape[1]]
                    loss = forecasting_loss(pred, targets)
                totals["model"] += float(loss.item())
            n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}
