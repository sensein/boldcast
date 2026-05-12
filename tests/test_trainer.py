"""CPU end-to-end test for boldcast.training.trainer.Trainer.

Uses BOLDcastDemo(n_layers=0) so the test path does not import mamba_ssm
and runs under uv on the login node.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from boldcast.models.boldcast_demo import BOLDcastDemo
from boldcast.training.optim import build_optimizer
from boldcast.training.trainer import Trainer
from torch import nn
from torch.utils.data import DataLoader, Dataset


def _identity_adjacency(n_patches: int, k: int) -> torch.Tensor:
    adj = np.zeros((n_patches, k), dtype=np.int64)
    for i in range(n_patches):
        adj[i, 0] = i
        for j in range(1, k):
            adj[i, j] = (i + j) % n_patches
    return torch.from_numpy(adj)


class _SyntheticWindows(Dataset[dict[str, torch.Tensor]]):
    """Synthetic dataset that yields fixed (T, P) AR(1) windows.

    Each item is a dict with key ``tokens`` of shape (T, P), matching
    HCPRestingDataset's __getitem__ contract.

    Uses an AR(1) process (rho=0.9) so that successive time-steps are
    strongly correlated.  i.i.d. random data would make the MSE floor
    equal to var(target) ≈ 1.0 regardless of model capacity, preventing
    the overfit test from reaching < 0.5 * initial in 50 steps."""

    def __init__(self, n: int, T: int, P: int, rho: float = 0.9, seed: int = 0) -> None:
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
) -> tuple[BOLDcastDemo, torch.optim.AdamW, DataLoader[dict[str, torch.Tensor]], Trainer]:
    n_patches, k = 8, 4
    adj = _identity_adjacency(n_patches, k)
    model = BOLDcastDemo(
        d_in=1, d_model=8, n_layers=0,
        n_patches=n_patches, k_neighbors=k,
        adjacency=adj, horizons=horizons,
    )
    opt = build_optimizer(model, lr=3e-3, weight_decay=0.0, betas=(0.9, 0.95))
    ds = _SyntheticWindows(n=4, T=16, P=n_patches, seed=0)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    trainer = Trainer(
        model=model,
        optimizer=opt,
        scheduler=None,
        device=torch.device("cpu"),
        horizons=horizons,
        grad_clip_norm=1.0,
        precision="fp32",   # CPU has no BF16 autocast support; trainer must respect this
        log_every=10,
        out_dir=tmp_path,
    )
    return model, opt, loader, trainer


def test_trainer_overfit_reduces_loss_on_cpu(tmp_path: Path) -> None:
    """Trainer.fit on a 4-window batch with a 0-layer model should
    measurably reduce loss on a tiny synthetic task.

    The 0.7x threshold is calibrated for white-noise targets at horizon=5,
    where the irreducible noise floor (var of the noise) constrains how
    far MSE can drop. The point of this test is to prove the gradient
    path is alive, not to verify convergence to zero — the GPU overfit
    on real data (scripts/day4_overfit.py) gates on the < 1% threshold.
    """
    _, _, loader, trainer = _build_tiny_setup(tmp_path)
    history = trainer.fit(loader, max_steps=50)
    assert len(history["loss"]) == 50
    assert history["loss"][-1] < 0.7 * history["loss"][0], (
        f"loss did not decrease enough: initial={history['loss'][0]:.4f}, "
        f"final={history['loss'][-1]:.4f}, ratio="
        f"{history['loss'][-1]/history['loss'][0]:.4f} (target < 0.7)"
    )


def test_trainer_writes_jsonl_log(tmp_path: Path) -> None:
    _, _, loader, trainer = _build_tiny_setup(tmp_path)
    trainer.fit(loader, max_steps=10)
    log_path = tmp_path / "loss_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().splitlines()
    assert len(lines) == 10


def test_trainer_raises_on_non_finite_loss(tmp_path: Path) -> None:
    """A forward hook that returns NaN should make the trainer raise."""
    model, _, loader, trainer = _build_tiny_setup(tmp_path)

    def nan_hook(_module: nn.Module, _input: tuple[torch.Tensor, ...],
                 output: torch.Tensor) -> torch.Tensor:
        return output + float("nan")

    model.register_forward_hook(nan_hook)
    with pytest.raises(RuntimeError, match="non-finite"):
        trainer.fit(loader, max_steps=1)


def test_trainer_single_horizon_runs_to_completion(tmp_path: Path) -> None:
    """H=1 path (H axis still present per ADR 0005 D2) must not crash."""
    _, _, loader, trainer = _build_tiny_setup(tmp_path, horizons=(1,))
    history = trainer.fit(loader, max_steps=5)
    assert len(history["loss"]) == 5
