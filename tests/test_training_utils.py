"""CPU unit tests for boldcast.training.utils."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from boldcast.training.utils import (
    JsonlLogger,
    heldout_decreased_by,
    save_checkpoint,
    seed_everything,
)
from torch import nn


def test_seed_everything_is_deterministic_for_torch_randn() -> None:
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_seed_everything_different_seeds_differ() -> None:
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(456)
    b = torch.randn(5)
    assert not torch.equal(a, b)


def test_jsonl_logger_writes_one_line_per_record(tmp_path: Path) -> None:
    log = JsonlLogger(tmp_path / "log.jsonl")
    log.write({"step": 0, "loss": 1.5, "lr": 3e-4})
    log.write({"step": 1, "loss": 0.7, "lr": 3e-4})
    log.close()

    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"step": 0, "loss": 1.5, "lr": 3e-4}
    assert json.loads(lines[1]) == {"step": 1, "loss": 0.7, "lr": 3e-4}


def test_jsonl_logger_appends_to_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log1 = JsonlLogger(path)
    log1.write({"step": 0, "loss": 1.0})
    log1.close()

    log2 = JsonlLogger(path)
    log2.write({"step": 1, "loss": 0.5})
    log2.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 2


def test_save_checkpoint_round_trip(tmp_path: Path) -> None:
    model = nn.Linear(4, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Take one step so the optimizer has state to save.
    x = torch.randn(3, 4)
    loss = model(x).pow(2).mean()
    loss.backward()
    opt.step()

    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(model, opt, step=42, path=ckpt_path)

    assert ckpt_path.exists()
    loaded = torch.load(ckpt_path, weights_only=False)
    assert loaded["step"] == 42
    assert "model" in loaded
    assert "optimizer" in loaded
    # Reload into a fresh model and assert weight equality.
    fresh = nn.Linear(4, 2)
    fresh.load_state_dict(loaded["model"])
    for p_orig, p_fresh in zip(model.parameters(), fresh.parameters()):
        assert torch.equal(p_orig, p_fresh)


def test_jsonl_logger_context_manager(tmp_path: Path) -> None:
    """`with JsonlLogger(...) as log:` writes the record and closes on exit."""
    path = tmp_path / "log.jsonl"
    with JsonlLogger(path) as log:
        log.write({"step": 0, "loss": 0.5})
    assert path.read_text().strip() != ""
    assert log._fh.closed


def test_save_checkpoint_creates_parent_dirs(tmp_path: Path) -> None:
    """save_checkpoint creates intermediate directories that don't exist."""
    model = nn.Linear(2, 1)
    opt = torch.optim.SGD(model.parameters(), lr=1e-2)
    nested = tmp_path / "a" / "b" / "c" / "ckpt.pt"
    assert not nested.parent.exists()
    save_checkpoint(model, opt, step=0, path=nested)
    assert nested.exists()


def test_heldout_decreased_by_strong_decrease() -> None:
    """Strong decrease (1.0 → 0.5) passes 30% threshold."""
    history = {"val_loss": [1.0, 1.0, 1.0, 0.5, 0.5, 0.5]}
    assert heldout_decreased_by(history, frac=0.30, window=3) is True


def test_heldout_decreased_by_flat_fails() -> None:
    """Flat trajectory fails."""
    history = {"val_loss": [1.0] * 6}
    assert heldout_decreased_by(history, frac=0.30, window=3) is False


def test_heldout_decreased_by_noisy_decreasing_passes() -> None:
    """Noisy decreasing 1.0→0.65 (35% drop) passes."""
    history = {"val_loss": [1.0, 1.05, 0.95, 0.70, 0.60, 0.65]}
    assert heldout_decreased_by(history, frac=0.30, window=3) is True


def test_heldout_decreased_by_noisy_flat_fails() -> None:
    """Noisy with spike but no real trend fails."""
    history = {"val_loss": [1.0, 1.2, 0.8, 1.1, 0.9, 1.0]}
    assert heldout_decreased_by(history, frac=0.30, window=3) is False


def test_heldout_decreased_by_insufficient_data() -> None:
    """Fewer than 2*window measurements → False."""
    history = {"val_loss": [1.0, 0.5, 0.3]}
    assert heldout_decreased_by(history, frac=0.30, window=3) is False


def test_heldout_decreased_by_empty_history() -> None:
    """Empty val_loss → False."""
    history = {"val_loss": []}
    assert heldout_decreased_by(history, frac=0.30, window=3) is False
