"""Tests for boldcast.training.ddp environment detection utilities.

Pure os.environ probing; no torch.distributed or GPU calls.
Uses pytest monkeypatch to control environment variables.
"""

from __future__ import annotations

import pytest
from boldcast.training.ddp import (
    get_local_rank,
    get_rank,
    get_world_size,
    is_distributed_run,
    is_rank_zero,
)


def test_non_distributed_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """With RANK/WORLD_SIZE/LOCAL_RANK all unset, return sensible defaults."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_distributed_run() is False
    assert get_rank() == 0
    assert get_world_size() == 1
    assert get_local_rank() == 0
    assert is_rank_zero() is True


def test_distributed_rank_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """With RANK=0, WORLD_SIZE=2, LOCAL_RANK=0, rank-zero checks work."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert is_distributed_run() is True
    assert get_rank() == 0
    assert get_world_size() == 2
    assert get_local_rank() == 0
    assert is_rank_zero() is True


def test_distributed_rank_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """With RANK=1, WORLD_SIZE=2, LOCAL_RANK=1, non-zero checks work."""
    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert is_distributed_run() is True
    assert get_rank() == 1
    assert get_world_size() == 2
    assert get_local_rank() == 1
    assert is_rank_zero() is False


def test_partial_env_missing_world_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only RANK set (no WORLD_SIZE) → is_distributed_run() False."""
    monkeypatch.setenv("RANK", "0")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_distributed_run() is False
    # Individual getters still return parsed values or defaults
    assert get_rank() == 0
    assert get_world_size() == 1  # default


def test_partial_env_missing_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only WORLD_SIZE set (no RANK) → is_distributed_run() False."""
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    assert is_distributed_run() is False
    # Individual getters still return parsed values or defaults
    assert get_rank() == 0  # default
    assert get_world_size() == 4


def test_multi_node_scenario(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-node: RANK=3, WORLD_SIZE=8, LOCAL_RANK=3 (rank 3 of 8)."""
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "8")
    monkeypatch.setenv("LOCAL_RANK", "3")

    assert is_distributed_run() is True
    assert get_rank() == 3
    assert get_world_size() == 8
    assert get_local_rank() == 3
    assert is_rank_zero() is False
