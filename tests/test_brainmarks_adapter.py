"""Tests for boldcast.eval.brainmarks_adapter — pure-core module."""

import numpy as np
import pytest
import torch
from boldcast.eval.brainmarks_adapter import (
    BOLDcastTransform,
    cortex_slice,
    patch_tokens,
    pool_embeddings,
)

# ---------------------------------------------------------------------------
# Unit 1 — cortex_slice
# ---------------------------------------------------------------------------


def test_cortex_slice_identity() -> None:
    arr = np.arange(3 * 10, dtype=np.float32).reshape(3, 10)
    idx = np.arange(6)
    out = cortex_slice(arr, idx)
    assert out.shape == (3, 6)
    np.testing.assert_array_equal(out, arr[:, :6])


def test_cortex_slice_permutation() -> None:
    arr = np.arange(2 * 5, dtype=np.float32).reshape(2, 5)
    idx = np.array([4, 0, 2])
    out = cortex_slice(arr, idx)
    np.testing.assert_array_equal(out, arr[:, [4, 0, 2]])


def test_cortex_slice_rejects_out_of_range() -> None:
    arr = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        cortex_slice(arr, np.array([0, 9]))


def test_cortex_slice_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        cortex_slice(np.zeros(4, dtype=np.float32), np.array([0]))


def test_cortex_slice_empty_index_returns_zero_width() -> None:
    arr = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    out = cortex_slice(arr, np.array([], dtype=int))
    assert out.shape == (3, 0)


# ---------------------------------------------------------------------------
# Unit 2 — patch_tokens
# ---------------------------------------------------------------------------


def test_patch_tokens_means() -> None:
    cortex = np.array(
        [[1.0, 3.0, 10.0, 20.0], [2.0, 4.0, 30.0, 50.0]], dtype=np.float32
    )
    assignment = np.array([0, 0, 1, 1])
    out = patch_tokens(cortex, assignment, n_patches=2)
    assert tuple(out.shape) == (2, 2)
    expected = torch.tensor([[2.0, 15.0], [3.0, 40.0]])
    torch.testing.assert_close(out, expected)


# ---------------------------------------------------------------------------
# Unit 3 — pool_embeddings
# ---------------------------------------------------------------------------


def test_pool_embeddings_trait_shape_and_values() -> None:
    h = torch.ones(1, 2, 3, 4)
    cls = pool_embeddings(h, mode="trait")
    assert tuple(cls.shape) == (1, 1, 4)
    torch.testing.assert_close(cls, torch.ones(1, 1, 4))


def test_pool_embeddings_state_keeps_time() -> None:
    h = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 2, 3, 4)
    patch = pool_embeddings(h, mode="state")
    assert tuple(patch.shape) == (1, 2, 4)
    torch.testing.assert_close(patch, h.mean(dim=2))


def test_pool_embeddings_rejects_bad_mode() -> None:
    with pytest.raises(ValueError):
        pool_embeddings(torch.ones(1, 2, 3, 4), mode="nonsense")


def test_pool_embeddings_rejects_wrong_rank() -> None:
    with pytest.raises(ValueError):
        pool_embeddings(torch.ones(2, 3, 4), mode="trait")  # 3D, not 4D


# ---------------------------------------------------------------------------
# Unit 4 — BOLDcastTransform
# ---------------------------------------------------------------------------


def test_transform_end_to_end_small() -> None:
    bold = np.array(
        [[0.0, 1.0, 2.0, 3.0, 99.0], [4.0, 5.0, 6.0, 7.0, 88.0]], dtype=np.float32
    )
    cortex_index = np.array([0, 1, 2, 3])
    assignment = np.array([0, 0, 1, 1])
    tf = BOLDcastTransform(
        cortex_index=cortex_index, patch_assignment=assignment, n_patches=2
    )
    sample = {"bold": bold}
    out = tf(sample)
    assert "tokens" not in sample  # input dict must not be mutated
    assert "tokens" in out
    assert tuple(out["tokens"].shape) == (2, 2)
    expected = torch.tensor([[0.5, 2.5], [4.5, 6.5]])
    torch.testing.assert_close(out["tokens"], expected)


def test_transform_windowing_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        BOLDcastTransform(
            cortex_index=np.array([0]),
            patch_assignment=np.array([0]),
            n_patches=1,
            window=256,
        )


# ---------------------------------------------------------------------------
# GPU / micromamba-only — deselected by default addopts (-m 'not gpu')
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_wrapper_forward_smoke() -> None:
    """Deferred to a GPU/micromamba compute node: needs brainmarks + mamba-ssm.
    Lazy-imports so collection still succeeds in the dev venv."""
    pytest.importorskip("brainmarks")
    pytest.importorskip("mamba_ssm")
    from brainmarks_plugin.boldcast_register import BOLDcastBrainmarks  # noqa: F401
    # Full forward smoke is run manually on the compute node against a real
    # checkpoint + caches; logic = BOLDcastDemo.embed (tested elsewhere) +
    # pool_embeddings (tested above).
