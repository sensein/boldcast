"""HCP resting-state Dataset with per-(subject, run) tokenized cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from boldcast.data.transforms import standardize_run
from boldcast.io.cifti import (
    extract_cortex_grayordinates,
    load_dtseries,
)
from boldcast.tokenize.patcher import Patcher

__all__ = ["HCPRestingDataset"]


def _short_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class HCPRestingDataset(Dataset[dict[str, Any]]):
    """Map-style PyTorch Dataset over (subject, run, window) triples.

    Parameters
    ----------
    subjects : list[str]
        HCP subject IDs (e.g. ``["100307", "115825"]``).
    runs : list[str]
        Run names matching ``dtseries_pattern`` placeholders
        (e.g. ``["rfMRI_REST1_7T_PA", "rfMRI_REST2_7T_AP"]``).
    dtseries_pattern : str
        Path template with ``{subject}`` and ``{run}`` placeholders.
    cache_dir : Path | str
        Where per-(subject, run) tokenized caches are written/read.
    patch_assignment : ndarray of shape ``(V_cortex,)``
        Cortex-grayordinate → patch ID mapping. Shared across all subjects.
    n_patches : int
        Total patch count (== ``patch_assignment.max() + 1``).
    window_size : int
        Number of TRs per window.
    stride : int
        TRs between consecutive window starts.
    subject_id_offset : int
        Added to each subject's index before being returned as ``subject_id``.
        Use 0 for train, ``n_train`` for heldout, so train and heldout never
        share an int.
    standardize_method : str
        Currently only ``"run_wise"`` is supported. Stored in the cache
        metadata for invalidation.

    Notes
    -----
    Window enumeration is eager: at ``__init__`` time we build a list of
    ``(subject_idx, run_idx, window_start)`` triples. ``__len__`` is the
    length of that list. The cached ``(T_full, P)`` tensor for each
    (subject, run) is materialized lazily on first ``__getitem__``.
    """

    def __init__(
        self,
        subjects: list[str],
        runs: list[str],
        dtseries_pattern: str,
        cache_dir: Path | str,
        patch_assignment: NDArray[np.integer[Any]],
        n_patches: int,
        window_size: int,
        stride: int,
        subject_id_offset: int = 0,
        standardize_method: str = "run_wise",
    ) -> None:
        super().__init__()
        if standardize_method != "run_wise":
            raise ValueError(
                f"unknown standardize_method {standardize_method!r}; "
                "only 'run_wise' supported"
            )
        if int(patch_assignment.max()) >= n_patches:
            raise ValueError(
                f"patch_assignment max {int(patch_assignment.max())} "
                f">= n_patches {n_patches}"
            )

        self.subjects = list(subjects)
        self.runs = list(runs)
        self.dtseries_pattern = dtseries_pattern
        self.cache_dir = Path(cache_dir)
        self.patch_assignment = np.asarray(patch_assignment, dtype=np.int64)
        self.n_patches = int(n_patches)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.subject_id_offset = int(subject_id_offset)
        self.standardize_method = standardize_method
        self._assignment_sha = _short_sha(self.patch_assignment.tobytes())
        self._windows = self._enumerate_windows()

    def _enumerate_windows(self) -> list[tuple[int, int, int]]:
        """Build the (subject_idx, run_idx, window_start) index list.

        Reads each dtseries header (cheap — header-only, not data) to learn
        the run's TR count, then slides the window.
        """
        import nibabel as nib  # local: avoid import at module load

        windows: list[tuple[int, int, int]] = []
        for s_idx, subject in enumerate(self.subjects):
            for r_idx, run in enumerate(self.runs):
                path = self.dtseries_pattern.format(subject=subject, run=run)
                img = nib.load(path)  # type: ignore[attr-defined]
                series_axis = img.header.get_axis(0)  # type: ignore[attr-defined]
                t_full = int(series_axis.size)
                if t_full < self.window_size:
                    raise ValueError(
                        f"{path}: T={t_full} < window_size={self.window_size}"
                    )
                for start in range(0, t_full - self.window_size + 1, self.stride):
                    windows.append((s_idx, r_idx, start))
        return windows

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        s_idx, r_idx, start = self._windows[idx]
        subject = self.subjects[s_idx]
        run = self.runs[r_idx]
        tokens_full = self._load_or_build_tokens(subject, run)
        end = start + self.window_size
        return {
            "tokens": torch.from_numpy(tokens_full[start:end]),
            "subject_id": s_idx + self.subject_id_offset,
            "run_id": r_idx,
            "window_start": int(start),
        }

    def _load_or_build_tokens(self, subject: str, run: str) -> NDArray[np.float32]:
        """Return ``(T_full, P) float32`` for one (subject, run), reading or
        writing the cache as needed. Cache mismatch raises."""
        dtseries_path = self.dtseries_pattern.format(subject=subject, run=run)
        cache_path = self.cache_dir / f"{subject}_{run}.npz"

        dtseries_sha = _short_sha(Path(dtseries_path).read_bytes())
        expected_meta: dict[str, int | str] = {
            "dtseries_sha": dtseries_sha,
            "assignment_sha": self._assignment_sha,
            "n_patches": self.n_patches,
            "standardize_method": self.standardize_method,
        }

        if cache_path.exists():
            loaded = np.load(cache_path, allow_pickle=False)
            cached_meta: dict[str, int | str] = {
                k: loaded[k].item() for k in expected_meta if k in loaded
            }
            if cached_meta != expected_meta:
                raise ValueError(
                    f"cache metadata mismatch at {cache_path}: "
                    f"requested {expected_meta}, cached {cached_meta}. "
                    "Delete the cache file to rebuild."
                )
            tokens: NDArray[np.float32] = loaded["tokens"]
            return tokens

        # Build path: load → cortex → standardize → patcher → cache.
        data, header = load_dtseries(dtseries_path)
        cortex = extract_cortex_grayordinates(data, header)
        cortex_std = standardize_run(cortex)

        patcher = Patcher(
            torch.from_numpy(self.patch_assignment),
            n_patches=self.n_patches,
        )
        tokens_t = patcher.forward(torch.from_numpy(cortex_std))
        tokens_arr: NDArray[np.float32] = tokens_t.numpy().astype(np.float32)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            str(cache_path),
            tokens=tokens_arr,
            **{k: np.asarray(v) for k, v in expected_meta.items()},  # type: ignore[arg-type]
        )
        return tokens_arr
