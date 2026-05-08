"""HCP resting-state Dataset with per-(subject, run) tokenized cache."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from torch.utils.data import Dataset

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
        raise NotImplementedError("Task 4 wires this up.")
