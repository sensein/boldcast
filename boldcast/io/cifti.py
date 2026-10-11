"""CIFTI I/O — project-side wrapper.

Re-exports from :mod:`boldcast._upstream.cifti_io`. When the upstream
module lands in ``nobrainer.io``, this file's imports flip to
``from nobrainer.io import ...``.
"""

from boldcast._upstream.cifti_io import (
    cortex_grayordinate_indices,
    extract_cortex_grayordinates,
    load_dtseries,
    load_gifti_surface,
    save_dtseries,
)

__all__ = [
    "cortex_grayordinate_indices",
    "extract_cortex_grayordinates",
    "load_dtseries",
    "load_gifti_surface",
    "save_dtseries",
]
