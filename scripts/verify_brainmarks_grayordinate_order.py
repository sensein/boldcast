"""Verify BRAINMARKS ``fslr91k`` grayordinate ordering vs BOLDcast's tokenizer.

**Why this exists.** The BRAINMARKS adapter (see schist-vault note
``research/2026-05-30-brain-fm-competitive-landscape-for-boldcast`` →
"BRAINMARKS adapter plan") feeds the
``fslr91k`` reader's raw ``(T, 91282)`` grayordinate array into BOLDcast's
cortical ``Patcher``. That only produces correct tokens if the first
``V_cortex = 59412`` columns BOLDcast expects (``CIFTI_STRUCTURE_CORTEX_LEFT``
grayordinates, then ``CORTEX_RIGHT``, vertex-ascending, medial wall excluded)
line up column-for-column with what ``fslr91k`` delivers.

Code inspection (BOLDcast ``boldcast/_upstream/cifti_io.py`` +
BRAINMARKS ``src/brainmarks/{readers,nisc}.py``) says they match for the
standard HCP dense grayordinate template: ``fslr91k_reader`` returns the raw
CIFTI array with no reordering, and BRAINMARKS' own ``get_cifti_surf_indices``
concatenates LH-then-RH exactly as BOLDcast does. **The only thing not
verifiable from code is that the actual stored files use the standard
structure ordering** (CORTEX_LEFT first, CORTEX_RIGHT second, contiguous).
This script asserts that against a real CIFTI header and emits the static
``cortex_index`` vector the wrapper will use.

**DUA note (read before running).** Requires access under the WU-Minn HCP
Data Use Agreement.
The script reads a CIFTI *header* (structure names, grayordinate offsets,
mesh-vertex index arrays) and prints **only structural metadata** — names,
counts, and index ranges. It NEVER prints, returns, or writes BOLD signal
values. It is safe to run on an HCP dtseries you hold under the DUA, or on a
BRAINMARKS ``fslr91k`` sample once dataset access lands.

Usage
-----
    python scripts/verify_brainmarks_grayordinate_order.py \
        --cifti /path/to/standard_91k.dtseries.nii \
        --out   cache/brainmarks_cortex_index.npz

Exit code 0 = PASS (ordering matches; index vector written). Non-zero = the
adapter needs the emitted permutation vector (still written) rather than a
plain ``[:, :59412]`` slice — read the printed diagnosis.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse BOLDcast's exact header-parsing + cortex-ordering logic so this test
# checks the *same* code the dataloader uses, not a re-implementation.
from boldcast.io.cifti import (
    cortex_grayordinate_indices,
    load_dtseries,
)

V_CORTEX_LH_STD = 29696
V_CORTEX_RH_STD = 29716
V_CORTEX_STD = V_CORTEX_LH_STD + V_CORTEX_RH_STD  # 59412
V_GRAYORD_STD = 91282


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cifti",
        required=True,
        help="A CIFTI on the standard 91k grayordinate space "
        "(HCP dtseries, or a BRAINMARKS fslr91k sample).",
    )
    ap.add_argument(
        "--out",
        default="cache/brainmarks_cortex_index.npz",
        help="Where to write the static cortex_index vector + metadata.",
    )
    args = ap.parse_args()

    # Header only — load_dtseries reads get_fdata(), but we use only the
    # brain-model axis metadata below and never inspect/print the BOLD array.
    _data, header = load_dtseries(args.cifti)

    n_grayord = int(header["n_grayordinates"])
    bm_names = [bm["name"] for bm in header["brain_models"]]
    print(f"[info] grayordinates in file: {n_grayord}")
    print(f"[info] brain-model structures ({len(bm_names)}):")
    for bm in header["brain_models"]:
        slc = bm["slice"]
        print(f"         {bm['name']:<34} cols [{slc.start}:{slc.stop}]  n={slc.stop - slc.start}")

    problems: list[str] = []
    if n_grayord != V_GRAYORD_STD:
        problems.append(
            f"grayordinate count {n_grayord} != standard {V_GRAYORD_STD} "
            "— this is not the standard 91k space; the wrapper's slice must "
            "be recomputed for this space."
        )

    # BOLDcast cortex order: CORTEX_LEFT slice then CORTEX_RIGHT slice.
    lh = next(bm for bm in header["brain_models"] if bm["name"] == "CIFTI_STRUCTURE_CORTEX_LEFT")
    rh = next(bm for bm in header["brain_models"] if bm["name"] == "CIFTI_STRUCTURE_CORTEX_RIGHT")
    lh_cols = np.arange(lh["slice"].start, lh["slice"].stop, dtype=np.int64)
    rh_cols = np.arange(rh["slice"].start, rh["slice"].stop, dtype=np.int64)
    cortex_index = np.concatenate([lh_cols, rh_cols])  # fslr91k col -> BOLDcast col

    n_lh, n_rh = lh_cols.size, rh_cols.size
    print(f"[info] CORTEX_LEFT  cols: [{lh_cols[0]}:{lh_cols[-1] + 1}]  n={n_lh}")
    print(f"[info] CORTEX_RIGHT cols: [{rh_cols[0]}:{rh_cols[-1] + 1}]  n={n_rh}")

    if n_lh != V_CORTEX_LH_STD:
        problems.append(f"LH cortex n={n_lh} != standard {V_CORTEX_LH_STD}")
    if n_rh != V_CORTEX_RH_STD:
        problems.append(f"RH cortex n={n_rh} != standard {V_CORTEX_RH_STD}")

    # Sanity: BOLDcast's own cortex_grayordinate_indices must agree on counts
    # (it returns mesh-vertex arrays, length == grayordinate count per hemi).
    lh_vtx, rh_vtx = cortex_grayordinate_indices(header)
    if lh_vtx.shape[0] != n_lh or rh_vtx.shape[0] != n_rh:
        problems.append(
            "cortex_grayordinate_indices disagrees with brain-model slices "
            f"(vtx {lh_vtx.shape[0]}/{rh_vtx.shape[0]} vs "
            f"slice {n_lh}/{n_rh})"
        )

    is_direct_slice = bool(
        cortex_index.size == V_CORTEX_STD and np.array_equal(cortex_index, np.arange(V_CORTEX_STD))
    )

    # Always write the index vector — the wrapper loads it regardless, so a
    # non-standard ordering is handled transparently via fancy-indexing.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out),
        cortex_index=cortex_index,
        n_grayordinates=np.asarray(n_grayord),
        n_lh_cortex=np.asarray(n_lh),
        n_rh_cortex=np.asarray(n_rh),
        is_direct_slice=np.asarray(is_direct_slice),
        source_cifti=np.asarray(Path(args.cifti).name),
    )
    print(f"[write] {out}  (cortex_index: {cortex_index.size} columns)")

    if problems:
        print("\n[FAIL] ordering is non-standard:")
        for p in problems:
            print(f"   - {p}")
        print(
            "\n   The wrapper MUST use the emitted cortex_index vector "
            "(fancy-index), not a [:, :59412] slice."
        )
        return 1

    if is_direct_slice:
        print(
            "\n[PASS] CORTEX_LEFT+RIGHT occupy the first 59412 columns "
            "contiguously.\n"
            "   Wrapper may use a plain  fslr91k[:, :59412]  slice "
            "(equivalent to the saved index)."
        )
    else:
        print(
            "\n[PASS] cortex grayordinate counts match standard, but cortex "
            "columns are NOT the leading 59412.\n"
            "   Wrapper must fancy-index with the saved cortex_index vector."
        )
    print(
        "\n   Follow-up: once BRAINMARKS dataset access lands, re-run this on "
        "an actual fslr91k sample to confirm the stored files share this "
        "ordering."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
