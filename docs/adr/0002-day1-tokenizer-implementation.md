# 0002. Day 1 CIFTI tokenizer — implementation choices

**Status:** accepted
**Date:** 2026-05-04

## Context

Day 1 of the demo plan ([`docs/10_day_plan.md`](../10_day_plan.md)) calls
for an atlas-free CIFTI tokenizer: load HCP `*_Atlas_MSMAll_hp2000_clean
.dtseries.nii`, partition the cortical surface into 1,024 geodesic
patches (per-hemisphere FPS, 512 + 512), and verify round-trip parity
between grayordinates and patch means.

The prose spec leaves three implementation-level choices unresolved:

1. **File layout.** The Day 1 plan writes file paths as
   `boldcast/io/cifti.py` and `boldcast/tokenize/geodesic.py`, but
   [`boldcast/_upstream/README.md`](../../boldcast/_upstream/README.md)
   says CIFTI I/O and the geodesic patcher must live under
   `boldcast/_upstream/` and be held to nobrainer-grade standards
   (no internal imports, full type hints, NumPy docstrings, isolated
   tests under `tests/_upstream/`).
2. **FPS algorithm.** The spec says geodesic FPS via
   `scipy.sparse.csgraph.dijkstra` and documents Euclidean-3D-coords
   FPS as a runtime fallback "if geodesic FPS is slow". A third option
   (heat-method geodesics via `potpourri3d` or `pygeodesic`) was raised
   but not committed.
3. **Test fixtures under HCP DUA.** Yibei holds the WU-Minn HCP Data Use
   Agreement; Claude does not. Per [`CLAUDE.md`](../../CLAUDE.md),
   Claude must never read `.dtseries.nii`, `.gii`, or
   `Restricted_*.csv` files. Tests still need to exercise CIFTI I/O,
   FPS, and round-trip end-to-end.

Upstream context: the `_upstream/README.md` lists `cifti_io.py →
nobrainer.io.cifti` and `geodesic_patcher.py → nobrainer.layers` as
target nobrainer modules. Inspection of the current nobrainer repo
(neuronets/nobrainer) shows it is 3D-volume / NIfTI-focused (MeshNet,
SegFormer3D for segmentation; a single `nobrainer/io.py`; `layers/`
contains dropout variants), with no CIFTI submodule and no
surface/mesh tokenization. Upstreaming would require adding new
submodules to a library that has never had them — plausible but not
a drop-in. We treat upstreaming as aspirational and keep the
`_upstream/` discipline as project-internal hygiene regardless.

## Decision

### 1. Layered file layout

Real implementation lives in `boldcast/_upstream/`; project-side files
re-export and add project glue (caching, config-aware paths):

| File | Role |
|---|---|
| `boldcast/_upstream/cifti_io.py` | `load_dtseries`, `save_dtseries`, `cortex_grayordinate_indices_from_header`. Self-contained, full type hints, NumPy-style docstrings. |
| `boldcast/_upstream/geodesic_patcher.py` | `precompute_patches(mesh_lh, mesh_rh, n_patches, seed, metric)` — per-hemisphere FPS, returns `(V_cortex,) int` patch assignment. Self-contained. |
| `boldcast/io/cifti.py` | Thin re-export from `_upstream.cifti_io`. |
| `boldcast/tokenize/geodesic.py` | Project-side wrapper: cache I/O (`cache/patches_fsLR_32k_n1024_seed0_geo.npz`), HCP mesh-path resolution, calls `_upstream.geodesic_patcher`. |
| `boldcast/tokenize/patcher.py` | `Patcher` class — mean-pool BOLD per patch per TR via scatter-mean. Project-style; not `_upstream` material. |
| `tests/_upstream/test_cifti_io.py` | Isolated unit tests for `_upstream/cifti_io.py`. Synthetic CIFTI fixtures. |
| `tests/_upstream/test_geodesic_patcher.py` | Isolated unit tests for `_upstream/geodesic_patcher.py`. Synthetic icosphere meshes. |
| `tests/test_round_trip.py` | Day-1 acceptance test: synthetic dtseries → patcher → de-patch → patcher reproduces patch means to floating-point precision. |
| `scripts/day1_validate_tokenizer.py` | Real-data validation. Loads HCP dtseries; **Yibei runs**, Claude never executes. |

This satisfies both the Day 1 plan's path expectations (via the
project-side re-exports) and the `_upstream/` discipline (via the
real implementations).

### 2. FPS algorithm: edge-graph Dijkstra (default), Euclidean-3D fallback, heat-method deferred

`precompute_patches` takes a `metric` parameter:

```python
def precompute_patches(
    mesh_lh: tuple[np.ndarray, np.ndarray],
    mesh_rh: tuple[np.ndarray, np.ndarray],
    n_patches: int = 1024,
    seed: int = 0,
    metric: Literal["geodesic_dijkstra", "euclidean3d"] = "geodesic_dijkstra",
) -> np.ndarray: ...
```

**Default (`geodesic_dijkstra`).** Edge-graph Dijkstra on the
hemisphere mesh, weighted by Euclidean distance between adjacent
vertices. FPS via incremental Dijkstra: maintain a `min_dist_to_S`
array, run one Dijkstra from each newly-picked source, update
`min_dist_to_S = np.minimum(min_dist_to_S, dist_from_new_source)`,
pick the next source as `argmax(min_dist_to_S)`. Uses
`scipy.sparse.csgraph.dijkstra` (C implementation, already a
dependency). Estimated cache-build cost: ~1–3 minutes per
hemisphere on the `32k_fs_LR` mesh (29.7k vertices, ~6 neighbors
average), ~3–5 minutes total. One-time, cached as `.npz`.

**Fallback (`euclidean3d`).** FPS in 3D Euclidean space on vertex
coordinates extracted from the surface `.gii` file. Vectorized in
numpy; sub-second runtime. Distances do not respect cortical sheet
geometry — patches may span sulcal walls — so this is a documented
weakening of the methodological claim, accepted only as an escape
hatch if the geodesic build blows up at runtime.

**Cache filename embeds the metric** so swapping does not silently
invalidate the cache: `cache/patches_fsLR_32k_n1024_seed0_geo.npz`
or `..._eu3d.npz`.

**Heat-method (deferred).** Heat-method geodesics
[Crane et al. 2013] via `potpourri3d` factor a Cholesky solver
once per mesh and run any number of single-source distance
queries in O(V) backsubstitution time. For 512 sources per
hemisphere this would reduce cache-build time from minutes to
~30 s total. Both edge-graph Dijkstra and heat-method are
graph-level *approximations* of the true polyhedral geodesic
distance (the exact MMP algorithm is much slower); neither is
methodologically more honest for FPS purposes. We defer
heat-method to the full project, where higher-resolution meshes
(post-`164k_fs_LR` or whole-brain inflated meshes) and multiple
parcellation builds during ablations would make Dijkstra's
runtime annoying. For the demo's one-shot, one-hemisphere-pair
build at `32k_fs_LR`, scipy's C Dijkstra is enough.

### 3. Test fixtures: synthetic, programmatic, no HCP

Tests construct fixtures in-process; no HCP file is ever read by
Claude or by automated CI:

- `tests/_upstream/conftest.py` provides:
  - `synthetic_dtseries`: a tiny `nibabel.cifti2.Cifti2Image` built
    in memory (e.g., `T = 10` TRs, `V = 100` grayordinates with a
    50/50 LH/RH split). Exercises the actual nibabel CIFTI API
    end-to-end.
  - `synthetic_mesh_lh`, `synthetic_mesh_rh`: small subdivided
    icospheres (~50 vertices each) as `(vertices: (V, 3) float,
    faces: (F, 3) int)` tuples.

- `tests/_upstream/test_cifti_io.py` writes the synthetic
  `Cifti2Image` to a tmp file, loads via `load_dtseries`, asserts
  shape, dtype, and brain-model round-trip.

- `tests/_upstream/test_geodesic_patcher.py` runs FPS on the
  synthetic mesh, asserts:
  1. patch count equals request,
  2. every vertex has exactly one patch assignment (partition
     property),
  3. no patch is empty,
  4. patch-size std < 30 % of mean (the demo-acceptance criterion,
     applied loosely on a regular toy mesh).

- `tests/test_round_trip.py` is the Day-1 acceptance integration
  test using both fixtures: `dtseries → Patcher → scatter back to
  grayordinates as patch-mean → Patcher` reproduces patch means
  to floating-point precision.

The HCP-data validation lives in `scripts/day1_validate_tokenizer
.py`. Yibei runs that script on a real subject; the script writes
metrics (mean / std vertices per patch, ~900-TR runtime,
round-trip residual) to `figures/day1_patches.png` and a JSON
companion. Claude never executes it.

## Consequences

- The `_upstream/` discipline costs us a thin re-export layer in
  `boldcast/io/` and `boldcast/tokenize/` but buys testable,
  upstreamable modules with zero project-internal imports. If the
  nobrainer maintainers accept the contribution post-demo, deletion
  is a one-line `__init__.py` swap.
- Geodesic-Dijkstra default + Euclidean-3D fallback gives us
  methodological correctness as the path of least resistance, with a
  release valve if the cache build proves intolerably slow on a
  cluster login node. The runtime `metric` param means the swap is
  a config flag, not a code change.
- Synthetic fixtures keep CI fully reproducible and HCP-DUA-clean.
  They cost ~50 lines of `conftest.py` but make the test suite run
  in <2 s, which matters for TDD inner loops.
- The full FPS-algorithm tradeoff is documented (both here and as a
  future-work note in
  [`docs/methods.md`](../methods.md) §"Atlas-Free CIFTI
  Tokenization"), so the heat-method upgrade path is on record for
  the full project.

## Alternatives considered

- **Flat layout with no `_upstream/` separation** — rejected: loses
  the test-isolation guarantee, makes a future nobrainer PR
  significantly harder to factor out, and we already paid the cost
  of creating the directory.
- **Heat-method (`potpourri3d`) as the default** — rejected for the
  demo: adds a new dependency for an O(minutes) one-time cost we eat
  once. Reasonable to revisit at full-project scale; documented as
  future work.
- **Sample HCP file as test fixture** — rejected: HCP "Q1" is itself
  DUA-gated; nibabel ships small `.dscalar.nii` test assets but no
  dtseries; brings external-fetch fragility and risks the DUA rule.
- **Mocking `nibabel.cifti2.load`** — rejected: tests internals of
  nibabel rather than our code, loses end-to-end round-trip
  verification.

## References

- [`docs/10_day_plan.md`](../10_day_plan.md) §"Day 1 — CIFTI tokenizer"
- [`docs/methods.md`](../methods.md) §"Atlas-Free CIFTI Tokenization"
- [`boldcast/_upstream/README.md`](../../boldcast/_upstream/README.md)
- [`CLAUDE.md`](../../CLAUDE.md) §"HCP Data Use Agreement"
- Crane, K., Weischedel, C., Wardetzky, M. (2013). Geodesics in heat.
  *ACM Trans. Graph.* 32(5).
