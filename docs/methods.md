# Methods

## Overview and Contributions

We propose an atlas-free, surface-based foundation model for joint stimulus–brain
latent state tracking from naturalistic fMRI. The model has three primary
technical contributions.

**(1) Atlas-free tokenization** of whole-brain activity directly on cortical
surface and subcortical/cerebellar grayordinates, replacing the parcellation-based
or voxel-grid representations used in prior fMRI foundation models
[BrainLM, Caro et al. 2024; SwiFT, Kim et al. 2024; Brain-JEPA, Dong et al. 2024;
NeuroSTORM]. Patches are obtained once via geodesic farthest-point sampling on
the cortical mesh, preserving the non-Euclidean geometry of the cortex without
committing to any single anatomical parcellation scheme.

**(2) A linear-time selective state-space (Mamba) backbone**
[Gu & Dao 2023; Dao & Gu 2024] enabling training over 256–512 TR contexts —
well beyond the ≤64 TR windows typical of attention-based fMRI models. This is
essential for capturing the multi-minute dynamics characteristic of narrative
comprehension and continuous task engagement [Baldassano et al. 2017;
Vidaurre et al. 2017].

**(3) Dynamic brain–stimulus alignment** via frozen high-rate stimulus features
(CLIP ViT-L/14) bridged to slow BOLD signals through a hemodynamic alignment
module. This generalizes prior static brain–stimulus alignment work
[MindEye2, Scotti et al. 2024; BrainCLIP, Liu et al.] to dynamic naturalistic
viewing without requiring per-subject decoder heads.

The model is trained in two phases: (i) brain-only forecasting on HCP 3T
resting-state and 7T movie-watching data, (ii) multimodal forecasting plus
contrastive brain–stimulus retrieval on HCP 7T and CNeuroMod
[Boyle et al. 2023]. We evaluate held-out-subject and held-out-stimulus
generalization, plus three downstream transfer tasks: subject fingerprinting,
phenotype/trait prediction, and HCP cognitive task decoding.

## Atlas-Free CIFTI Tokenization

"Atlas-free" here refers specifically to the absence of a parcellation atlas
(Schaefer, Glasser MMP, AAL, etc.) for tokenization. Inter-subject
registration to the HCP standard reference space is required and assumed:
inputs are dense CIFTI grayordinate scalars from the HCP minimal-preprocessing
pipeline `*_Atlas_MSMAll_hp2000_clean.dtseries.nii` files (multimodal surface
matching, FIX-ICA denoised, high-pass filtered at 2000s) on the `32k_fs_LR`
cortical mesh and MNI152 subcortical grid. Tokenization operates on this
grayordinate input; no ROI averaging is applied.

Each TR is represented as a fixed set of `P = 1,792` spatial tokens drawn from
the HCP CIFTI grayordinate space [Glasser et al. 2013]: `1,024` cortical
patches plus `768` subcortical/cerebellar clusters. Cortical patches are
obtained once via geodesic farthest-point sampling on the `32k_fs_LR` mesh,
yielding patches of approximately 30 vertices each that respect the
non-Euclidean geometry of the cortical sheet. Subcortical and cerebellar
grayordinates are partitioned by k-means in MNI coordinates (`k = 768`).
Per-token features are mean BOLD per patch per TR (scalar) after run-wise
standardization. Round-trip decoding from patches to grayordinates is verified
to numerical precision on all training subjects.

This design contrasts with parcellation-based tokenization
[e.g., Schaefer-400, Schaefer et al. 2018], which discards within-parcel
spatial structure, and with 4D voxel grids [SwiFT], which expend compute on
non-brain voxels. It also avoids commitment to any single anatomical
parcellation scheme — a known source of methodological variability in dynamic
functional connectivity research.

## Spatial Mixing and Subject Conditioning

Tokens interact spatially via local kNN attention with `k = 8` over precomputed
cortical adjacency, interleaved between Mamba blocks. This yields an explicit,
neuroanatomically grounded inductive bias and avoids the `O(P²)` cost of dense
attention over `P = 1,792` tokens. Hybrid SSM+attention designs
[Jamba, Lieber et al. 2024; Hymba] have established that interleaving local
attention with state-space layers improves long-context modeling; our spatial
kNN attention serves the analogous role of providing global spatial mixing while
Mamba handles temporal evolution.

Subject anatomy enters via FiLM modulation [Perez et al. 2018] of token
features, parameterized by a per-subject vector of cortical thickness, surface
area, sulcal depth, and myelin (T1w/T2w ratio) extracted from FreeSurfer outputs
provided with HCP and CNeuroMod. This conditioning encodes the principle that
functional dynamics are shaped by structural morphology and provides a route to
cross-subject application without learned subject embeddings (which by
construction do not transfer to held-out subjects).

## Long-Context Mamba Backbone

The temporal core is a stack of 6 causal Mamba blocks (`d_model = 256`, default
config; `d_model = 512`, 12 layers, scaled config) operating on the per-TR
token sequence. Mamba's `O(N)` scaling in sequence length allows training at
256–512 TR contexts (~3–6 minutes of continuous viewing) at memory comparable
to attention-based models trained at 64 TR. We use truncated backpropagation
through time at 256 TRs (128 TR fallback) with explicit detachment, and BF16
mixed-precision training.

## Stimulus Stream and Hemodynamic Alignment

For multimodal training, naturalistic stimuli are encoded by frozen CLIP
ViT-L/14 [Radford et al. 2021], with embeddings precomputed and cached per
stimulus. A small learned MLP projects CLIP features to the model's latent
dimension. Alignment to BOLD is handled by a two-stage hemodynamic module:
(i) convolution with a canonical SPM double-gamma HRF, (ii) a learned 1D
residual lag/blur filter (FIR, length 7 TR) that adjusts for subject- and
dataset-specific timing. This decomposition isolates the
subject-/dataset-invariant canonical response from learnable residuals and
avoids the computational infeasibility of full end-to-end stimulus encoding
from raw video at our scale.

We train no per-subject stimulus encoders and no per-subject decoder heads;
all subject-specific behavior enters via the structural FiLM conditioning.

## Training Objectives

**Phase 1 (brain-only):** single- and multi-step forecasting at 1, 5, and 10 TR
horizons, with mean-squared error aggregated over all tokens and all horizons.

**Phase 2 (multimodal):** forecasting loss plus contrastive brain–stimulus
InfoNCE [van den Oord et al. 2018] computed on 30-second non-overlapping
windows. Within-batch hard negatives are drawn from the same stimulus to
prevent the model from solving the task by stimulus identification alone. Loss
weights for the two objectives are fixed equal after a brief sweep; we
explicitly do not learn a balancing scalar, to preserve training reproducibility.

## Cross-Dataset Harmonization

HCP 3T (TR = 0.72 s), HCP 7T (TR = 1.0 s), and CNeuroMod Friends/Movie10
(TR = 1.49 s) are harmonized at the stimulus side rather than the model side:
stimulus features are interpolated to each dataset's native TR grid, and the
model operates uniformly on TR sequences regardless of source. Run-level
balanced sampling weights CNeuroMod up to compensate for HCP's volume
dominance. This avoids introducing per-dataset model components or
learned-temporal-resolution mechanisms whose behavior would be difficult to
audit.

## Evaluation

Generalization is assessed along two axes: (i) **held-out-subject**
generalization on a 20% subject split of HCP 3T and 7T, and on
leave-one-subject-out across CNeuroMod's deeply-sampled subjects;
(ii) **held-out-stimulus** generalization via leave-one-clip-out on HCP 7T
movies and leave-one-season-out on CNeuroMod Friends.

Three downstream transfer tasks evaluate the learned representations using
**frozen-backbone linear probing**:

1. **Subject fingerprinting** — top-`k` subject identification from frozen
   embeddings of held-out runs, with a Schaefer-400 ROI matched-architecture
   baseline.

2. **Phenotype prediction** — Pearson `r` for fluid intelligence, working
   memory (list sorting), and selected NEO scales via subject-CV linear
   regression on frozen embeddings, compared to a behavior-from-FC baseline
   [Finn et al. 2015].

3. **Cognitive task decoding** — 7-class HCP task accuracy via per-block-mean
   embeddings and held-out-subject CV, compared to a volumetric MVPA baseline.

We frame task decoding as forward inference [Poldrack 2006]: we test whether
learned representations preserve task-relevant structure given known task
labels at training, not whether brain states "represent" cognitive processes.

## Implementation and Compute

PyTorch with `mamba-ssm` for the SSM backbone, `nibabel` and `nilearn` for
CIFTI I/O, and DistributedDataParallel for multi-GPU training. Memory
profiling on H200 and B200 nodes confirms the default 5.2M-parameter config
fits 8–18 sequences per H200 GPU; the scaled 26.5M-parameter config fits 3–6
per H200 and benefits substantially from B200 memory. All tokenizers, data
loaders, and CIFTI I/O components will be contributed upstream as PRs to the
nobrainer framework [Ghosh lab, MIT]. The model itself will be released under
Apache 2.0 with reproducible training and evaluation scripts; JOSS submission
will accompany the software release. The research paper will target NeurIPS or
ICLR 2027.

## Scope and Honest Limitations

We deliberately do not pursue several directions occasionally proposed in this
space. We do not perform variational inference or explicit uncertainty
modeling; deterministic forecasting is sufficient for the proposed evaluations,
and post-hoc temperature-calibrated ensembling can provide uncertainty
estimates if needed. We do not train end-to-end on raw video; cached
frozen-encoder features are a well-validated and computationally tractable
alternative. We do not claim "zero-shot" cross-subject generalization;
held-out-subject performance is reported and interpreted as such. We do not
interpret latent states as "representing" specific cognitive processes;
downstream task decoding tests for representational structure consistent with
task labels, which is a forward inference [Poldrack 2006].