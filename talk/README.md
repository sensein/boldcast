# BOLDcast seed-grant pitch deck

Marp-based Markdown deck for the 20-min seed-grant pitch (30-min slot,
~10 min Q&A). Source of truth for the talk structure is the plan file:
[`/home/yibei/.claude/plans/let-s-prepare-the-presentation-lucky-star.md`](../../.claude/plans/let-s-prepare-the-presentation-lucky-star.md).

## Layout

```
talk/
├── seed_pitch.md        # the deck source (one .md file, 22 slides)
├── seed_pitch.html      # rendered HTML (built artefact, regenerable)
├── seed_pitch.pdf       # rendered PDF (built artefact, regenerable)
├── build_figures.py     # regenerates all generated figures from constants
├── figures/             # generated PNGs referenced by the deck
└── README.md            # this file
```

## Rendering

```bash
# HTML (browser-viewable, animated transitions if supported)
npx -y @marp-team/marp-cli@latest talk/seed_pitch.md \
    -o talk/seed_pitch.html --html --allow-local-files

# PDF (what you actually present from)
npx -y @marp-team/marp-cli@latest talk/seed_pitch.md \
    --pdf --html --allow-local-files -o talk/seed_pitch.pdf

# PPTX (if the panel uses PowerPoint)
npx -y @marp-team/marp-cli@latest talk/seed_pitch.md \
    --pptx --html --allow-local-files -o talk/seed_pitch.pptx
```

`--allow-local-files` is required so Chromium (used by Marp under the
hood) can load `figures/*.png`. Without it the slides render blank
where figures should be.

## Regenerating figures

Figures are deterministic from constants in `build_figures.py`. Rerun
when constants change:

```bash
$BOLDCAST_ENV/bin/python talk/build_figures.py
```

Outputs to `talk/figures/`:

| File | Slide | Source |
|---|---|---|
| `hook_scene.gif` + `hook_dynamics.gif` | 2a / 2b | **Placeholder** copies of the old hook GIFs at new filenames. Real visuals to be produced via Claude Design (see `claude_design_prompts.md` Prompt 3). |
| `tokenization_wall.png` | 6 | Constants from proposal + methods |
| `architecture.png` | 5 | Schematic; to be redesigned via Claude Design (Prompt 1) |
| `hybrid_block.png` | 9 | Schematic; to be redesigned via Claude Design (Prompt 2) |
| `mem_scaling.gif` | 8 (why Mamba) | Animated, anchored on Day-3 6.09 GB. First frame held ≥2 s for Marp PDF render fallback |
| `day3_metrics.png` | 12 | PR #3 measurements (params / forward / memory) |
| `day4_loss_curve.png` | 10 | `results/day4_sanity_lr1e3/loss_log.jsonl` — 3000-step Day-4 overfit run (0.361 → 0.081, ratio 20.1%) |

## Design system

The deck uses a centralized design system located at
`~/.claude/skills/slide-design/` (symlink → `~/schist-vault/shared/skills/slide-design/`).
The system provides:

- **Palette YAMLs** — `palettes/mit.yaml` (current), `palettes/editorial-warm.yaml`, `palettes/cool-minimal.yaml`
- **Build tools** — matplotlib rcParams generator, CSS variables emitter, WCAG contrast auditor
- **BOLDcast integration** — wrapper at `talk/_design/boldcast_palette.py` loads the current palette and re-exports `COLORS` and `apply_rcparams`; generated `talk/_design/palette.css` contains CSS variables used by the deck's `<style>` block

To switch palettes: edit `PALETTE_NAME` in `talk/_design/boldcast_palette.py`, then regenerate CSS:

```bash
~/.claude/skills/slide-design/_build/palette_css.py mit > talk/_design/palette.css
```

Paste the new variable block into `seed_pitch.md`'s `<style>` tag. Instructions for adding a new palette are in `~/.claude/skills/slide-design/palettes/README.md`.

**Animations** — `hook.gif` (slide 2) and `mem_scaling.gif` (slide 8); first frame is the complete state for PDF fallback. **Callout convention** — red gap annotations on slides 3, 4, 14, 16 use `.fade-in-late` class (fade in 2 s after slide load). The `_palette_options.py` exploratory script in `talk/figures/` is planning only and will be removed during cleanup.

## Structure

13 main slides + 1 title + 1 thank-you + 5 backup slides = 22 total.

| # | Title | Time | Role |
|---|---|---|---|
| 1 | BOLDcast (title) | 0.1 min | Title only — do not narrate |
| 2 | Hook: forecast a brain mid-movie | 1.5 min | Earn attention |
| 3 | The gap isn't scale — it's geometry and stimulus | 1 min | Gap part 1 (7 brain foundation models: BrainLM, SwiFT, Brain-JEPA, NeuroSTORM, Omni-fMRI, Brain-DiT, BrainGFM) |
| 4 | Stimulus alignment: three schools, one missing piece | 1 min | Gap part 2 (incl. TRIBE v2 from Meta as the contemporary encoding model) |
| 5 | BOLDcast in one slide | 1 min | Anchor diagram |
| 6 | Tokenization is a compression problem | 1 min | Concrete problem |
| 7 | Geodesic patches + k-means subcortex | 1.5 min | Tokenization detail + fidelity |
| 8 | Why Mamba, not transformer | 2 min | Pre-empt skeptic Q1 |
| 9 | Hybrid architecture + scaling | 1.5 min | Stack diagram + configs |
| 10 | Day 1 — Tokenizer on real HCP | 1.5 min | Readiness anchor 1/3 |
| 11 | Day 2 — Dataset, cached + validated | 1 min | Readiness anchor 2/3 |
| 12 | Day 3 — Model on H200 | 1.5 min | Readiness anchor 3/3 (load-bearing) |
| 13 | Phase 1 — brain-only foundation | 1.5 min | Funded work part 1 |
| 14 | Phase 2 — joint stimulus + brain | 1.5 min | Funded work + honest bracket |
| 15 | Months 4–6 — evaluation, release, manuscript | 1 min | Deliverables |
| 16 | The ask | 1.5 min | Compute envelope + closing line |
| 17 | Thank you (Q&A backdrop) | — | Acknowledgments slide |
| 18 | B-1 · Mamba vs FlashAttention | backup | Triggered Q: "why not FA-2?" |
| 19 | B-2 · Geodesic FPS + Lloyd | backup | Triggered Q: patch-construction stability |
| 20 | B-3 · HRF: canonical + learned residual | backup | Triggered Q: does learned HRF overfit? |
| 21 | B-4 · Cross-dataset harmonization | backup | Triggered Q: HCP+CNeuroMod blending |
| 22 | B-5 · Family-disjoint splits + CIs | backup | Triggered Q: demo n=8 too small? |

## Numbers audit

Run this audit before every rehearsal — see
`superpowers:results-verification` discipline.

| Slide | Number | Canonical source |
|---|---|---|
| §4b · §10 | 6.2 × 10⁻¹¹ float64 round-trip | Day-1 PR #1 validation on subject 115825 (memory: `session_handoff_may11.md`) |
| §5 | 67 M attention ops per sample | Derive: 256² × 1024 ≈ 67 M |
| §5 · §12 | 6.09 GB F+B on H200 | Commit `2564b5e` ("verified at 6.09 GB on H200"); PR #3 |
| §12 | 139 ms forward | Day-3 measurement; PR #3 |
| §12 | 0.733 M params | Day-3 CI assert; ADR `0004` (range [0.5M, 1.5M]) |
| §9 | 5.2 M / 26.5 M (default / scaled) | `configs/arch/default.yaml`, `configs/arch/scaled.yaml`; proposal Table 1 |
| §14 | ≥ 15 % MSE improvement, ≥ 60 % top-5 retrieval | Proposal § 3 success criteria |
| §16 | 6.9 / 18.7 GB ckpt; 8–18 / 3–6 seq/GPU H200 | Proposal § 2 Table 1 |

## Literature snapshot

Slides 3 + 4 are the load-bearing positioning slides. The picture as of
2026-05:

**Brain foundation models on fMRI (slide 3)**

- BrainLM (NeurIPS '24), SwiFT (NeurIPS '23), Brain-JEPA (NeurIPS '24)
- NeuroSTORM (Nature BME '26) — Mamba on 4D voxel volumes
- Omni-fMRI (arXiv '26) — atlas-free dynamic voxel patching
- Brain-DiT (arXiv '26) — diffusion transformer, metadata-conditioned
- BrainGFM (OpenReview '26) — graph contrastive / masked autoencoder

**Stimulus-brain alignment (slide 4)**

- Static-image retrieval: MindEye, MindEye2, BrainCLIP. Thin brain
  head; frozen CLIP on a single frame.
- Stimulus → brain encoding: **TRIBE v2 (Meta FAIR, March 2026)** —
  trained on 451.6 h of naturalistic fMRI (movies, podcasts, silent
  videos). The three "TRIBE" encoders (V-JEPA2, LLaMA 3.2,
  Wav2Vec-BERT) are **stimulus-side and frozen** — TRIBE v2 itself is
  a regression head from stimulus features to voxel activity, not a
  brain foundation model. One-way mapping, no brain backbone.
- BOLDcast: real pretrained brain backbone (surface-tokenized) +
  frozen CLIP on the stimulus side + bidirectional joint latent +
  forecasting head.

**BOLDcast's defensible lane (after the 2026 SOTA update):**

- ✗ Not the first Mamba in fMRI (NeuroSTORM)
- ✗ Not the first atlas-free (Omni-fMRI)
- ✗ Not the first naturalistic foundation model (TRIBE v2)
- ✓ Surface-based geodesic tokenization (every published model is
  volumetric or atlas-graph)
- ✓ Bidirectional brain ↔ stimulus latent (TRIBE v2 is one-way; nobody
  else ingests stimulus)
- ✓ Multi-step brain-state forecasting on top of joint latent

Re-check this section before each rehearsal — the literature is moving
fast and an unaddressed rival on stage is more damaging than a
mediocre claim.

## User-action items before rehearsal

These are things Claude cannot do — they require HCP data access, the
user's identity (Restricted DUA, proposal authorship), or external
scheduling.

1. ~~**Slide 10 cortex render.**~~
   Superseded: slide 10 now shows `day4_loss_curve.png` (Day-4 overfit
   training run) instead of the cortex visualization. The old
   `cortex_patches.png` was removed during the post-talk iteration.

2. **Slide 16 GPU-hour numbers.**
   The current Slide 16 "ask" frames the compute envelope qualitatively
   (`8 × H200 DDP` per phase). The proposal text doesn't quantify a
   GPU-hour total. If the seed-fund process expects a number, fill it
   in before rehearsal. Otherwise leave as scoped phases.

3. **Talk date — rehearsal scheduling.**
   The plan recommends a timed full run 48 h before the actual talk
   plus an adversarial dry-run with the top-three skeptic Q&A. The
   plan file has the rehearsal verification gate.

## Rehearsal gate

Per the plan, the rehearsal pass succeeds when:

- (a) Timed run hits 19 min ± 30 s. If over, cut a slide; do not speed up.
  Likely first cut: collapse §6 into §5.
- (b) Every number checks against the audit table above.
- (c) The three pre-empted questions ("Why not FlashAttention?", "Why
  not just an atlas?", "What about Phase 2?") each get a confident
  < 60 s answer using main + backup slides.
- (d) Backup-slide trigger pointers on main slides (e.g., "→ backup B-3")
  correspond to actually-present backup slides.

## What is *not* in this deck (intentional)

- No biological introduction to fMRI / BOLD signal. Audience is
  ML-fluent, neuro-light — they don't need it and it costs a slide.
- No "literature review" beyond the three foundation-model rivals on
  slide 3 and the two stimulus-alignment rivals on slide 4. The talk
  is a pitch, not a review.
- No detailed math for the Mamba selective scan. Backup B-1 has the
  comparison table; the speaker should derive cost intuitively.
- No real loss curves yet — Days 4–7 of the demo plan are still in
  progress. Slide 12 stops at "training loop lands this week" and does
  not pretend results that don't exist.
- No fingerprinting demo result. The deck deliberately frames the
  fingerprinting eval as a *Phase 1* deliverable, not a *demo*
  deliverable (consistent with the 10-day plan's Day 7 caveats).
