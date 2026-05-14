# BOLDcast interactive demo (Tier B)

Single-page interactive companion to the seed-grant pitch deck.
Two interactive moments:

1. **Tokenization** — click any patch on either hemisphere, sidebar
   shows patch ID + hemisphere + vertex count. The selected patch
   highlights in cadmium-yellow on the cortex.
2. **Memory scaling** — drag the `T` slider to see Mamba (linear)
   vs Transformer attention (quadratic) memory grow; the cursor
   line tracks on the log-log plot, GB readouts update live.

No live model inference, no HCP data load, no DUA concerns.

## Launch (≤ 1 minute)

```bash
cd talk/demo
$BOLDCAST_ENV/bin/python -m http.server 8765
```

Then open `http://localhost:8765/` in a browser. For the talk, do this
**before** the session starts so the page is loaded — switching mid-talk
is then alt-tab + ready.

Important: **`file://` won't work.** Chromium blocks `getImageData` on
local-disk PNGs for CORS reasons. Must serve via http.server (or any
static server).

## Demo integration into the talk

Suggested moments to alt-tab:

- After **slide 6** (tokenization compression wall): switch to the
  demo, click 2–3 patches in different cortical regions, narrate
  *"these 1,024 patches are exactly what the model sees per TR — and
  every one round-trips to native CIFTI within 6 × 10⁻¹¹"*. 30–60 s.
- After **slide 8** (Why Mamba): switch to the demo's scaling chart,
  drag `T` from 256 to 1024, show the transformer line shoot through
  the H200 ceiling while Mamba stays linear. 20–30 s.

Total demo time budget: ~90 s across two moments. Leaves the 19-min
talk budget intact.

## Files

| File | Purpose |
|---|---|
| `index.html` | Single-page app — vanilla JS, Plotly via CDN |
| `cortex_{lh,rh}_display.png` | Pretty 1,024-color view |
| `cortex_{lh,rh}_picking.png` | Patch-ID-encoded RGB (canvas pixel pick) |
| `patch_meta.json` | Per-patch hemisphere + vertex count for sidebar |
| `_palette.npy` | Cached HSV palette (debug; not used at runtime) |
| `templates/{L,R}.atlasroi.32k_fs_LR.shape.gii` | HCP-public cortex masks (medial-wall removal) |
| `build_demo_assets.py` | Regenerates PNGs + JSON from cache + templates |

## Regenerating assets

Only needed if the patch assignment changes (different seed, different
n_patches, etc.):

```bash
$BOLDCAST_ENV/bin/python talk/demo/build_demo_assets.py
```

Deterministic; same patch cache + seed 0 palette → identical PNGs.

## Known limitations

- **Lateral view only.** Medial wall isn't visible — patches on the
  inner cortical surface aren't reachable by click. The patcher does
  cover them; this is a visualization gap, not a tokenization gap.
- **No subcortex / cerebellum view.** The "+ 768" subcortex tokens are
  shown as a stat but not visualized. (Adding a glass-brain volumetric
  view would be Tier C territory.)
- **Display PNGs ~1 MB each.** Total page weight ~2.5 MB. Fine for
  in-room demo over local server; not optimised for a public deploy.
- **No model inference.** Forecasting / stimulus alignment are talked
  about on the deck but the page only visualizes the tokenizer +
  scaling math.

## Style provenance

Palette: MIT brand (`--primary #002896`, `--gap #750014`,
`--success #004d1a`, full ramp from
`~/.claude/skills/slide-design/palettes/mit.yaml`). CSS variables match
the deck's `seed_pitch.md` exactly so the demo looks like a continuation
of the same slide.
