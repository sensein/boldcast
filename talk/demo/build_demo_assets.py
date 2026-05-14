"""Build cortex assets for the interactive seed-grant demo page.

Outputs to `talk/demo/`:

  cortex_lh_display.png   — pretty 1024-color view of LH patches (lateral view)
  cortex_lh_picking.png   — same projection, RGB-encoded patch_id (canvas pixel lookup)
  cortex_rh_display.png   — same for RH
  cortex_rh_picking.png   — same for RH
  patch_meta.json         — per-patch metadata for the sidebar (hemisphere, vertex count)

Inputs (no HCP data):
  - Conte69 fs_LR 32k template via brainspace.datasets.load_conte69 (PUBLIC)
  - {L,R}.atlasroi.32k_fs_LR.shape.gii from HCPpipelines GitHub (PUBLIC template,
    cached in talk/demo/templates/) — the medial-wall mask that maps
    cortex grayordinate index → mesh vertex index.
  - cache/patches_fsLR_32k_n1024_seed0_geo.npz (computed by Day-1 tokenizer)

Picking-PNG encoding:
  Each cortex pixel encodes its patch_id in RGB:
      R = (patch_id // 256) & 0xFF
      G = patch_id & 0xFF
      B = 255                    (sentinel: "this pixel is brain")
  Non-cortex pixels have B = 0 so the JS pick handler can filter them.

Run from repo root:
    $BOLDCAST_ENV/bin/python talk/demo/build_demo_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from brainspace.datasets import load_conte69
from matplotlib.colors import LightSource
from matplotlib.tri import Triangulation

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "cache" / "patches_fsLR_32k_n1024_seed0_geo.npz"
OUT = Path(__file__).resolve().parent
TEMPLATES = OUT / "templates"

OUT.mkdir(parents=True, exist_ok=True)


def _load_atlas_roi() -> tuple[np.ndarray, np.ndarray]:
    """Load HCP-standard cortex ROI masks for fs_LR 32k. Returns boolean
    arrays of shape (32492,) where True = cortex grayordinate."""
    import nibabel as nib

    lh = nib.load(TEMPLATES / "L.atlasroi.32k_fs_LR.shape.gii").darrays[0].data
    rh = nib.load(TEMPLATES / "R.atlasroi.32k_fs_LR.shape.gii").darrays[0].data
    return lh.astype(bool), rh.astype(bool)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _vtk_to_points_faces(surf):
    """Return (verts, faces) numpy arrays from a brainspace BSPolyData surface."""
    verts = np.asarray(surf.Points)
    n_polys = surf.GetNumberOfPolys()
    # brainspace polys are triangles; pull as (n, 3)
    polys = surf.GetPolys2D()
    faces = np.asarray(polys).reshape(-1, 3) if polys is not None else None
    if faces is None or faces.size == 0:
        # Fallback: walk cells
        from vtkmodules.util.numpy_support import vtk_to_numpy

        raw = vtk_to_numpy(surf.GetPolys().GetData()).reshape(-1, 4)
        faces = raw[:, 1:]
    return verts, faces


def _lateral_projection(verts: np.ndarray, hemisphere: str) -> np.ndarray:
    """Project 3D vertices to 2D for a lateral view of one hemisphere.

    Conte69 vertices are in MNI-ish space. For LH the lateral view looks
    down +X axis (anatomical left = patient left); for RH it looks down -X.
    Returns (n, 2) array of (y, z) coordinates appropriate for the view.
    """
    if hemisphere == "lh":
        # Lateral LH: viewer looks from -X toward +X, sees Y (post→ant) and Z (inf→sup).
        # Mirror y so anterior appears to the right (conventional radiological).
        xy = np.column_stack([-verts[:, 1], verts[:, 2]])
    elif hemisphere == "rh":
        # Lateral RH: viewer looks from +X toward -X.
        xy = np.column_stack([verts[:, 1], verts[:, 2]])
    else:
        raise ValueError(hemisphere)
    return xy


def _patch_palette(n_patches: int = 1024, seed: int = 0) -> np.ndarray:
    """Generate `n_patches` RGB colors that look perceptually distinct on a brain.

    Strategy: HSV with shuffled hue order so adjacent patch IDs (which are
    spatial neighbors after FPS) get different hues. Saturation and value
    are jittered slightly so same-hue patches still look different.
    """
    rng = np.random.default_rng(seed)
    hues = (np.arange(n_patches) / n_patches) ** 0.85  # slight non-linearity
    rng.shuffle(hues)
    sat = rng.uniform(0.55, 0.95, size=n_patches)
    val = rng.uniform(0.65, 0.95, size=n_patches)
    hsv = np.stack([hues, sat, val], axis=1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb


def _render_hemi(
    verts: np.ndarray,
    faces: np.ndarray,
    patches: np.ndarray,
    hemisphere: str,
    palette: np.ndarray,
    out_display: Path,
    out_picking: Path,
    img_size_px: int = 1100,
    margin: float = 6.0,
) -> dict:
    """Render one hemisphere as two PNGs (display + picking).

    Both PNGs are pure tripcolor renders with `shading='flat'` so every
    triangle is exactly one color — picking-PNG pixel reads are then
    unambiguous (no antialiasing-induced gradient between patch colors).

    Returns a dict with the 2D bounding box used for the render, which the
    HTML frontend uses to map clicks back to patches if needed.
    """
    proj = _lateral_projection(verts, hemisphere)

    # Per-vertex patch ID; -1 means medial wall.  Majority-vote per triangle.
    face_patches = np.zeros(faces.shape[0], dtype=np.int32)
    for i, (a, b, c) in enumerate(faces):
        ids = (int(patches[a]), int(patches[b]), int(patches[c]))
        if ids[0] == ids[1] or ids[0] == ids[2]:
            face_patches[i] = ids[0]
        elif ids[1] == ids[2]:
            face_patches[i] = ids[1]
        else:
            face_patches[i] = ids[0]

    is_medial = face_patches < 0  # triangles inside the medial wall

    # Face colors for display PNG: palette for cortex, neutral grey for medial wall.
    face_colors = np.empty((face_patches.size, 3), dtype=np.float64)
    cortex_mask = ~is_medial
    face_colors[cortex_mask] = palette[face_patches[cortex_mask]]
    face_colors[is_medial] = (0.78, 0.78, 0.80)  # cool-grey medial wall

    # Cheap shading: per-face normal z component → brightness scale
    v_a = verts[faces[:, 0]]
    v_b = verts[faces[:, 1]]
    v_c = verts[faces[:, 2]]
    normals = np.cross(v_b - v_a, v_c - v_a)
    n_len = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals_n = normals / n_len
    # For LH lateral view, light coming from -X; for RH, from +X.
    light_dir = np.array([-1.0, 0.0, 0.5]) if hemisphere == "lh" else np.array([1.0, 0.0, 0.5])
    light_dir = light_dir / np.linalg.norm(light_dir)
    shading = (normals_n @ light_dir).clip(-1, 1)
    brightness = 0.55 + 0.45 * (shading * 0.5 + 0.5)  # in [0.55, 1.0]
    shaded = face_colors * brightness[:, None]
    shaded = shaded.clip(0, 1)

    # Picking colors: RGB-encoded patch_id (cortex) or B=0 sentinel (medial wall)
    safe_patches = np.where(is_medial, 0, face_patches)
    pick_colors = np.zeros((face_patches.size, 3), dtype=np.float64)
    pick_colors[:, 0] = ((safe_patches // 256) & 0xFF) / 255.0
    pick_colors[:, 1] = (safe_patches & 0xFF) / 255.0
    pick_colors[:, 2] = np.where(is_medial, 0.0, 1.0)  # B=255 = brain; B=0 = not brain

    # Build (n_tri, 3, 2) polygon vertex array for PolyCollection — direct
    # control over per-face RGB without matplotlib's colormap pipeline.
    tri_xy = proj[faces]  # (n_tri, 3, 2)

    bbox = dict(
        xmin=float(proj[:, 0].min()),
        xmax=float(proj[:, 0].max()),
        ymin=float(proj[:, 1].min()),
        ymax=float(proj[:, 1].max()),
    )
    width = bbox["xmax"] - bbox["xmin"]
    height = bbox["ymax"] - bbox["ymin"]
    aspect = width / height
    img_w = img_size_px
    img_h = int(img_size_px / aspect)
    dpi = 100.0
    fig_w = img_w / dpi
    fig_h = img_h / dpi

    from matplotlib.collections import PolyCollection

    for kind, colors, out_path in [
        ("display", shaded, out_display),
        ("picking", pick_colors, out_picking),
    ]:
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
        fig.patch.set_facecolor("none")
        ax.set_facecolor("none")
        ax.set_xlim(bbox["xmin"] - margin, bbox["xmax"] + margin)
        ax.set_ylim(bbox["ymin"] - margin, bbox["ymax"] + margin)
        ax.set_aspect("equal")
        ax.axis("off")
        # PolyCollection: one polygon per triangle, explicit per-face RGB.
        # Picking PNG: antialiasing OFF so per-pixel RGB == intended patch_id.
        antialiased = (kind == "display")
        coll = PolyCollection(
            tri_xy,
            facecolors=colors,
            edgecolors="none",
            linewidths=0,
            antialiased=antialiased,
        )
        ax.add_collection(coll)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        fig.savefig(out_path, dpi=dpi, transparent=True, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        print(f"  wrote {out_path.name} ({img_w}×{img_h})")

    return dict(bbox=bbox, img_w=img_w, img_h=img_h)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"loading Conte69 fs_LR 32k template via brainspace …")
    surf_lh, surf_rh = load_conte69()
    verts_lh, faces_lh = _vtk_to_points_faces(surf_lh)
    verts_rh, faces_rh = _vtk_to_points_faces(surf_rh)
    print(f"  LH: {verts_lh.shape[0]} verts, {faces_lh.shape[0]} tris")
    print(f"  RH: {verts_rh.shape[0]} verts, {faces_rh.shape[0]} tris")

    print(f"loading HCP atlas-ROI masks from {TEMPLATES.relative_to(REPO)} …")
    mask_lh, mask_rh = _load_atlas_roi()
    print(f"  LH cortex verts: {mask_lh.sum()}, RH cortex verts: {mask_rh.sum()}")

    print(f"loading patch assignment from {CACHE.relative_to(REPO)} …")
    cache = np.load(CACHE)
    print(f"  cache keys: {list(cache.files)}")
    patches_grayord = cache["assignment"].astype(np.int32)
    n_lh_cortex = int(cache["n_lh_cortex"])
    n_rh_cortex = int(cache["n_rh_cortex"])
    assert mask_lh.sum() == n_lh_cortex, (
        f"LH mask sum {mask_lh.sum()} != n_lh_cortex {n_lh_cortex} from cache"
    )
    assert mask_rh.sum() == n_rh_cortex, (
        f"RH mask sum {mask_rh.sum()} != n_rh_cortex {n_rh_cortex} from cache"
    )

    # Expand cortex-only patch IDs to per-vertex arrays of length 32492 each.
    # Medial-wall vertices get sentinel -1 (no patch).
    n_lh, n_rh = verts_lh.shape[0], verts_rh.shape[0]
    patches_lh = -np.ones(n_lh, dtype=np.int32)
    patches_rh = -np.ones(n_rh, dtype=np.int32)
    patches_lh[mask_lh] = patches_grayord[:n_lh_cortex]
    patches_rh[mask_rh] = patches_grayord[n_lh_cortex:]
    n_patches = int(patches_grayord.max()) + 1
    print(f"  {n_patches} patches; LH range {patches_lh[mask_lh].min()}..{patches_lh[mask_lh].max()}, "
          f"RH range {patches_rh[mask_rh].min()}..{patches_rh[mask_rh].max()}")

    palette = _patch_palette(n_patches=n_patches, seed=0)
    np.save(OUT / "_palette.npy", palette)

    print("rendering LH …")
    info_lh = _render_hemi(
        verts_lh, faces_lh, patches_lh, "lh", palette,
        OUT / "cortex_lh_display.png",
        OUT / "cortex_lh_picking.png",
    )
    print("rendering RH …")
    info_rh = _render_hemi(
        verts_rh, faces_rh, patches_rh, "rh", palette,
        OUT / "cortex_rh_display.png",
        OUT / "cortex_rh_picking.png",
    )

    # Per-patch metadata for the sidebar
    meta = []
    for pid in range(n_patches):
        if pid < (n_patches // 2):
            n_verts = int((patches_lh == pid).sum())
            hemi = "LH"
        else:
            n_verts = int((patches_rh == pid).sum())
            hemi = "RH"
        meta.append({"id": pid, "hemisphere": hemi, "n_vertices": n_verts})
    meta_path = OUT / "patch_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "n_patches": n_patches,
                "lh_render": info_lh,
                "rh_render": info_rh,
                "patches": meta,
            },
            indent=2,
        )
    )
    print(f"  wrote {meta_path.name} ({n_patches} patches)")
    print("done.")


if __name__ == "__main__":
    main()
