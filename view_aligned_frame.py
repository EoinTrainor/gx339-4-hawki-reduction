"""
view_aligned_frame.py
---------------------
Quick visual inspection of a fully processed (aligned) science frame.
Shows the full 2048×2048 image and a zoomed view centred on GX 339-4.

Run in VS Code terminal — displays interactively, nothing saved to disk.

SETTINGS
--------
  OB     : observing block name  (or "random" to pick at random from OBs 1–10)
  INDEX  : frame index within OB  (ignored when OB = "random")
  ZOOM   : half-width of zoom box in pixels  (150 px ≈ 16 arcsec each side)
"""

import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from matplotlib.patches import Rectangle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Settings ──────────────────────────────────────────────────────────────────
OB    = "random"   # e.g. "GX339_Ks_Imaging_7"  or  "random"
INDEX = 0          # frame index within OB (0 = first); ignored if OB = "random"
ZOOM  = 37         # zoom half-width in pixels (75 px box each side)
# ──────────────────────────────────────────────────────────────────────────────

# ── Select frame ──────────────────────────────────────────────────────────────
if OB == "random":
    all_frames = []
    for ob_num in range(1, 11):   # quiescent OBs only
        ob_dir = config.ALIGNED_DIR / f"GX339_Ks_Imaging_{ob_num}"
        all_frames += list(ob_dir.glob("HAWKI.*_cal_aligned.fits"))
    if not all_frames:
        print(f"No aligned frames found under {config.ALIGNED_DIR}")
        sys.exit(1)
    fpath = random.choice(all_frames)
else:
    frames = sorted((config.ALIGNED_DIR / OB).glob("HAWKI.*_cal_aligned.fits"))
    if not frames:
        print(f"No aligned frames found in {config.ALIGNED_DIR / OB}")
        sys.exit(1)
    fpath = frames[INDEX]

print(f"Frame : {fpath.parent.name} / {fpath.name}")

# ── Load frame ────────────────────────────────────────────────────────────────
with fits.open(fpath) as h:
    data = h[0].data.astype(np.float32)

# ── GX 339-4 pixel position ───────────────────────────────────────────────────
# All frames share the same reference pixel grid after alignment.
# Position confirmed by visual inspection — WCS is ~7 arcsec off in Y.
px, py = 750, 1012
print(f"GX 339-4 pixel : ({px}, {py})")

# ── Stretch ───────────────────────────────────────────────────────────────────
valid = data[np.isfinite(data)]
vmin  = np.percentile(valid, 1)
vmax  = np.percentile(valid, 99.5)

# ── Zoom cutout ───────────────────────────────────────────────────────────────
x0 = max(0, px - ZOOM);  x1 = min(data.shape[1], px + ZOOM)
y0 = max(0, py - ZOOM);  y1 = min(data.shape[0], py + ZOOM)
cutout = data[y0:y1, x0:x1]
cv     = cutout[np.isfinite(cutout)]
vmin_z = np.percentile(cv, 2)  if cv.size > 0 else vmin
vmax_z = np.percentile(cv, 99) if cv.size > 0 else vmax

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.patch.set_facecolor("black")

# Full frame
ax = axes[0]
ax.imshow(data, origin="lower", cmap="inferno",
          vmin=vmin, vmax=vmax, interpolation="nearest")
ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                        lw=1.2, edgecolor="cyan", facecolor="none"))
ax.set_title(f"{fpath.parent.name}\n{fpath.name}", color="white", fontsize=7)
ax.set_xlabel("X (px)", color="white")
ax.set_ylabel("Y (px)", color="white")
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor("white")

# Zoom on GX 339-4
ax2 = axes[1]
ax2.imshow(cutout, origin="lower", cmap="inferno",
           vmin=vmin_z, vmax=vmax_z, interpolation="nearest",
           extent=[x0, x1, y0, y1])
ax2.set_title(
    f"GX 339-4  —  {2*ZOOM*0.106:.1f}\" × {2*ZOOM*0.106:.1f}\" region\n"
    f"RA {config.TARGET_RA:.4f}   Dec {config.TARGET_DEC:.4f}",
    color="white", fontsize=9)
ax2.set_xlabel("X (px)", color="white")
ax2.set_ylabel("Y (px)", color="white")
ax2.tick_params(colors="white")
for sp in ax2.spines.values():
    sp.set_edgecolor("white")

fig.suptitle(
    "GX 339-4  ·  Fully Processed Aligned Frame"
    "  (dark sub · flat field · sky sub · aligned)",
    color="white", fontsize=11, y=1.01)
fig.tight_layout()
plt.show()
