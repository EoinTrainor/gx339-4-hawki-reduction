"""
view_aligned_frame.py
---------------------
Quick visual check of a fully processed (aligned) FITS frame.
Shows the full 2048x2048 image and a zoomed view centred on GX 339-4.
Run in VS Code — displays interactively, nothing saved to disk.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# ── Pick a frame ──────────────────────────────────────────────────────────────
# Change OB or index here to view a different frame
OB      = "GX339_Ks_Imaging_7"   # OB7 — best seeing night (0.90")
INDEX   = 0                        # 0 = first frame of that OB

# Zoom box half-width around GX 339-4 (pixels)
ZOOM_PX = 80   # 80 px ≈ 8.5 arcsec each side

# ── Load frame ────────────────────────────────────────────────────────────────
frames = sorted((config.ALIGNED_DIR / OB).glob("HAWKI.*_cal_aligned.fits"))
if not frames:
    print(f"No aligned frames found in {config.ALIGNED_DIR / OB}")
    sys.exit(1)

fpath = frames[INDEX]
print(f"Loading: {fpath.name}")

with fits.open(fpath) as hdul:
    data   = hdul[0].data.astype(np.float32)
    header = hdul[0].header

# ── Find GX 339-4 pixel position using the REFERENCE frame WCS ───────────────
# All aligned frames are mapped onto the reference grid, so the reference
# frame's WCS is the correct one for locating objects in pixel space.
ref_ob    = "GX339_Ks_Imaging_1"
ref_frame = sorted((config.ALIGNED_DIR / ref_ob).glob("HAWKI.*_cal_aligned.fits"))[0]
with fits.open(ref_frame) as hdul_ref:
    ref_wcs = WCS(hdul_ref[0].header)

px, py = ref_wcs.all_world2pix([[config.TARGET_RA, config.TARGET_DEC]], 0)[0]
px, py = int(np.round(px)), int(np.round(py))
print(f"GX 339-4 pixel position: ({px}, {py})")

# ── Stretch limits (percentile, ignoring NaNs) ────────────────────────────────
valid  = data[np.isfinite(data)]
vmin   = np.percentile(valid, 1)
vmax   = np.percentile(valid, 99.5)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.patch.set_facecolor("black")

# --- Full frame ---
ax = axes[0]
im = ax.imshow(data, origin="lower", cmap="inferno",
               vmin=vmin, vmax=vmax, interpolation="nearest")
ax.plot(px, py, "+", color="cyan", ms=14, mew=1.5, label="GX 339-4")
ax.set_title(f"Full frame — {OB}\n{fpath.name[:40]}",
             color="white", fontsize=9)
ax.set_xlabel("X (px)", color="white")
ax.set_ylabel("Y (px)", color="white")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("white")
ax.legend(fontsize=8, loc="upper right",
          labelcolor="cyan", facecolor="black", edgecolor="white")

# Draw zoom box on full frame
from matplotlib.patches import Rectangle
rect = Rectangle((px - ZOOM_PX, py - ZOOM_PX),
                 2 * ZOOM_PX, 2 * ZOOM_PX,
                 linewidth=1.2, edgecolor="cyan", facecolor="none")
ax.add_patch(rect)

# --- Zoomed view ---
x0 = max(0, px - ZOOM_PX)
x1 = min(data.shape[1], px + ZOOM_PX)
y0 = max(0, py - ZOOM_PX)
y1 = min(data.shape[0], py + ZOOM_PX)
cutout = data[y0:y1, x0:x1]

vmin_z = np.nanpercentile(cutout, 2)
vmax_z = np.nanpercentile(cutout, 99)

ax2 = axes[1]
ax2.imshow(cutout, origin="lower", cmap="inferno",
           vmin=vmin_z, vmax=vmax_z, interpolation="nearest",
           extent=[x0, x1, y0, y1])
ax2.plot(px, py, "+", color="cyan", ms=18, mew=2.0)
ax2.set_title(f"GX 339-4 zoom  ({2*ZOOM_PX*0.106:.1f}\" × {2*ZOOM_PX*0.106:.1f}\")\n"
              f"centre: RA={config.TARGET_RA:.5f}  Dec={config.TARGET_DEC:.5f}",
              color="white", fontsize=9)
ax2.set_xlabel("X (px)", color="white")
ax2.set_ylabel("Y (px)", color="white")
ax2.tick_params(colors="white")
for spine in ax2.spines.values():
    spine.set_edgecolor("white")

scale_px = 50
scale_as = scale_px * 0.106
ax2.plot([x0 + 8, x0 + 8 + scale_px], [y0 + 8, y0 + 8],
         color="white", lw=2)
ax2.text(x0 + 8 + scale_px / 2, y0 + 14,
         f'{scale_as:.1f}"', color="white", fontsize=8, ha="center")

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.suptitle("GX 339-4 — Fully Processed Frame  (dark sub · flat field · sky sub · aligned)",
             color="white", fontsize=11, y=0.99)

plt.show()
