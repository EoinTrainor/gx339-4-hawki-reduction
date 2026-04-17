# title: GX 339-4 WCS Trust Check Viewer (cutout + aperture/annulus overlay for every FITS)

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus


# ---------------- USER INPUTS ----------------
FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"

# GX 339-4 coordinates (decimal degrees)
GX339_RA_DEG  = 255.7057818297
GX339_DEC_DEG = -48.7897466540

# Photometry geometry (pixels)
AP_RADIUS_PX = 6.0
ANN_IN_R_PX  = 10.0
ANN_OUT_R_PX = 16.0

# Viewer controls
CUTOUT_HALF_SIZE_PX = 80   # 80 => 161x161 cutout (good for checking placement)
PERCENTILE_STRETCH = (5, 99.7)


# ---------------- FIND FITS FILES ----------------
patterns = [
    os.path.join(FITS_DIR, "*.fits"),
    os.path.join(FITS_DIR, "*.fit"),
    os.path.join(FITS_DIR, "*.fits.fz"),
    os.path.join(FITS_DIR, "*.fz"),
]
fits_files = []
for p in patterns:
    fits_files.extend(glob.glob(p))
fits_files = sorted(set(fits_files))

if len(fits_files) == 0:
    raise FileNotFoundError(f"No FITS files found in: {FITS_DIR}")

print(f"Found {len(fits_files)} FITS files.")


# ---------------- HELPERS ----------------
def load_first_2d_hdu(fits_path: str):
    with fits.open(fits_path) as hdul:
        hdu_idx = None
        for i, h in enumerate(hdul):
            if getattr(h, "data", None) is None:
                continue
            if h.data is None:
                continue
            if np.ndim(h.data) == 2:
                hdu_idx = i
                break
        if hdu_idx is None:
            raise ValueError("No 2D image HDU found.")
        data = hdul[hdu_idx].data.astype(float)
        header = hdul[hdu_idx].header
        wcs = WCS(header)
    data[~np.isfinite(data)] = np.nan
    return data, wcs, header


def world_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(c)
    return float(x), float(y)


def cutout(data: np.ndarray, x: float, y: float, half: int):
    ny, nx = data.shape
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = max(0, x0 - half)
    x2 = min(nx, x0 + half + 1)
    y1 = max(0, y0 - half)
    y2 = min(ny, y0 + half + 1)
    sub = data[y1:y2, x1:x2]
    return sub, x1, y1  # plus offsets


# ---------------- INTERACTIVE VIEWER ----------------
idx = 0
fig, ax = plt.subplots(figsize=(7, 7))

def draw(i):
    ax.clear()
    fp = fits_files[i]
    data, wcs, header = load_first_2d_hdu(fp)

    x, y = world_to_pixel(wcs, GX339_RA_DEG, GX339_DEC_DEG)

    sub, xoff, yoff = cutout(data, x, y, CUTOUT_HALF_SIZE_PX)

    # Robust stretch
    finite = sub[np.isfinite(sub)]
    if finite.size > 0:
        vmin, vmax = np.nanpercentile(finite, PERCENTILE_STRETCH)
    else:
        vmin, vmax = 0, 1

    ax.imshow(sub, origin="lower", vmin=vmin, vmax=vmax, cmap = "gray")

    # Convert global coords into cutout coords
    xc = x - xoff
    yc = y - yoff

    ap = CircularAperture([(xc, yc)], r=AP_RADIUS_PX)
    ann = CircularAnnulus([(xc, yc)], r_in=ANN_IN_R_PX, r_out=ANN_OUT_R_PX)
    ap.plot(ax=ax, lw=2.5, color="yellow")
    ann.plot(ax=ax, lw=1.5, color="yellow", alpha=0.8)


    ax.scatter([xc], [yc], s=40)

    date_obs = header.get("DATE-OBS", "DATE-OBS not found")
    ax.set_title(f"[{i+1}/{len(fits_files)}] {os.path.basename(fp)}\n{date_obs}")
    ax.set_xlabel("x (cutout pixels)")
    ax.set_ylabel("y (cutout pixels)")
    fig.canvas.draw_idle()

draw(idx)

def on_key(event):
    global idx

    if event.key in ("right",):      # Next image
        idx = min(idx + 1, len(fits_files) - 1)
        draw(idx)

    elif event.key in ("left",):     # Previous image
        idx = max(idx - 1, 0)
        draw(idx)

    elif event.key in ("enter", "return"):   # Quit viewer
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)
plt.show()

print("Controls: n=next, b=back, q=quit")
