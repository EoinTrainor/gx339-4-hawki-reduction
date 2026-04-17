# title: Stage 2 – WCS propagation and local centroid refinement (diagnostic only)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# ---------------- USER INPUT ----------------
FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"

# Two FITS files to compare (reference + another)
import os

FITS_FILES = [
    r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
]

# add a second file automatically, but with full path
all_fits = sorted([
    os.path.join(FITS_DIR, f)
    for f in os.listdir(FITS_DIR)
    if f.endswith(".fits")
])

if len(all_fits) < 2:
    raise RuntimeError("Need at least two FITS files for Stage 2.")

FITS_FILES.append(all_fits[1])


# Object sky coordinates (from Stage 1)
OBJECTS = {
    "GX 339-4": (255.7057818297, -48.7897466540),
    "Star 1":   (255.707140, -48.788802),  # <- fill from Stage 1 output
    "Star 2":   (255.711256, -48.791347),
    "Star 3":   (255.711298, -48.788757),
}

# Size of centroiding window (pixels)
CENTROID_HALF_SIZE = 10

# ---------------- HELPERS ----------------
def load_image(fp):
    with fits.open(fp) as hdul:
        for h in hdul:
            if h.data is not None and h.data.ndim == 2:
                data = h.data.astype(float)
                header = h.header
                break
    data[~np.isfinite(data)] = np.nan
    return data, WCS(header)

def centroid_local(data, x0, y0, half):
    x0i, y0i = int(round(float(x0))), int(round(float(y0)))
    y1, y2 = y0i - half, y0i + half + 1
    x1, x2 = x0i - half, x0i + half + 1

    cut = data[y1:y2, x1:x2]
    if cut.size == 0 or not np.isfinite(cut).any():
        return np.nan, np.nan

    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = np.isfinite(cut)
    flux = cut[mask]

    xc = np.sum(xx[mask] * flux) / np.sum(flux)
    yc = np.sum(yy[mask] * flux) / np.sum(flux)
    return xc, yc

# ---------------- MAIN ----------------
for fp in FITS_FILES:
    print("\n==============================")
    print("Processing:", fp)
    print("==============================")

    data, wcs = load_image(fp)

    plt.figure(figsize=(6, 6))
    plt.imshow(data, origin="lower", cmap="gray",
               vmin=np.nanpercentile(data, 5),
               vmax=np.nanpercentile(data, 99.7))

    for name, (ra, dec) in OBJECTS.items():
        if ra is None:
            continue  # fill these after pasting Stage 1 values

        sky = SkyCoord(ra * u.deg, dec * u.deg)
        x_wcs, y_wcs = wcs.world_to_pixel(sky)
        x_wcs = float(np.squeeze(x_wcs))
        y_wcs = float(np.squeeze(y_wcs))

        x_cent, y_cent = centroid_local(
            data, x_wcs, y_wcs, CENTROID_HALF_SIZE
        )

        dx = x_cent - x_wcs
        dy = y_cent - y_wcs

        print(f"{name}:")
        print(f"  WCS guess     x={x_wcs:.2f}, y={y_wcs:.2f}")
        print(f"  Centroid     x={x_cent:.2f}, y={y_cent:.2f}")
        print(f"  Offset       dx={dx:.2f}, dy={dy:.2f}")

        plt.plot(x_wcs, y_wcs, "r+", ms=12, mew=2)
        plt.plot(x_cent, y_cent, "go", ms=6)
        plt.text(x_cent + 4, y_cent + 4, name, color="lime")

    plt.title("Stage 2: WCS guess (red) vs centroid (green)")
    plt.tight_layout()
    plt.show()
