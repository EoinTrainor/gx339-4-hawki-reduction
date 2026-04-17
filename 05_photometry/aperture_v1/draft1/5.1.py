# title: Stage 1 – Reference FITS inspection, centroiding comparison stars, WCS verification

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

# ---------------- USER INPUT ----------------
REF_FITS = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

# GX 339-4 sky coordinates (fixed)
GX_RA_DEG  = 255.7057818297
GX_DEC_DEG = -48.7897466540

# Comparison star pixel boxes (x_min, x_max, y_min, y_max)
COMP_BOXES = {
    "Star 1": (1245, 1280, 1300, 1320),
    "Star 2": (1140, 1200, 1200, 1250),
    "Star 3": (1150, 1190, 1300, 1325),
}

# ---------------- LOAD FITS ----------------
with fits.open(REF_FITS) as hdul:
    for h in hdul:
        if h.data is not None and h.data.ndim == 2:
            data = h.data.astype(float)
            header = h.header
            break

data[~np.isfinite(data)] = np.nan
wcs = WCS(header)

print("Loaded reference FITS.")
print("Image shape:", data.shape)

# ---------------- GX POSITION ----------------
gx_coord = SkyCoord(GX_RA_DEG * u.deg, GX_DEC_DEG * u.deg, frame="icrs")
gx_x, gx_y = wcs.world_to_pixel(gx_coord)

print("\nGX 339-4 position:")
print(f"  Pixel: x = {gx_x:.2f}, y = {gx_y:.2f}")
print(f"  RA/Dec: {GX_RA_DEG:.6f}, {GX_DEC_DEG:.6f}")

# ---------------- COMPARISON STAR CENTROIDS ----------------
comp_results = {}

for name, (x1, x2, y1, y2) in COMP_BOXES.items():
    cut = data[y1:y2, x1:x2]

    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = np.isfinite(cut)

    flux = cut[mask]
    if flux.size == 0:
        raise RuntimeError(f"No valid pixels in box for {name}")

    x_cent = np.sum(xx[mask] * flux) / np.sum(flux)
    y_cent = np.sum(yy[mask] * flux) / np.sum(flux)

    sky = wcs.pixel_to_world(x_cent, y_cent)

    comp_results[name] = {
        "x": x_cent,
        "y": y_cent,
        "ra": sky.ra.deg,
        "dec": sky.dec.deg
    }

# ---------------- PRINT RESULTS ----------------
print("\nComparison star centroids:")
for name, r in comp_results.items():
    print(f"{name}:")
    print(f"  Pixel  x={r['x']:.2f}, y={r['y']:.2f}")
    print(f"  Sky    RA={r['ra']:.6f}, Dec={r['dec']:.6f}")

# ---------------- VISUAL CHECK ----------------
plt.figure(figsize=(7, 7))
plt.imshow(data, origin="lower", cmap="gray",
           vmin=np.nanpercentile(data, 5),
           vmax=np.nanpercentile(data, 99.7))

# GX marker
plt.plot(gx_x, gx_y, "r+", ms=12, mew=2, label="GX 339-4")

# Comparison stars
for name, r in comp_results.items():
    plt.plot(r["x"], r["y"], "go", ms=8)
    plt.text(r["x"] + 5, r["y"] + 5, name, color="lime")

# Draw boxes
for (x1, x2, y1, y2) in COMP_BOXES.values():
    plt.gca().add_patch(
        plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                      edgecolor="yellow", facecolor="none", lw=1.5)
    )

plt.legend()
plt.title("Stage 1: Reference FITS – GX and Comparison Stars")
plt.tight_layout()
plt.show()
