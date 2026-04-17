# title: GX 339-4 Circular Aperture Photometry + FWHM + Cropped Image Export + Viewer + CSV SNR + Console Stats

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clipped_stats
from astropy.modeling import models, fitting

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.centroids import centroid_2dg


# ---------------- USER INPUTS ----------------
FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"

OUT_BASE_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/4) Circular Aperture"
IMAGE_OUT_DIR = os.path.join(OUT_BASE_DIR, "aperture_cutouts")
CSV_OUT = os.path.join(OUT_BASE_DIR, "gx3394_aperture_photometry.csv")

# Ensure output dirs exist
os.makedirs(OUT_BASE_DIR, exist_ok=True)
os.makedirs(IMAGE_OUT_DIR, exist_ok=True)

# GX 339-4 coordinates (decimal degrees)
GX339_RA_DEG  = 255.7057818297
GX339_DEC_DEG = -48.7897466540

# Aperture geometry (pixels)
AP_RADIUS_PX = 6.0
ANN_IN_R_PX  = 10.0
ANN_OUT_R_PX = 16.0

# Cutout sizing (scaled to annulus)
CUTOUT_HALF_SIZE_PX = int(ANN_OUT_R_PX * 1.4)

# Image display stretch
PERCENTILE_STRETCH = (5, 99.7)


# ---------------- FIND FITS FILES ----------------
patterns = [
    os.path.join(FITS_DIR, "*.fits"),
    os.path.join(FITS_DIR, "*.fit"),
    os.path.join(FITS_DIR, "*.fits.fz"),
    os.path.join(FITS_DIR, "*.fz"),
]
fits_files = sorted({f for p in patterns for f in glob.glob(p)})

if len(fits_files) == 0:
    raise FileNotFoundError(f"No FITS files found in: {FITS_DIR}")

print(f"Found {len(fits_files)} FITS files.")


# ---------------- HELPERS ----------------
def load_first_2d_hdu(fp: str):
    """Load the first 2D image HDU and return data, WCS, header."""
    with fits.open(fp) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data is not None and np.ndim(h.data) == 2:
                data = h.data.astype(float)
                header = h.header
                wcs = WCS(header)
                data[~np.isfinite(data)] = np.nan
                return data, wcs, header
    raise ValueError("No 2D image HDU found")


def world_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(c)
    return float(x), float(y)


def cutout(data: np.ndarray, x: float, y: float, half: int):
    ny, nx = data.shape
    x0, y0 = int(round(x)), int(round(y))
    x1, x2 = max(0, x0 - half), min(nx, x0 + half + 1)
    y1, y2 = max(0, y0 - half), min(ny, y0 + half + 1)
    return data[y1:y2, x1:x2], x1, y1


def measure_fwhm(data: np.ndarray, x: float, y: float, half: int = 25):
    """
    Estimate FWHM (pixels) at (x,y) using a 2D Gaussian fit.
    Returns np.nan if fit fails or cutout is too empty.
    """
    cut, _, _ = cutout(data, x, y, half)
    finite = cut[np.isfinite(cut)]
    if finite.size < 200:
        return np.nan

    _, med, _ = sigma_clipped_stats(finite)
    cut_b = cut - med
    cut_b[~np.isfinite(cut_b)] = 0.0

    try:
        cy, cx = centroid_2dg(cut_b)
    except Exception:
        return np.nan

    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]

    g0 = models.Gaussian2D(
        amplitude=float(np.nanmax(cut_b)),
        x_mean=float(cx),
        y_mean=float(cy),
        x_stddev=2.0,
        y_stddev=2.0
    )
    c0 = models.Const2D(amplitude=float(med))
    model = g0 + c0

    fitter = fitting.LevMarLSQFitter()

    try:
        fitted = fitter(model, xx, yy, cut, weights=np.isfinite(cut))
        g = fitted[0]
        sx = float(abs(g.x_stddev.value))
        sy = float(abs(g.y_stddev.value))
        return float(2.354820045 * 0.5 * (sx + sy))
    except Exception:
        return np.nan


def safe_percentile_limits(arr: np.ndarray, lo: float, hi: float):
    """Return vmin/vmax safely even if array is empty or constant."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        # Fallback if image is flat / constant
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)


# ---------------- MAIN LOOP ----------------
rows = []

for i, fp in enumerate(fits_files, start=1):
    try:
        data, wcs, header = load_first_2d_hdu(fp)
        gx_x, gx_y = world_to_pixel(wcs, GX339_RA_DEG, GX339_DEC_DEG)

        fwhm_px = measure_fwhm(data, gx_x, gx_y)

        # Aperture photometry
        ap = CircularAperture([(gx_x, gx_y)], r=AP_RADIUS_PX)
        ann = CircularAnnulus([(gx_x, gx_y)], r_in=ANN_IN_R_PX, r_out=ANN_OUT_R_PX)

        ap_tbl = aperture_photometry(data, ap)
        raw_sum = float(ap_tbl["aperture_sum"][0])

        ann_mask = ann.to_mask(method="exact")[0]
        ann_data = ann_mask.multiply(data)
        ann_vals = ann_data[ann_mask.data > 0]
        ann_vals = ann_vals[np.isfinite(ann_vals)]

        if ann_vals.size < 50:
            sky_med = np.nan
            sky_std = np.nan
        else:
            _, sky_med, sky_std = sigma_clipped_stats(ann_vals)

        net_sum = np.nan
        net_err = np.nan
        snr = np.nan

        if np.isfinite(sky_med) and np.isfinite(sky_std):
            net_sum = raw_sum - float(sky_med) * float(ap.area)
            net_err = float(sky_std) * np.sqrt(float(ap.area))
            if np.isfinite(net_err) and net_err > 0:
                snr = net_sum / net_err

        # Cutout for image saving
        sub, xoff, yoff = cutout(data, gx_x, gx_y, CUTOUT_HALF_SIZE_PX)
        xc, yc = gx_x - xoff, gx_y - yoff

        vmin, vmax = safe_percentile_limits(sub, PERCENTILE_STRETCH[0], PERCENTILE_STRETCH[1])

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

        CircularAperture([(xc, yc)], r=AP_RADIUS_PX).plot(ax=ax, color="yellow", lw=2.2)
        CircularAnnulus([(xc, yc)], r_in=ANN_IN_R_PX, r_out=ANN_OUT_R_PX).plot(
            ax=ax, color="yellow", lw=1.3, alpha=0.8
        )

        ax.axis("off")
        out_png = os.path.join(IMAGE_OUT_DIR, f"gx3394_{i:04d}.png")
        plt.tight_layout(pad=0.3)
        plt.savefig(out_png, dpi=200)
        plt.close(fig)

        rows.append({
            "filename": os.path.basename(fp),
            "cutout_image": os.path.basename(out_png),
            "date_obs": header.get("DATE-OBS", ""),
            "mjd_obs": header.get("MJD-OBS", np.nan),
            "gx_x_px": gx_x,
            "gx_y_px": gx_y,
            "ap_radius_px": AP_RADIUS_PX,
            "ann_in_px": ANN_IN_R_PX,
            "ann_out_px": ANN_OUT_R_PX,
            "fwhm_px": fwhm_px,
            "raw_flux": raw_sum,
            "sky_median": sky_med,
            "sky_std": sky_std,
            "net_flux": net_sum,
            "net_flux_err": net_err,
            "snr": snr
        })

    except Exception as e:
        rows.append({
            "filename": os.path.basename(fp),
            "error": str(e)
        })


# ---------------- SAVE CSV ----------------
df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
print(f"Saved CSV: {CSV_OUT}")


# =============================================================================
# Analysis (Console stats)
# =============================================================================
print("\n================= SUMMARY STATISTICS =================")

good = df[df["net_flux"].notna() & df["net_flux_err"].notna()].copy()

print(f"Total frames processed: {len(df)}")
print(f"Frames with valid photometry: {len(good)}")

# FWHM
valid_fwhm = good["fwhm_px"].dropna()
if len(valid_fwhm) > 0:
    print("\n--- FWHM (pixels) ---")
    print(f"Median FWHM: {np.nanmedian(valid_fwhm):.3f}")
    print(f"Min FWHM:    {np.nanmin(valid_fwhm):.3f}")
    print(f"Max FWHM:    {np.nanmax(valid_fwhm):.3f}")
else:
    print("\nNo valid FWHM measurements.")

# Flux
net_flux = good["net_flux"].values
net_err = good["net_flux_err"].values
snr = good["snr"].values

print("\n--- Net Flux (instrumental units) ---")
print(f"Median flux: {np.nanmedian(net_flux):.3e}")
print(f"Min flux:    {np.nanmin(net_flux):.3e}")
print(f"Max flux:    {np.nanmax(net_flux):.3e}")
print(f"Std flux:    {np.nanstd(net_flux):.3e}")

print("\n--- Signal-to-Noise ---")
print(f"Median SNR:  {np.nanmedian(snr):.2f}")
print(f"Min SNR:     {np.nanmin(snr):.2f}")
print(f"Max SNR:     {np.nanmax(snr):.2f}")

print("\n======================================================")


# ---------------- IMAGE VIEWER (saved PNGs) ----------------
images = sorted(glob.glob(os.path.join(IMAGE_OUT_DIR, "*.png")))

if len(images) == 0:
    print("No cutout PNGs found to display.")
else:
    idx = 0
    fig, ax = plt.subplots(figsize=(5, 5))

    def draw(i):
        ax.clear()
        img = mpimg.imread(images[i])
        ax.imshow(img)
        ax.set_title(f"[{i+1}/{len(images)}] {os.path.basename(images[i])}")
        ax.axis("off")
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal_idx = None  # just to prevent accidental shadowing
        global idx
        if event.key == "right":
            idx = min(idx + 1, len(images) - 1)
            draw(idx)
        elif event.key == "left":
            idx = max(idx - 1, 0)
            draw(idx)
        elif event.key in ("enter", "return"):
            plt.close(fig)

    draw(idx)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()

    print("Controls: ← previous | → next | Enter quit")