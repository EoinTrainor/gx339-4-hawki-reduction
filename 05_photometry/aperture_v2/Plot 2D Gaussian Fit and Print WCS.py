# title: Stage A Diagnostic – Plot 2D Gaussian Fit + Print WCS RA Dec (Chosen Star or All 5)

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting

# ---------------- USER INPUTS ----------------
FITS_FILE = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

# Candidate star boxes (same as Stage A)
STAR_BOXES = {
    "Star 1": (1500, 1580, 1520, 1580),
    "Star 2": (1020, 1100, 1370, 1450),
    "Star 3": (2080, 2150, 1410, 1460),
    "Star 4": (1210, 1270, 1310, 1370),
    "Star 5": (1360, 1430,  880,  940),
}

# Choose ONE star to plot, or set to "ALL" to print WCS for all 5 but only plot STAR_NAME
STAR_NAME = "Star 5"
PRINT_WCS_FOR_ALL_STARS = True  # prints a 5-star WCS table every run

FIT_HALF_SIZE = 10
STRETCH = (5, 99.7)
# ---------------- END INPUTS ----------------


# ---------------- HELPERS ----------------
def load_first_2d_hdu(fp: str):
    with fits.open(fp) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data is not None and np.ndim(h.data) == 2:
                data = h.data.astype(float)
                data[~np.isfinite(data)] = np.nan
                return data, h.header
    raise ValueError("No 2D image HDU found in FITS file.")

def safe_percentile_limits(arr, lo, hi):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)

def extract_cutout(data, x, y, half):
    ny, nx = data.shape
    x0, y0 = int(round(float(x))), int(round(float(y)))
    x1, x2 = max(0, x0 - half), min(nx, x0 + half + 1)
    y1, y2 = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = data[y1:y2, x1:x2]
    return sub, x1, y1

def centroid_bgsub_in_box(data, x1, x2, y1, y2):
    """Peak seed + local median bgsub + weighted centroid (same idea as Stage A)."""
    cut = data[y1:y2, x1:x2]
    if cut.size == 0 or not np.isfinite(cut).any():
        return np.nan, np.nan

    finite = np.isfinite(cut)
    idx = np.nanargmax(np.where(finite, cut, -np.inf))
    iy, ix = np.unravel_index(idx, cut.shape)
    x_seed = x1 + ix
    y_seed = y1 + iy

    half = 10
    win, xoff, yoff = extract_cutout(data, x_seed, y_seed, half)
    if win.size == 0 or not np.isfinite(win).any():
        return float(x_seed), float(y_seed)

    yy, xx = np.mgrid[yoff:yoff + win.shape[0], xoff:xoff + win.shape[1]]
    m = np.isfinite(win)

    bkg = float(np.nanmedian(win[m]))
    w = win[m] - bkg
    w[w < 0] = 0

    if w.size == 0 or np.nansum(w) == 0:
        return float(x_seed), float(y_seed)

    xc = float(np.sum(xx[m] * w) / np.sum(w))
    yc = float(np.sum(yy[m] * w) / np.sum(w))
    return xc, yc

def xy_to_radec_deg(wcs_obj, x, y):
    """
    Pixel (x,y) -> ICRS RA/Dec in degrees.
    Uses origin=0 convention implicitly via pixel_to_world (matches 0-based centroids).
    """
    try:
        sky = wcs_obj.pixel_to_world(float(x), float(y))
        return float(sky.ra.deg), float(sky.dec.deg)
    except Exception:
        return np.nan, np.nan

def fit_2d_gaussian(data, x_cen, y_cen, half=10):
    """
    Fits Gaussian2D + Const2D to a local cutout.

    Returns:
      fit, sub, xoff, yoff, mask, model_img, resid, fwhm_x, fwhm_y, fwhm_mean, resid_rms
    """
    sub, xoff, yoff = extract_cutout(data, x_cen, y_cen, half)
    if sub.size == 0 or not np.isfinite(sub).any():
        raise RuntimeError("Fit cutout empty or non-finite.")

    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]

    amp0 = np.nanmax(sub) - np.nanmedian(sub)
    bkg0 = np.nanmedian(sub)
    x0 = (x_cen - xoff)
    y0 = (y_cen - yoff)
    sig0 = max(1.0, half / 3)

    model0 = (
        models.Gaussian2D(amplitude=amp0, x_mean=x0, y_mean=y0,
                          x_stddev=sig0, y_stddev=sig0)
        + models.Const2D(amplitude=bkg0)
    )

    fitter = fitting.LevMarLSQFitter()
    mask = np.isfinite(sub)
    if mask.sum() < 20:
        raise RuntimeError("Not enough finite pixels to fit.")

    fit = fitter(model0, xx[mask], yy[mask], sub[mask])

    sx = float(np.abs(fit[0].x_stddev.value))
    sy = float(np.abs(fit[0].y_stddev.value))
    fwhm_x = 2.35482 * sx
    fwhm_y = 2.35482 * sy
    fwhm_mean = 0.5 * (fwhm_x + fwhm_y)

    model_img = np.full_like(sub, np.nan, dtype=float)
    model_img[mask] = fit(xx[mask], yy[mask])
    resid = sub - model_img
    resid_rms = float(np.sqrt(np.nanmean((resid[mask]) ** 2)))

    return fit, sub, xoff, yoff, mask, model_img, resid, fwhm_x, fwhm_y, fwhm_mean, resid_rms


# ---------------- RUN ----------------
data, header = load_first_2d_hdu(FITS_FILE)
wcs = WCS(header)

# Print WCS for all 5 comps (this is the “whole code fix” you wanted)
if PRINT_WCS_FOR_ALL_STARS:
    print("\nObject      x(px)      y(px)        RA(deg)        Dec(deg)")
    print("------------------------------------------------------------")
    for nm, (x1, x2, y1, y2) in STAR_BOXES.items():
        xc, yc = centroid_bgsub_in_box(data, x1, x2, y1, y2)
        ra, dec = xy_to_radec_deg(wcs, xc, yc)
        print(f"{nm:<8} {xc:>9.2f} {yc:>9.2f} {ra:>13.8f} {dec:>13.8f}")

# Proceed with plotting for the chosen STAR_NAME
x1, x2, y1, y2 = STAR_BOXES[STAR_NAME]
x_cen, y_cen = centroid_bgsub_in_box(data, x1, x2, y1, y2)

ra_deg, dec_deg = xy_to_radec_deg(wcs, x_cen, y_cen)

print("\n==============================")
print("FITS:", FITS_FILE)
print("Star:", STAR_NAME)
print(f"Centroid estimate: x={x_cen:.3f}, y={y_cen:.3f}")
print(f"WCS (ICRS):        RA={ra_deg:.8f} deg | Dec={dec_deg:.8f} deg")
print("==============================")

fit, sub, xoff, yoff, mask, model_img, resid, fwhm_x, fwhm_y, fwhm_mean, resid_rms = fit_2d_gaussian(
    data, x_cen, y_cen, half=FIT_HALF_SIZE
)

# Fit parameter readout
A = float(fit[0].amplitude.value)
x0 = float(fit[0].x_mean.value)
y0 = float(fit[0].y_mean.value)
sx = float(np.abs(fit[0].x_stddev.value))
sy = float(np.abs(fit[0].y_stddev.value))
C = float(fit[1].amplitude.value)

axis_ratio = max(fwhm_x, fwhm_y) / max(1e-9, min(fwhm_x, fwhm_y))

print("\n---- Gaussian fit params (local cutout coords) ----")
print(f"A (amplitude)    = {A:.3f} ADU")
print(f"C (background)   = {C:.3f} ADU")
print(f"x0, y0           = {x0:.3f}, {y0:.3f} px  (cutout coords)")
print(f"sigma_x, sigma_y = {sx:.3f}, {sy:.3f} px")
print(f"FWHM_x, FWHM_y    = {fwhm_x:.3f}, {fwhm_y:.3f} px")
print(f"FWHM_mean         = {fwhm_mean:.3f} px")
print(f"axis_ratio        = {axis_ratio:.3f}")
print(f"residual RMS      = {resid_rms:.3f} ADU")

# ---------------- PLOTS ----------------
vmin, vmax = safe_percentile_limits(sub, *STRETCH)
vmin_r, vmax_r = safe_percentile_limits(resid, 1, 99)

fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6))

axs[0].imshow(sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
axs[0].set_title(f"{STAR_NAME} data (cutout)\ncentroid + fit centre\nRA={ra_deg:.6f} Dec={dec_deg:.6f}")
axs[0].plot([x_cen - xoff], [y_cen - yoff], "r+", ms=12, mew=2, label="centroid")
axs[0].plot([x0], [y0], "c+", ms=12, mew=2, label="fit centre")
axs[0].set_axis_off()

axs[1].imshow(model_img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
axs[1].set_title("Gaussian2D + Const2D model")
axs[1].plot([x0], [y0], "c+", ms=12, mew=2)
axs[1].set_axis_off()

axs[2].imshow(resid, origin="lower", cmap="inferno", vmin=vmin_r, vmax=vmax_r)
axs[2].set_title(f"Residual (data − model)\nRMS = {resid_rms:.2f} ADU")
axs[2].set_axis_off()

plt.tight_layout()
plt.show()

# Optional: 1D cuts through the fitted centre
cx = int(round(x0))
cy = int(round(y0))
if 0 <= cy < sub.shape[0] and 0 <= cx < sub.shape[1]:
    row_data = sub[cy, :]
    row_model = model_img[cy, :]
    col_data = sub[:, cx]
    col_model = model_img[:, cx]

    fig2, ax2 = plt.subplots(figsize=(8.6, 4.6))
    ax2.plot(row_data, marker="o", ms=3, linestyle="-", label="Data row through y0")
    ax2.plot(row_model, linestyle="--", label="Model row through y0")
    ax2.set_title(f"1D slice through fitted centre (row) – {STAR_NAME}")
    ax2.set_xlabel("x (cutout pixels)")
    ax2.set_ylabel("ADU")
    ax2.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(8.6, 4.6))
    ax3.plot(col_data, marker="o", ms=3, linestyle="-", label="Data col through x0")
    ax3.plot(col_model, linestyle="--", label="Model col through x0")
    ax3.set_title(f"1D slice through fitted centre (column) – {STAR_NAME}")
    ax3.set_xlabel("y (cutout pixels)")
    ax3.set_ylabel("ADU")
    ax3.legend(frameon=False)
    plt.tight_layout()
    plt.show()
