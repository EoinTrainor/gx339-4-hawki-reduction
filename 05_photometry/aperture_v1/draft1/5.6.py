# title: Stage 3f – Measure FWHM from comparison stars (2D Gaussian fit) on reference FITS

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.modeling import models, fitting


# ---------------- USER INPUT ----------------
FITS_FILE = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

# Comparison star sky coordinates (from your message)
STARS = {
    "Star 1": (255.707140, -48.788802),
    "Star 2": (255.711256, -48.791347),
    "Star 3": (255.711298, -48.788757),
}

# Centroiding window (pixels)
CENTROID_HALF_SIZE = 10

# Gaussian fit window (cutout half-size, pixels)
FIT_HALF_SIZE = 12  # -> cutout is (2*12+1) = 25 px square

# Diagnostics
PLOT_FITS = True      # set False if you just want printed numbers
PLOT_CONTOURS = True  # overlay Gaussian contours on the data


# ---------------- HELPERS ----------------
def load_first_2d_hdu(fp: str):
    with fits.open(fp) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data is not None and np.ndim(h.data) == 2:
                data = h.data.astype(float)
                header = h.header
                break
    data[~np.isfinite(data)] = np.nan
    return data, WCS(header), header

def centroid_local_fluxweighted(data, x0, y0, half):
    """Flux-weighted centroid in a local square window around (x0,y0)."""
    x0i, y0i = int(round(float(x0))), int(round(float(y0)))
    ny, nx = data.shape

    x1, x2 = max(0, x0i - half), min(nx, x0i + half + 1)
    y1, y2 = max(0, y0i - half), min(ny, y0i + half + 1)

    cut = data[y1:y2, x1:x2]
    if cut.size == 0 or not np.isfinite(cut).any():
        return np.nan, np.nan

    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = np.isfinite(cut)
    flux = cut[mask]
    if flux.size == 0 or np.nansum(flux) == 0:
        return np.nan, np.nan

    xc = np.sum(xx[mask] * flux) / np.sum(flux)
    yc = np.sum(yy[mask] * flux) / np.sum(flux)
    return float(xc), float(yc)

def extract_cutout(data, x, y, half):
    ny, nx = data.shape
    x0, y0 = int(round(float(x))), int(round(float(y)))

    x1, x2 = max(0, x0 - half), min(nx, x0 + half + 1)
    y1, y2 = max(0, y0 - half), min(ny, y0 + half + 1)

    sub = data[y1:y2, x1:x2]
    return sub, x1, y1

def fit_gaussian_2d_with_const(sub):
    """
    Fit 2D Gaussian + constant background to a cutout.
    Returns fit model and fit quality metrics.
    """
    # Grid in cutout coordinates
    ny, nx = sub.shape
    y, x = np.mgrid[0:ny, 0:nx]

    # Mask invalid pixels
    mask = np.isfinite(sub)
    if mask.sum() < 50:
        return None, {"ok": False, "reason": "too few finite pixels"}

    z = sub.copy()
    z[~mask] = np.nan

    # Robust initial background estimate
    b0 = float(np.nanmedian(z))
    z0 = z - b0

    # Initial guesses
    amp0 = float(np.nanmax(z0)) if np.isfinite(np.nanmax(z0)) else 1.0
    if not np.isfinite(amp0) or amp0 <= 0:
        return None, {"ok": False, "reason": "non-positive amplitude guess"}

    # initial centre at brightest pixel
    iy, ix = np.unravel_index(np.nanargmax(z0), z0.shape)
    x_mean0, y_mean0 = float(ix), float(iy)

    # stddev initial guess (pixels)
    sig0 = 2.0

    g_init = models.Gaussian2D(
        amplitude=amp0,
        x_mean=x_mean0,
        y_mean=y_mean0,
        x_stddev=sig0,
        y_stddev=sig0,
        theta=0.0
    )
    c_init = models.Const2D(amplitude=b0)
    model_init = g_init + c_init

    fitter = fitting.LevMarLSQFitter()

    # Fit only finite pixels (flattened)
    x_fit = x[mask]
    y_fit = y[mask]
    z_fit = sub[mask]

    try:
        model_fit = fitter(model_init, x_fit, y_fit, z_fit)
    except Exception as e:
        return None, {"ok": False, "reason": f"fit failed: {e}"}

    # Extract fitted gaussian parameters
    g = model_fit[0]
    c = model_fit[1]

    # Sanity checks
    if (g.x_stddev.value <= 0) or (g.y_stddev.value <= 0):
        return None, {"ok": False, "reason": "non-positive stddev"}

    # Compute simple residual RMS for fit quality
    z_model = model_fit(x, y)
    resid = (sub - z_model)
    resid_rms = float(np.nanstd(resid))

    return model_fit, {
        "ok": True,
        "bkg": float(c.amplitude.value),
        "amp": float(g.amplitude.value),
        "x_mean": float(g.x_mean.value),
        "y_mean": float(g.y_mean.value),
        "x_std": float(g.x_stddev.value),
        "y_std": float(g.y_stddev.value),
        "theta": float(g.theta.value),
        "resid_rms": resid_rms
    }


# ---------------- MAIN ----------------
data, wcs, header = load_first_2d_hdu(FITS_FILE)

results = []

for name, (ra, dec) in STARS.items():
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    x_wcs, y_wcs = wcs.world_to_pixel(sky)
    x_wcs = float(np.squeeze(x_wcs))
    y_wcs = float(np.squeeze(y_wcs))

    # Refine by centroid first (important)
    x_cen, y_cen = centroid_local_fluxweighted(data, x_wcs, y_wcs, CENTROID_HALF_SIZE)

    sub, xoff, yoff = extract_cutout(data, x_cen, y_cen, FIT_HALF_SIZE)
    model_fit, info = fit_gaussian_2d_with_const(sub)

    if not info.get("ok", False):
        results.append({
            "star": name,
            "ok": False,
            "reason": info.get("reason", "unknown"),
            "x_cen": x_cen, "y_cen": y_cen
        })
        print(f"{name}: FIT FAILED -> {info.get('reason', 'unknown')}")
        continue

    # FWHM from sigma: FWHM = 2.355 * sigma
    fwhm_x = 2.355 * info["x_std"]
    fwhm_y = 2.355 * info["y_std"]
    fwhm_mean = 0.5 * (fwhm_x + fwhm_y)
    fwhm_geom = np.sqrt(fwhm_x * fwhm_y)

    results.append({
        "star": name,
        "ok": True,
        "x_cen": x_cen, "y_cen": y_cen,
        "x_std": info["x_std"], "y_std": info["y_std"],
        "fwhm_x": fwhm_x, "fwhm_y": fwhm_y,
        "fwhm_mean": fwhm_mean, "fwhm_geom": float(fwhm_geom),
        "bkg": info["bkg"], "amp": info["amp"],
        "resid_rms": info["resid_rms"]
    })

    print(f"\n{name}:")
    print(f"  Centroid (px): x={x_cen:.2f}, y={y_cen:.2f}")
    print(f"  Fit sigma (px): sx={info['x_std']:.3f}, sy={info['y_std']:.3f}")
    print(f"  FWHM (px): fx={fwhm_x:.3f}, fy={fwhm_y:.3f}")
    print(f"  FWHM mean={fwhm_mean:.3f}, geom={fwhm_geom:.3f}")
    print(f"  Fit quality: resid_rms={info['resid_rms']:.3f}")

    if PLOT_FITS:
        # Plot cutout with optional model contours
        ny, nx = sub.shape
        yy, xx = np.mgrid[0:ny, 0:nx]

        vmin = np.nanpercentile(sub, 5)
        vmax = np.nanpercentile(sub, 99.7)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.plot(info["x_mean"], info["y_mean"], "r+", ms=12, mew=2)
        ax.set_title(f"{name} – 2D Gaussian fit (FWHM mean={fwhm_mean:.2f} px)")
        ax.axis("off")

        if PLOT_CONTOURS and model_fit is not None:
            model_img = model_fit(xx, yy)
            # contour levels as fractions of peak above background
            peak = np.nanmax(model_img)
            levels = [peak * f for f in [0.2, 0.4, 0.6, 0.8]]
            ax.contour(model_img, levels=levels, linewidths=1)

        plt.tight_layout()
        plt.show()

# ---------------- SUMMARY ----------------
good = [r for r in results if r.get("ok", False)]
if len(good) == 0:
    raise RuntimeError("All FWHM fits failed. Increase FIT_HALF_SIZE or check star positions.")

fwhm_vals = np.array([r["fwhm_mean"] for r in good], dtype=float)
fwhm_med = float(np.nanmedian(fwhm_vals))
fwhm_iqr = float(np.nanpercentile(fwhm_vals, 75) - np.nanpercentile(fwhm_vals, 25))

print("\n================= FWHM SUMMARY (reference frame) =================")
print(f"Stars fitted: {len(good)} / {len(results)}")
print(f"FWHM mean values (px): {', '.join([f'{v:.2f}' for v in fwhm_vals])}")
print(f"Median FWHM (px): {fwhm_med:.3f}")
print(f"IQR FWHM (px):    {fwhm_iqr:.3f}")

# Suggest geometry from median FWHM
r_ap_15 = 1.5 * fwhm_med
r_ap_20 = 2.0 * fwhm_med
ann_in_3 = 3.0 * fwhm_med
ann_out_5 = 5.0 * fwhm_med

print("\nSuggested geometry from median FWHM:")
print(f"  Aperture ~1.5×FWHM: r_ap ≈ {r_ap_15:.2f} px")
print(f"  Aperture ~2.0×FWHM: r_ap ≈ {r_ap_20:.2f} px")
print(f"  Annulus: r_in ≈ 3×FWHM = {ann_in_3:.2f} px, r_out ≈ 5×FWHM = {ann_out_5:.2f} px")
print("===============================================================")
