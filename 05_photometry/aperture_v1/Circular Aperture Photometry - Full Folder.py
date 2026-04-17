# title: Circular Aperture Photometry for GX 339-4 (HAWK-I FITS) + CSV Output (Fixed + Clean)

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry
from astropy.stats import sigma_clipped_stats

from astropy.modeling import models, fitting
from photutils.centroids import centroid_2dg

# ---------------- USER INPUTS ----------------
# FITS folder (your confirmed path)
FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"

# GX 339-4 (decimal degrees)
GX339_RA_DEG  = 255.7057818297
GX339_DEC_DEG = -48.7897466540

# Aperture/annulus radii (pixels) - start here, refine after checking FWHM
AP_RADIUS_PX = 6.0
ANN_IN_R_PX  = 10.0
ANN_OUT_R_PX = 16.0

# Comparison star 
USE_COMPARISON_STAR = False
COMP_RA_DEG  = 0.0
COMP_DEC_DEG = 0.0

# Output CSV
OUT_CSV = "C:/Users/40328449/OneDrive - University College Cork/GX 339-4/4) Circular Aperture/gx3394_aperture_photometry.csv"

# Quick-look plot
PLOT_RANDOM_EXAMPLE = True
RANDOM_SEED = 7


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

print("Folder exists:", os.path.isdir(FITS_DIR))
print("Found FITS files:", len(fits_files))
if len(fits_files) == 0:
    raise FileNotFoundError(
        "No FITS files found. Check FITS_DIR or file extensions.\n"
        f"FITS_DIR: {FITS_DIR}\n"
        f"Patterns: {patterns}"
    )


# ---------------- HELPERS ----------------
def load_fits_data_and_wcs(fits_path: str):
    """Loads the first 2D image HDU and returns (data, wcs, header)."""
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
            raise ValueError(f"No 2D image data found in: {fits_path}")

        data = hdul[hdu_idx].data.astype(float)
        header = hdul[hdu_idx].header
        wcs = WCS(header)

    data = np.array(data, dtype=float)
    data[~np.isfinite(data)] = np.nan
    return data, wcs, header


def sky_background_from_annulus(data: np.ndarray, annulus_mask):
    """Sigma-clipped background stats from annulus pixels."""
    annulus_data = annulus_mask.multiply(data)
    annulus_1d = annulus_data[annulus_mask.data > 0]
    annulus_1d = annulus_1d[np.isfinite(annulus_1d)]

    if annulus_1d.size < 50:
        return np.nan, np.nan, np.nan

    mean, median, std = sigma_clipped_stats(annulus_1d, sigma=3.0, maxiters=10)
    return float(mean), float(median), float(std)


def aperture_flux_and_error(data: np.ndarray, x: float, y: float,
                            ap_r: float, ann_in: float, ann_out: float):
    """Background-subtracted aperture sum + conservative uncertainty estimate."""
    pos = [(x, y)]
    ap = CircularAperture(pos, r=ap_r)
    ann = CircularAnnulus(pos, r_in=ann_in, r_out=ann_out)

    ann_mask = ann.to_mask(method="exact")[0]
    tbl = aperture_photometry(data, ap)
    raw_sum = float(tbl["aperture_sum"][0])

    sky_mean, sky_median, sky_std = sky_background_from_annulus(data, ann_mask)
    if not np.isfinite(sky_median):
        return raw_sum, np.nan, np.nan, np.nan, np.nan, np.nan

    ap_area = float(ap.area)
    bkg_sum = sky_median * ap_area
    net_sum = raw_sum - bkg_sum

    # Conservative error: background scatter * sqrt(Npix)
    net_err = sky_std * np.sqrt(ap_area)

    return raw_sum, net_sum, net_err, sky_mean, sky_median, sky_std


def world_to_pixel_safe(wcs: WCS, ra_deg: float, dec_deg: float):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(c)
    return float(x), float(y)

def measure_fwhm(data, x, y, half_size=25):
    """
    Measure FWHM at (x, y) using a 2D Gaussian fit.
    Returns FWHM in pixels, or np.nan if fit fails.
    """
    ny, nx = data.shape
    x0 = int(np.round(x))
    y0 = int(np.round(y))

    x1 = max(0, x0 - half_size)
    x2 = min(nx, x0 + half_size + 1)
    y1 = max(0, y0 - half_size)
    y2 = min(ny, y0 + half_size + 1)

    cut = data[y1:y2, x1:x2]
    cut[~np.isfinite(cut)] = np.nan

    if np.sum(np.isfinite(cut)) < 100:
        return np.nan

    # Background stats
    mean, med, std = sigma_clipped_stats(cut, sigma=3.0)

    # Subtract background for centroiding
    cut_b = cut - med
    cut_b[~np.isfinite(cut_b)] = 0.0

    try:
        cy, cx = centroid_2dg(cut_b)
    except Exception:
        return np.nan

    yy, xx = np.mgrid[0:cut.shape[0], 0:cut.shape[1]]

    g_init = models.Gaussian2D(
        amplitude=np.nanmax(cut_b),
        x_mean=cx,
        y_mean=cy,
        x_stddev=2.0,
        y_stddev=2.0
    )
    c_init = models.Const2D(amplitude=med)
    model = g_init + c_init

    fit = fitting.LevMarLSQFitter()
    weights = np.isfinite(cut).astype(float)

    try:
        fitted = fit(model, xx, yy, cut, weights=weights)
        g = fitted[0]
        sx = abs(g.x_stddev.value)
        sy = abs(g.y_stddev.value)
        fwhm = 2.3548 * 0.5 * (sx + sy)
        return float(fwhm)
    except Exception:
        return np.nan

# ---------------- MAIN PHOTOMETRY ----------------
rows = []
gx = SkyCoord(ra=GX339_RA_DEG * u.deg, dec=GX339_DEC_DEG * u.deg, frame="icrs")

for fp in fits_files:
    try:
        data, wcs, header = load_fits_data_and_wcs(fp)

        gx_x, gx_y = world_to_pixel_safe(wcs, GX339_RA_DEG, GX339_DEC_DEG)
        fwhm_px = measure_fwhm(data, gx_x, gx_y)

        raw_sum, net_sum, net_err, sky_mean, sky_median, sky_std = aperture_flux_and_error(
            data=data, x=gx_x, y=gx_y,
            ap_r=AP_RADIUS_PX, ann_in=ANN_IN_R_PX, ann_out=ANN_OUT_R_PX
        )

        comp_raw = comp_net = comp_err = np.nan
        if USE_COMPARISON_STAR:
            comp_x, comp_y = world_to_pixel_safe(wcs, COMP_RA_DEG, COMP_DEC_DEG)
            comp_raw, comp_net, comp_err, _, _, _ = aperture_flux_and_error(
                data=data, x=comp_x, y=comp_y,
                ap_r=AP_RADIUS_PX, ann_in=ANN_IN_R_PX, ann_out=ANN_OUT_R_PX
            )

        flux_ratio = np.nan
        if USE_COMPARISON_STAR and np.isfinite(net_sum) and np.isfinite(comp_net) and comp_net != 0:
            flux_ratio = net_sum / comp_net

        # Extract observation time if present (optional, helps later)
        date_obs = header.get("DATE-OBS", "")
        mjd_obs = header.get("MJD-OBS", np.nan)

        rows.append({
            "filename": os.path.basename(fp),
            "date_obs": date_obs,
            "mjd_obs": mjd_obs,
            "gx_x_px": gx_x,
            "gx_y_px": gx_y,
            "fwhm_px": fwhm_px,
            "ap_radius_px": AP_RADIUS_PX,
            "ann_in_px": ANN_IN_R_PX,
            "ann_out_px": ANN_OUT_R_PX,
            "raw_sum": raw_sum,
            "net_sum": net_sum,
            "net_err": net_err,
            "sky_mean": sky_mean,
            "sky_median": sky_median,
            "sky_std": sky_std,
            "comp_raw_sum": comp_raw,
            "comp_net_sum": comp_net,
            "comp_net_err": comp_err,
            "gx_to_comp_flux_ratio": flux_ratio
        })

    except Exception as e:
        rows.append({
            "filename": os.path.basename(fp),
            "error": str(e)
        })

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV, index=False)
print(f"Saved aperture photometry results to: {os.path.abspath(OUT_CSV)}")


# ---------------- QUICK-LOOK PLOT (one random file) ----------------
if PLOT_RANDOM_EXAMPLE:
    good = df[df["net_sum"].notna() & df["sky_median"].notna()]
    if len(good) > 0:
        np.random.seed(RANDOM_SEED)
        pick = good.sample(1).iloc[0]

        pick_fp = None
        for fp in fits_files:
            if os.path.basename(fp) == pick["filename"]:
                pick_fp = fp
                break

        data, wcs, _ = load_fits_data_and_wcs(pick_fp)
        gx_x, gx_y = float(pick["gx_x_px"]), float(pick["gx_y_px"])

        fig, ax = plt.subplots(figsize=(7, 7))
        vmin, vmax = np.nanpercentile(data, [5, 99.5])
        ax.imshow(data, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"GX 339-4 Aperture Check\n{pick['filename']}")

        ap = CircularAperture([(gx_x, gx_y)], r=AP_RADIUS_PX)
        ann = CircularAnnulus([(gx_x, gx_y)], r_in=ANN_IN_R_PX, r_out=ANN_OUT_R_PX)

        ap.plot(ax=ax, lw=2)
        ann.plot(ax=ax, lw=1)

        ax.scatter([gx_x], [gx_y], s=40)
        plt.tight_layout()
        plt.show()
    else:
        print("No valid rows to plot (all rows failed or missing net_sum).")
