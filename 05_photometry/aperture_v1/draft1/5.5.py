# title: Stage 3e – Sanity overlays + tight 2D (aperture × annulus) validation for 3 comparison stars

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# ---------------- USER INPUT ----------------
FITS_FILE = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

STARS = {
    "Star 1": (255.707140, -48.788802),
    "Star 2": (255.711256, -48.791347),
    "Star 3": (255.711298, -48.788757),
}

CENTROID_HALF_SIZE = 10
CUTOUT_HALF_SIZE = 60

# Baseline geometry for sanity overlay
AP_R_BASE = 6.0
ANN_BASE = (15.0, 30.0)

# Tight 2D sweep candidates
AP_RADII = [5.0, 6.0, 7.0, 8.0, 9.0]
ANNULI = [(15.0, 30.0), (25.0, 40.0), (22.0, 35.0)]  # keep or remove 22–35 if you want 2-only

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

    ny, nx = data.shape
    if x2 <= 0 or y2 <= 0 or x1 >= nx or y1 >= ny:
        return np.nan, np.nan

    y1c, y2c = max(0, y1), min(ny, y2)
    x1c, x2c = max(0, x1), min(nx, x2)

    cut = data[y1c:y2c, x1c:x2c]
    if cut.size == 0 or not np.isfinite(cut).any():
        return np.nan, np.nan

    yy, xx = np.mgrid[y1c:y2c, x1c:x2c]
    mask = np.isfinite(cut)
    flux = cut[mask]
    if flux.size == 0 or np.nansum(flux) == 0:
        return np.nan, np.nan

    xc = np.sum(xx[mask] * flux) / np.sum(flux)
    yc = np.sum(yy[mask] * flux) / np.sum(flux)
    return float(xc), float(yc)

def annulus_values(data, x, y, r_in, r_out):
    ann = CircularAnnulus([(x, y)], r_in=r_in, r_out=r_out)
    m = ann.to_mask(method="exact")[0]
    arr = m.multiply(data)
    vals = arr[m.data > 0]
    return vals[np.isfinite(vals)]

def snr3(net_flux, ap_area, sky_sigma, n_ann):
    if not np.isfinite(net_flux) or sky_sigma <= 0 or n_ann <= 0:
        return np.nan
    var = sky_sigma**2 * (ap_area + ap_area**2 / n_ann)
    return net_flux / np.sqrt(var) if var > 0 else np.nan

def cutout(data, x, y, half):
    ny, nx = data.shape
    x0, y0 = int(round(x)), int(round(y))
    x1, x2 = max(0, x0 - half), min(nx, x0 + half + 1)
    y1, y2 = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = data[y1:y2, x1:x2]
    return sub, x1, y1

# ---------------- MAIN ----------------
data, wcs = load_image(FITS_FILE)

# --- SANITY OVERLAYS ---
for star_name, (ra, dec) in STARS.items():
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    x_wcs, y_wcs = wcs.world_to_pixel(sky)
    x_wcs = float(np.squeeze(x_wcs))
    y_wcs = float(np.squeeze(y_wcs))

    x_cen, y_cen = centroid_local(data, x_wcs, y_wcs, CENTROID_HALF_SIZE)

    sub, xoff, yoff = cutout(data, x_cen, y_cen, CUTOUT_HALF_SIZE)
    xc, yc = x_cen - xoff, y_cen - yoff

    vmin = np.nanpercentile(sub, 5)
    vmax = np.nanpercentile(sub, 99.7)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    CircularAperture([(xc, yc)], r=AP_R_BASE).plot(ax=ax, color="yellow", lw=2)
    CircularAnnulus([(xc, yc)], r_in=ANN_BASE[0], r_out=ANN_BASE[1]).plot(ax=ax, color="cyan", lw=1.5)

    ax.plot([xc], [yc], "r+", ms=12, mew=2)
    ax.set_title(f"{star_name} overlay: ap={AP_R_BASE}px, ann={ANN_BASE[0]}–{ANN_BASE[1]}px")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

# --- TIGHT 2D SWEEP ---
rows = []
for star_name, (ra, dec) in STARS.items():
    sky = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    x_wcs, y_wcs = wcs.world_to_pixel(sky)
    x_wcs = float(np.squeeze(x_wcs))
    y_wcs = float(np.squeeze(y_wcs))

    x_cen, y_cen = centroid_local(data, x_wcs, y_wcs, CENTROID_HALF_SIZE)

    for ap_r in AP_RADII:
        ap = CircularAperture([(x_cen, y_cen)], r=ap_r)
        ap_tbl = aperture_photometry(data, ap)
        raw_sum = float(ap_tbl["aperture_sum"][0])
        ap_area = float(ap.area)

        for r_in, r_out in ANNULI:
            vals = annulus_values(data, x_cen, y_cen, r_in, r_out)
            n_ann = int(vals.size)
            if n_ann < 80:
                continue

            B = float(np.nanmedian(vals))
            sky_sigma = float(np.nanstd(vals))

            net_flux = raw_sum - B * ap_area
            snr = snr3(net_flux, ap_area, sky_sigma, n_ann)

            rows.append({
                "star": star_name,
                "ap_r": ap_r,
                "ann_in": r_in,
                "ann_out": r_out,
                "annulus": f"{int(r_in)}-{int(r_out)}",
                "n_ann": n_ann,
                "B": B,
                "sky_sigma": sky_sigma,
                "net_flux": net_flux,
                "snr3": snr
            })

df = pd.DataFrame(rows)
print("\n================= TOP 5 COMBOS PER STAR (2D sweep) =================")
for star in STARS.keys():
    sub = df[df["star"] == star].sort_values("snr3", ascending=False).head(5)
    print(f"\n{star}:")
    print(sub[["ap_r", "annulus", "snr3", "net_flux", "sky_sigma", "n_ann"]].to_string(index=False))

# Optional: print a global compromise based on mean rank
df["rank"] = df.groupby("star")["snr3"].rank(ascending=False, method="min")
combo = df.groupby(["ap_r", "annulus"])["rank"].mean().sort_values().head(10)
print("\n================= GLOBAL COMPROMISE (mean rank) =================")
print(combo.to_string())
