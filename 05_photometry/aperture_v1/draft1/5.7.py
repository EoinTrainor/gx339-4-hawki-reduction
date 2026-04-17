# title: Stage 3g – Validate FWHM-anchored aperture/annulus choices on comparison stars (single frame)

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

# FWHM median from Stage 3f
FWHM_MED = 7.462

# Aperture candidates (rounded)
AP_RADII = [8.0, 11.0, 15.0]  # ~1.0×, ~1.5×, ~2.0×

# Annulus candidates: (r_in, r_out)
ANNULI = [
    (15.0, 30.0),                    # earlier local winner
    (round(3*FWHM_MED), round(5*FWHM_MED))  # FWHM-based ~22–37
]

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
    ny, nx = data.shape
    x1, x2 = max(0, x0i-half), min(nx, x0i+half+1)
    y1, y2 = max(0, y0i-half), min(ny, y0i+half+1)
    cut = data[y1:y2, x1:x2]
    if cut.size == 0 or not np.isfinite(cut).any():
        return np.nan, np.nan
    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = np.isfinite(cut)
    flux = cut[mask]
    if flux.size == 0 or np.nansum(flux) == 0:
        return np.nan, np.nan
    xc = np.sum(xx[mask]*flux)/np.sum(flux)
    yc = np.sum(yy[mask]*flux)/np.sum(flux)
    return float(xc), float(yc)

def annulus_vals(data, x, y, r_in, r_out):
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

# ---------------- MAIN ----------------
data, wcs = load_image(FITS_FILE)

rows = []

for star, (ra, dec) in STARS.items():
    sky = SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")
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
            vals = annulus_vals(data, x_cen, y_cen, r_in, r_out)
            n_ann = int(vals.size)
            if n_ann < 100:
                continue

            B = float(np.nanmedian(vals))
            sky_sigma = float(np.nanstd(vals))

            net_flux = raw_sum - B*ap_area
            snr = snr3(net_flux, ap_area, sky_sigma, n_ann)

            rows.append({
                "star": star,
                "ap_r": ap_r,
                "annulus": f"{int(r_in)}-{int(r_out)}",
                "B": B,
                "sky_sigma": sky_sigma,
                "n_ann": n_ann,
                "net_flux": net_flux,
                "snr3": snr
            })

df = pd.DataFrame(rows)
print("\n================= FWHM-BASED GEOMETRY COMPARISON =================")
print(df.sort_values(["star", "snr3"], ascending=[True, False]).to_string(index=False))

# Quick visual: compare SNR across apertures for each annulus
plt.figure(figsize=(7, 4))
for ann in df["annulus"].unique():
    sub = df[df["annulus"] == ann]
    # average across stars (for quick read)
    mean_snr = sub.groupby("ap_r")["snr3"].mean()
    plt.plot(mean_snr.index, mean_snr.values, "o-", label=f"Ann {ann}")
plt.xlabel("Aperture radius (px)")
plt.ylabel("Mean SNR3 across stars")
plt.title("FWHM-based candidates: mean SNR across comparison stars")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
