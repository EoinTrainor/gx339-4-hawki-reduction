# title: Stage 3 – Comparison star SNR vs aperture radius (single frame)

import numpy as np
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

# Comparison star (Star 1) sky coordinates from Stage 1
STAR_RA  =  255.711298   # <- paste value
STAR_DEC =  -48.788757   # <- paste value

# Aperture sweep (pixels)
AP_RADII = np.arange(2.0, 16.5, 0.5)

# Background annulus (fixed for this stage)
ANN_IN  = 20.0
ANN_OUT = 30.0

# Centroiding window
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

def compute_snr(raw_sum, ap_area, sky_sigma, n_ann):
    if sky_sigma <= 0 or n_ann <= 0:
        return np.nan
    var = sky_sigma**2 * (ap_area + ap_area**2 / n_ann)
    return raw_sum / np.sqrt(var) if var > 0 else np.nan

# ---------------- MAIN ----------------
data, wcs = load_image(FITS_FILE)

# Initial WCS guess
sky = SkyCoord(STAR_RA * u.deg, STAR_DEC * u.deg)
x_wcs, y_wcs = wcs.world_to_pixel(sky)
x_wcs = float(np.squeeze(x_wcs))
y_wcs = float(np.squeeze(y_wcs))

# Refine centroid
x_cen, y_cen = centroid_local(data, x_wcs, y_wcs, CENTROID_HALF_SIZE)

print("Comparison star centroid:")
print(f"  x = {x_cen:.2f}, y = {y_cen:.2f}")

snr_vals = []
flux_vals = []

for r in AP_RADII:
    ap = CircularAperture([(x_cen, y_cen)], r=r)
    ann = CircularAnnulus([(x_cen, y_cen)], r_in=ANN_IN, r_out=ANN_OUT)

    ap_tbl = aperture_photometry(data, ap)
    ann_tbl = aperture_photometry(data, ann)

    raw_sum = ap_tbl["aperture_sum"][0]
    ap_area = ap.area

    bkg_mean = ann_tbl["aperture_sum"][0] / ann.area
    ann_mask = ann.to_mask(method="exact")[0]
    ann_data = ann_mask.multiply(data)
    ann_vals = ann_data[ann_mask.data > 0]
    ann_vals = ann_vals[np.isfinite(ann_vals)]

    sky_sigma = np.nanstd(ann_vals)
    n_ann = ann_vals.size

    net_flux = raw_sum - bkg_mean * ap_area
    snr = compute_snr(net_flux, ap_area, sky_sigma, n_ann)

    flux_vals.append(net_flux)
    snr_vals.append(snr)

# ---------------- PLOT ----------------
plt.figure(figsize=(6, 4))
plt.plot(AP_RADII, snr_vals, "o-", lw=1.5)
plt.xlabel("Aperture radius (pixels)")
plt.ylabel("SNR")
plt.title("Comparison star SNR vs aperture radius")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
