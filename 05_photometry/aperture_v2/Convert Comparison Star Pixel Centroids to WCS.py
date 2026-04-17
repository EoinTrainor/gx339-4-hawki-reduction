# title: Stage A – Convert Comparison Star Pixel Centroids to WCS (RA/Dec)

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

FITS_FILE = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

# Pixel centroids from your console output
STAR_XY = {
    "Star 1": (1536.11, 1542.53),
    "Star 2": (1054.14, 1406.32),
    "Star 3": (2114.45, 1431.80),
    "Star 4": (1240.65, 1342.06),
    "Star 5": (1399.82,  900.78),
}

# Optional: GX target too
GX_XY = (1292.5, 1278.5)

def load_first_2d_header(fp: str):
    with fits.open(fp) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data is not None and np.ndim(h.data) == 2:
                return h.header
    raise ValueError("No 2D image HDU found in FITS file.")

hdr = load_first_2d_header(FITS_FILE)
w = WCS(hdr)

print("\nWCS solution summary:")
print(w)

print("\nStar WCS coordinates (ICRS):")
print(f"{'Object':<8} {'x(px)':>10} {'y(px)':>10} {'RA(deg)':>14} {'Dec(deg)':>14} {'RA(hms)':>16} {'Dec(dms)':>16}")

for name, (x, y) in STAR_XY.items():
    # Astropy WCS expects origin=0 for 0-based pixel coordinates (like your centroid outputs)
    sky = w.pixel_to_world(x, y)  # returns a SkyCoord-like object for celestial WCS
    ra_deg = float(sky.ra.deg)
    dec_deg = float(sky.dec.deg)
    ra_hms = sky.ra.to_string(unit="hour", sep=":", precision=2, pad=True)
    dec_dms = sky.dec.to_string(unit="deg", sep=":", precision=2, alwayssign=True, pad=True)

    print(f"{name:<8} {x:>10.2f} {y:>10.2f} {ra_deg:>14.8f} {dec_deg:>14.8f} {ra_hms:>16} {dec_dms:>16}")

# GX 339-4 if you want it
gx, gy = GX_XY
sky_gx = w.pixel_to_world(gx, gy)
print("\nGX 339-4 WCS (ICRS):")
print("  RA(deg) =", float(sky_gx.ra.deg), "| Dec(deg) =", float(sky_gx.dec.deg))
print("  RA(hms) =", sky_gx.ra.to_string(unit="hour", sep=":", precision=2, pad=True))
print("  Dec(dms)=", sky_gx.dec.to_string(unit="deg", sep=":", precision=2, alwayssign=True, pad=True))
