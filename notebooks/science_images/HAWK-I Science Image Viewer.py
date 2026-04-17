# title: HAWK-I Science Image Viewer (single image, greyscale, DATE-OBS title)

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits


# ---------------- USER INPUTS ----------------
FITS_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"
# --------------------------------------------


def robust_limits(img, p_lo=1, p_hi=99):
    finite = np.isfinite(img)
    if not np.any(finite):
        return None, None
    vmin = np.nanpercentile(img[finite], p_lo)
    vmax = np.nanpercentile(img[finite], p_hi)
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def first_2d_image_hdu(hdul):
    """
    Return (hdu_index, image_2d, header) for the first HDU containing 2D image data.
    """
    for i, hdu in enumerate(hdul):
        if hdu.data is None:
            continue

        data = np.array(hdu.data)
        while data.ndim > 2:
            data = data[0]

        if data.ndim == 2:
            return i, data, hdu.header

    return None


def find_date_obs_any_hdu(hdul):
    """
    Prefer DATE-OBS in primary header, otherwise search extensions.
    """
    if "DATE-OBS" in hdul[0].header:
        return str(hdul[0].header["DATE-OBS"]).strip()

    for hdu in hdul[1:]:
        if hdu.header is not None and "DATE-OBS" in hdu.header:
            return str(hdu.header["DATE-OBS"]).strip()

    return "DATE-OBS not found"


def main():
    if not os.path.exists(FITS_FILE):
        raise FileNotFoundError(f"File not found:\n{FITS_FILE}")

    with fits.open(FITS_FILE, memmap=False) as hdul:
        date_obs = find_date_obs_any_hdu(hdul)

        result = first_2d_image_hdu(hdul)
        if result is None:
            raise RuntimeError("No 2D image HDU found in this FITS file.")

        idx, img, hdr = result

        vmin, vmax = robust_limits(img, 1, 99)

        plt.figure(figsize=(10, 8))
        plt.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        plt.title(f"DATE-OBS: {date_obs}", fontsize=14)
        plt.xlabel("X (pix)")
        plt.ylabel("Y (pix)")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
