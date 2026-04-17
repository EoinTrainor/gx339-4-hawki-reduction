# Visualise only the decompressed HAWK-I FITS files (ignore .fits.Z and text files)

import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/7) ZOGY Difference Imaging Pipeline/1) Reference Frame PSF and Flux Scale/1.2) Reference Frame Construction"

WANTED = [
    "C:/Users/40328449/OneDrive - University College Cork/GX 339-4/7) ZOGY Difference Imaging Pipeline/1) Reference Frame PSF and Flux Scale/1.2) Reference Frame Construction/reference_coadd.fits"
]


def pick_2d(data):
    """Return 2D image from FITS data, handling >2D by taking the first plane(s)."""
    if data is None:
        return None
    arr = np.array(data)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def robust_limits(img, p_lo=1, p_hi=99):
    """Percentile stretch for nicer viewing."""
    finite = np.isfinite(img)
    if not np.any(finite):
        return None, None
    vmin = np.nanpercentile(img[finite], p_lo)
    vmax = np.nanpercentile(img[finite], p_hi)
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def show_one(path: str):
    print("\n" + "-" * 90)
    print(f"Opening: {os.path.basename(path)}")

    with fits.open(path, memmap=False) as hdul:

        # Print primary header info (still useful)
        hdr0 = hdul[0].header
        for key in ["DATE-OBS", "EXPTIME", "FILTER", "OBJECT"]:
            if key in hdr0:
                print(f"{key}: {hdr0.get(key)}")

        img = None
        hdu_index = None

        # Search for first HDU with image data
        for i, hdu in enumerate(hdul):
            if hdu.data is None:
                continue

            data = np.array(hdu.data)

            # Accept 2D images only
            if data.ndim == 2:
                img = data
                hdu_index = i
                break

            # Sometimes data is 3D: take first plane
            if data.ndim > 2:
                img = data[0]
                hdu_index = i
                break

        if img is None:
            print("[ERROR] No image data found in any HDU.")
            return

        print(f"Displaying image from HDU[{hdu_index}]")

        vmin, vmax = robust_limits(img, 1, 99)

        plt.figure(figsize=(8, 7))
        plt.imshow(img, origin="lower",cmap ="inferno", vmin=vmin, vmax=vmax)
        plt.colorbar(label="Counts")
        plt.title(f"{os.path.basename(path)}  (HDU {hdu_index})")
        plt.tight_layout()
        plt.show()



def main():
    # Build full paths (only .fits, not .fits.Z)
    paths = [os.path.join(FITS_DIR, fname) for fname in WANTED]

    print(f"Will visualise exactly {len(paths)} decompressed .fits files (and nothing else).")

    # Check missing files
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print("\n[ERROR] These .fits files are missing (decompression may not have produced them):")
        for p in missing:
            print("  ", p)
        print("\nFix: ensure the decompression step created the .fits versions.")
        return

    # Display each file
    for p in paths:
        try:
            show_one(p)
        except Exception as e:
            print(f"[ERROR] Could not open/display {os.path.basename(p)}:\n  {e}")


if __name__ == "__main__":
    main()
