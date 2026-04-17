# title: HAWK-I Science Image Viewer (4 chips, greyscale) with DATE-OBS title

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits


# ---------------- USER INPUTS ----------------
FITS_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/MEF Science Images/MEF_SI_Chronologic_DATE_OBS_FITS/2025-06-04_05-00-10.244100__ADP.2025-07-08T07-33-00.511.fits"
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


def is_image_hdu(hdu) -> bool:
    return (hdu.data is not None) and isinstance(hdu.data, np.ndarray) and (hdu.data.ndim >= 2)


def get_chip_image_hdus(hdul):
    """
    Return up to the first 4 image HDUs (prefer ones with chip-like EXTNAMEs).
    Each entry returned is: (hdu_index, data_2d, header, extname_upper)
    """
    candidates = []

    for i, hdu in enumerate(hdul):
        if not is_image_hdu(hdu):
            continue

        hdr = hdu.header
        extname = str(hdr.get("EXTNAME", "")).strip().upper()
        data = np.array(hdu.data)

        while data.ndim > 2:
            data = data[0]

        candidates.append((i, extname, data, hdr))

    if not candidates:
        return []

    patterns = ["CHIP", "DET", "CCD", "SG", "SCI"]

    named = []
    for (i, extname, data, hdr) in candidates:
        if any(p in extname for p in patterns) and extname != "":
            named.append((i, extname, data, hdr))

    if len(named) >= 4:
        named_sorted = sorted(named, key=lambda t: t[0])
        return [(i, data, hdr, extname) for (i, extname, data, hdr) in named_sorted[:4]]

    candidates_sorted = sorted(candidates, key=lambda t: t[0])
    return [(i, data, hdr, extname) for (i, extname, data, hdr) in candidates_sorted[:4]]


def header_summary(hdr):
    keys = [
        "DATE-OBS",
        "EXPTIME",
        "OBJECT",
        "FILTER",
        "HIERARCH ESO DPR CATG",
        "HIERARCH ESO DPR TYPE",
        "HIERARCH ESO PRO CATG",
        "HIERARCH ESO PRO TYPE",
    ]
    out = []
    for k in keys:
        if k in hdr:
            out.append(f"{k}: {hdr.get(k)}")
    return out


def find_date_obs_any_hdu(hdul):
    """
    Prefer DATE-OBS in primary header, otherwise look through extensions.
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
        print("\n--- Primary header summary ---")
        for line in header_summary(hdul[0].header):
            print(line)

        date_obs = find_date_obs_any_hdu(hdul)

        chip_hdus = get_chip_image_hdus(hdul)

        if len(chip_hdus) < 4:
            print(f"\n[WARN] Found only {len(chip_hdus)} image HDU(s). HAWK-I often has 4 chips.")
            print("Showing what was found anyway.")

        print("\n--- Image HDUs selected for display ---")
        for (idx, data, hdr, extname) in chip_hdus:
            print(f"HDU[{idx}] EXTNAME={extname if extname else '(none)'} shape={data.shape}")

        # Plot as 2x2 mosaic (up to 4 panels)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        # Big title (DATE-OBS) + smaller subtitle (filename)
        fig.suptitle(f"DATE-OBS: {date_obs}", fontsize=16, y=0.98)
        fig.text(0.5, 0.955, os.path.basename(FITS_FILE), ha="center", va="top", fontsize=11)

        for k in range(4):
            ax = axes[k]

            if k >= len(chip_hdus):
                ax.axis("off")
                continue

            idx, data, hdr, extname = chip_hdus[k]
            vmin, vmax = robust_limits(data, 1, 99)

            ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

            # Per-panel subtitle: HDU + EXTNAME
            panel_title = f"HDU[{idx}]"
            if extname:
                panel_title += f"  {extname}"
            ax.set_title(panel_title, fontsize=11)

            ax.set_xlabel("X (pix)")
            ax.set_ylabel("Y (pix)")

        # Leave room at the top for suptitle + filename line
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


if __name__ == "__main__":
    main()
