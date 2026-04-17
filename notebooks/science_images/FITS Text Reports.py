# FITS Text Reports

import os
import numpy as np
from astropy.io import fits

# ---------------- USER INPUTS ----------------
FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images/Science Images FITS (Download)"      # <-- folder with .fits files
OUTPUT_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images/Science Images Text"    # <-- folder for .txt outputs
# --------------------------------------------


def _safe_str(x, max_len=5000):
    """Convert to string safely and cap length so the txt stays readable."""
    try:
        s = str(x)
    except Exception:
        s = repr(x)
    if len(s) > max_len:
        s = s[:max_len] + f"\n... [truncated at {max_len} chars] ..."
    return s


def _data_stats(arr):
    """Return robust stats for numeric arrays."""
    finite = np.isfinite(arr)
    if not np.any(finite):
        return {
            "finite_pixels": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "std": None,
            "p01": None,
            "p99": None,
        }
    vals = arr[finite]
    return {
        "finite_pixels": int(np.sum(finite)),
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "mean": float(np.nanmean(vals)),
        "median": float(np.nanmedian(vals)),
        "std": float(np.nanstd(vals)),
        "p01": float(np.nanpercentile(vals, 1)),
        "p99": float(np.nanpercentile(vals, 99)),
    }


def dump_one_fits(fits_path: str, out_txt: str):
    """Dump one FITS file to one text file."""
    with fits.open(fits_path, memmap=False) as hdul, open(out_txt, "w", encoding="utf-8") as f:
        f.write("FITS FULL DUMP\n")
        f.write("=" * 80 + "\n")
        f.write(f"File: {fits_path}\n")
        f.write(f"Number of HDUs: {len(hdul)}\n\n")

        # ---- HDU summary (hdul.info-like) ----
        f.write("HDU SUMMARY\n")
        f.write("-" * 80 + "\n")
        for i, hdu in enumerate(hdul):
            hdu_type = type(hdu).__name__
            extname = hdu.header.get("EXTNAME", "")
            name = extname if extname else "(no EXTNAME)"
            shape = "(no data)"
            try:
                if hdu.data is not None:
                    shape = str(np.shape(hdu.data))
            except Exception:
                shape = "(data unreadable)"
            f.write(f"HDU[{i:02d}] {hdu_type:<20} {name:<25} shape={shape}\n")
        f.write("\n")

        # ---- Per-HDU details ----
        for i, hdu in enumerate(hdul):
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"HDU[{i}] DETAILS\n")
            f.write("=" * 80 + "\n")

            hdu_type = type(hdu).__name__
            extname = hdu.header.get("EXTNAME", "")
            f.write(f"Type: {hdu_type}\n")
            f.write(f"EXTNAME: {extname if extname else '(none)'}\n\n")

            # Header
            f.write("FULL HEADER\n")
            f.write("-" * 80 + "\n")
            try:
                f.write(hdu.header.tostring(sep="\n", endcard=True, padding=True))
                f.write("\n")
            except Exception as e:
                f.write(f"[ERROR] Header read failed: {e}\n")

            # Data summary
            f.write("\nDATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            if hdu.data is None:
                f.write("No data in this HDU.\n")
            else:
                try:
                    data = np.array(hdu.data)
                    f.write(f"dtype: {data.dtype}\n")
                    f.write(f"shape: {data.shape}\n")
                    f.write(f"ndim:  {data.ndim}\n")

                    if np.issubdtype(data.dtype, np.number):
                        stats = _data_stats(data.astype(float, copy=False))
                        for k, v in stats.items():
                            f.write(f"{k}: {v}\n")
                    else:
                        f.write("Non-numeric data: statistics skipped.\n")

                except Exception as e:
                    f.write(f"[ERROR] Data summary failed: {e}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF DUMP\n")


def main():
    if not os.path.isdir(FITS_DIR):
        raise NotADirectoryError(f"FITS_DIR not found:\n{FITS_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fits_files = sorted(
        f for f in os.listdir(FITS_DIR)
        if f.lower().endswith(".fits")
    )

    if not fits_files:
        print("No .fits files found.")
        return

    print(f"Found {len(fits_files)} FITS files.")
    print(f"Writing text dumps to:\n{OUTPUT_DIR}\n")

    for fname in fits_files:
        fits_path = os.path.join(FITS_DIR, fname)
        txt_name = os.path.splitext(fname)[0] + ".txt"
        out_txt = os.path.join(OUTPUT_DIR, txt_name)

        try:
            print(f"Processing: {fname}")
            dump_one_fits(fits_path, out_txt)
        except Exception as e:
            print(f"[ERROR] Failed on {fname}: {e}")

    print("\nAll FITS files processed.")


if __name__ == "__main__":
    main()
