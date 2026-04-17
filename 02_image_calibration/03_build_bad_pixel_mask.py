# Build Bad Pixel Mask from Master Dark and Dark RMS

from pathlib import Path
import numpy as np
from astropy.io import fits

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
CAL_FOLDER = ROOT / "04_Calibration_Products"

MASTER_DARK_FITS = CAL_FOLDER / "master_dark_Ks_science_match.fits"
DARK_RMS_FITS = CAL_FOLDER / "master_dark_Ks_science_match_rms.fits"

BAD_PIXEL_MASK_FITS = CAL_FOLDER / "bad_pixel_mask_from_dark.fits"
REPORT_TXT = CAL_FOLDER / "bad_pixel_mask_report.txt"

# ============================================================
# 2. SETTINGS
# ============================================================
HOT_SIGMA_THRESHOLD = 8.0
NOISY_SIGMA_THRESHOLD = 8.0

# ============================================================
# 3. HELPERS
# ============================================================
def robust_sigma(data):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return 1.4826 * mad

# ============================================================
# 4. LOAD FILES
# ============================================================
if not MASTER_DARK_FITS.exists():
    print(f"ERROR: Master dark not found:\n{MASTER_DARK_FITS}")
    raise SystemExit

if not DARK_RMS_FITS.exists():
    print(f"ERROR: Dark RMS map not found:\n{DARK_RMS_FITS}")
    raise SystemExit

master_dark, dark_header = fits.getdata(MASTER_DARK_FITS, header=True)
dark_rms, rms_header = fits.getdata(DARK_RMS_FITS, header=True)

master_dark = np.asarray(master_dark, dtype=np.float64)
dark_rms = np.asarray(dark_rms, dtype=np.float64)

if master_dark.shape != dark_rms.shape:
    print("ERROR: Master dark and RMS map shapes do not match.")
    raise SystemExit

print("=" * 90)
print("BUILD BAD PIXEL MASK")
print("=" * 90)
print(f"Input master dark : {MASTER_DARK_FITS}")
print(f"Input dark RMS    : {DARK_RMS_FITS}")
print(f"Image shape       : {master_dark.shape}")
print()

# ============================================================
# 5. ROBUST THRESHOLDS
# ============================================================
dark_med = np.median(master_dark)
dark_sig = robust_sigma(master_dark)

rms_med = np.median(dark_rms)
rms_sig = robust_sigma(dark_rms)

hot_threshold = dark_med + HOT_SIGMA_THRESHOLD * dark_sig
noisy_threshold = rms_med + NOISY_SIGMA_THRESHOLD * rms_sig

# ============================================================
# 6. BUILD MASKS
# ============================================================
hot_mask = master_dark > hot_threshold
noisy_mask = dark_rms > noisy_threshold

combined_mask = hot_mask | noisy_mask

# 0 = good pixel, 1 = bad pixel
bad_pixel_mask = combined_mask.astype(np.uint8)

hot_count = int(np.sum(hot_mask))
noisy_count = int(np.sum(noisy_mask))
combined_count = int(np.sum(combined_mask))
total_pixels = int(master_dark.size)
bad_fraction = combined_count / total_pixels

# ============================================================
# 7. SAVE FITS
# ============================================================
mask_header = dark_header.copy()
mask_header["HISTORY"] = "Bad pixel mask from master dark and dark RMS"
mask_header["HISTORY"] = f"Hot threshold: dark > {hot_threshold}"
mask_header["HISTORY"] = f"Noisy threshold: rms > {noisy_threshold}"
mask_header["IMAGETYP"] = "BAD_PIXEL_MASK"
mask_header["BPMHOT"] = hot_count
mask_header["BPMNOISY"] = noisy_count
mask_header["BPMTOTAL"] = combined_count

fits.writeto(BAD_PIXEL_MASK_FITS, bad_pixel_mask, mask_header, overwrite=True)

# ============================================================
# 8. SAVE REPORT
# ============================================================
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("BAD PIXEL MASK REPORT\n")
    f.write("=" * 90 + "\n\n")

    f.write(f"Master dark file : {MASTER_DARK_FITS}\n")
    f.write(f"Dark RMS file    : {DARK_RMS_FITS}\n")
    f.write(f"Output mask file : {BAD_PIXEL_MASK_FITS}\n\n")

    f.write(f"Image shape               : {master_dark.shape}\n")
    f.write(f"Total pixels              : {total_pixels}\n\n")

    f.write(f"Master dark median        : {dark_med}\n")
    f.write(f"Master dark robust sigma  : {dark_sig}\n")
    f.write(f"Hot threshold             : {hot_threshold}\n\n")

    f.write(f"Dark RMS median           : {rms_med}\n")
    f.write(f"Dark RMS robust sigma     : {rms_sig}\n")
    f.write(f"Noisy threshold           : {noisy_threshold}\n\n")

    f.write(f"Hot pixels flagged        : {hot_count}\n")
    f.write(f"Noisy pixels flagged      : {noisy_count}\n")
    f.write(f"Total bad pixels flagged  : {combined_count}\n")
    f.write(f"Bad pixel fraction        : {bad_fraction:.8f}\n")

# ============================================================
# 9. PRINT SUMMARY
# ============================================================
print("SUMMARY")
print("-" * 90)
print(f"Master dark median       : {dark_med}")
print(f"Master dark robust sigma : {dark_sig}")
print(f"Hot threshold            : {hot_threshold}")
print()
print(f"Dark RMS median          : {rms_med}")
print(f"Dark RMS robust sigma    : {rms_sig}")
print(f"Noisy threshold          : {noisy_threshold}")
print()
print(f"Hot pixels flagged       : {hot_count}")
print(f"Noisy pixels flagged     : {noisy_count}")
print(f"Total bad pixels flagged : {combined_count}")
print(f"Bad pixel fraction       : {bad_fraction:.8f}")
print()
print(f"Bad pixel mask saved to:\n{BAD_PIXEL_MASK_FITS}")
print(f"Report saved to:\n{REPORT_TXT}")