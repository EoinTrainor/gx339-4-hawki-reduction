# Build Master Dark for Ks Science

from pathlib import Path
import numpy as np
from astropy.io import fits

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
DARK_FOLDER = ROOT / "03_Ks_Working_Subset" / "Darks_match_10.0_NDIT9"

OUT_FOLDER = ROOT / "04_Calibration_Products"
MASTER_DARK_FITS = OUT_FOLDER / "master_dark_Ks_science_match.fits"
DARK_RMS_FITS = OUT_FOLDER / "master_dark_Ks_science_match_rms.fits"
REPORT_TXT = OUT_FOLDER / "master_dark_report.txt"

# ============================================================
# 2. CHECK INPUT
# ============================================================
if not DARK_FOLDER.exists():
    print(f"ERROR: Dark folder not found:\n{DARK_FOLDER}")
    raise SystemExit

dark_files = sorted(DARK_FOLDER.glob("HAWKI*.fits"))

if not dark_files:
    print(f"ERROR: No dark FITS files found in:\n{DARK_FOLDER}")
    raise SystemExit

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("BUILD MASTER DARK")
print("=" * 90)
print(f"Dark files found: {len(dark_files)}")
print(f"Input folder:     {DARK_FOLDER}")
print()

# ============================================================
# 3. LOAD STACK
# ============================================================
stack_list = []
header_for_output = None
used_files = []
skipped_files = []

for i, file in enumerate(dark_files, start=1):
    print(f"[{i}/{len(dark_files)}] Loading: {file.name}")

    try:
        with fits.open(file, ignore_missing_simple=True) as hdul:
            data = None
            hdr = None

            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    data = np.asarray(hdu.data, dtype=np.float64)
                    hdr = hdu.header
                    break

            if data is None:
                raise ValueError("No image data found")

            if data.shape != (2048, 2048):
                raise ValueError(f"Unexpected shape {data.shape}")

            stack_list.append(data)
            used_files.append(file.name)

            if header_for_output is None:
                header_for_output = hdr.copy()

    except Exception as e:
        skipped_files.append((file.name, str(e)))
        print(f"    SKIPPED -> {e}")

if not stack_list:
    print("ERROR: No usable dark frames loaded.")
    raise SystemExit

stack = np.stack(stack_list, axis=0)

print()
print(f"Usable dark frames: {stack.shape[0]}")
print(f"Stack shape:        {stack.shape}")

# ============================================================
# 4. BUILD MASTER DARK
# ============================================================
master_dark = np.median(stack, axis=0)
dark_rms = np.std(stack, axis=0)

# Global stats
master_dark_median = float(np.median(master_dark))
master_dark_mean = float(np.mean(master_dark))
master_dark_std = float(np.std(master_dark))
dark_rms_median = float(np.median(dark_rms))
dark_rms_mean = float(np.mean(dark_rms))

# ============================================================
# 5. PREPARE HEADERS
# ============================================================
if header_for_output is None:
    header_for_output = fits.Header()

header_for_output["HISTORY"] = "Master dark built from matched Ks science dark frames"
header_for_output["HISTORY"] = f"Number of dark frames used: {stack.shape[0]}"
header_for_output["HISTORY"] = "Combination method: median"
header_for_output["IMAGETYP"] = "MASTER_DARK"

rms_header = header_for_output.copy()
rms_header["HISTORY"] = "Per-pixel RMS map from dark stack"
rms_header["IMAGETYP"] = "DARK_RMS"

# ============================================================
# 6. SAVE FITS
# ============================================================
fits.writeto(MASTER_DARK_FITS, master_dark.astype(np.float32), header_for_output, overwrite=True)
fits.writeto(DARK_RMS_FITS, dark_rms.astype(np.float32), rms_header, overwrite=True)

# ============================================================
# 7. SAVE REPORT
# ============================================================
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("MASTER DARK REPORT\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Input folder: {DARK_FOLDER}\n")
    f.write(f"Total dark files found: {len(dark_files)}\n")
    f.write(f"Dark files used: {len(used_files)}\n")
    f.write(f"Dark files skipped: {len(skipped_files)}\n\n")

    f.write(f"Master dark output: {MASTER_DARK_FITS}\n")
    f.write(f"Dark RMS output:    {DARK_RMS_FITS}\n\n")

    f.write(f"Master dark shape: {master_dark.shape}\n")
    f.write(f"Master dark median: {master_dark_median}\n")
    f.write(f"Master dark mean:   {master_dark_mean}\n")
    f.write(f"Master dark std:    {master_dark_std}\n\n")

    f.write(f"Dark RMS median: {dark_rms_median}\n")
    f.write(f"Dark RMS mean:   {dark_rms_mean}\n\n")

    f.write("USED FILES\n")
    f.write("-" * 90 + "\n")
    for name in used_files:
        f.write(name + "\n")

    f.write("\nSKIPPED FILES\n")
    f.write("-" * 90 + "\n")
    for name, err in skipped_files:
        f.write(f"{name} -> {err}\n")

# ============================================================
# 8. PRINT SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Total dark files found : {len(dark_files)}")
print(f"Dark files used        : {len(used_files)}")
print(f"Dark files skipped     : {len(skipped_files)}")
print(f"Master dark shape      : {master_dark.shape}")
print(f"Master dark median     : {master_dark_median}")
print(f"Master dark mean       : {master_dark_mean}")
print(f"Master dark std        : {master_dark_std}")
print(f"Dark RMS median        : {dark_rms_median}")
print(f"Dark RMS mean          : {dark_rms_mean}")

print(f"\nMaster dark saved to:\n{MASTER_DARK_FITS}")
print(f"\nDark RMS map saved to:\n{DARK_RMS_FITS}")
print(f"\nReport saved to:\n{REPORT_TXT}")