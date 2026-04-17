# Build Master Flat from Clean Ks Flats

from pathlib import Path
import numpy as np
from astropy.io import fits

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
FLAT_FOLDER = ROOT / "04_Calibration_Products" / "Ks_Flat_Groups" / "DIT_1p676206"

OUT_FOLDER = ROOT / "04_Calibration_Products"
MASTER_FLAT_FITS = OUT_FOLDER / "master_flat_Ks_DIT1p676206.fits"
FLAT_RMS_FITS = OUT_FOLDER / "master_flat_Ks_DIT1p676206_rms.fits"
REPORT_TXT = OUT_FOLDER / "master_flat_Ks_DIT1p676206_report.txt"

# ============================================================
# 2. CHECK INPUT
# ============================================================
if not FLAT_FOLDER.exists():
    print(f"ERROR: Flat folder not found:\n{FLAT_FOLDER}")
    raise SystemExit

flat_files = sorted(FLAT_FOLDER.glob("HAWKI*.fits"))

if not flat_files:
    print(f"ERROR: No flat FITS files found in:\n{FLAT_FOLDER}")
    raise SystemExit

OUT_FOLDER.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("BUILD MASTER FLAT")
print("=" * 90)
print(f"Flat files found: {len(flat_files)}")
print(f"Input folder:     {FLAT_FOLDER}")
print()

# ============================================================
# 3. LOAD AND NORMALISE FLATS
# ============================================================
stack_list = []
header_for_output = None
used_files = []
skipped_files = []

for i, file in enumerate(flat_files, start=1):
    print(f"[{i}/{len(flat_files)}] Loading: {file.name}")

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

            finite = data[np.isfinite(data)]
            if finite.size == 0:
                raise ValueError("No finite pixels found")

            med = float(np.median(finite))
            if med <= 0:
                raise ValueError(f"Non-positive median {med}")

            norm_flat = data / med

            stack_list.append(norm_flat)
            used_files.append(file.name)

            if header_for_output is None:
                header_for_output = hdr.copy()

    except Exception as e:
        skipped_files.append((file.name, str(e)))
        print(f"    SKIPPED -> {e}")

if not stack_list:
    print("ERROR: No usable flat frames loaded.")
    raise SystemExit

stack = np.stack(stack_list, axis=0)

print()
print(f"Usable flat frames: {stack.shape[0]}")
print(f"Stack shape:        {stack.shape}")

# ============================================================
# 4. BUILD MASTER FLAT
# ============================================================
master_flat = np.median(stack, axis=0)
flat_rms = np.std(stack, axis=0)

# Renormalise final master flat to median = 1
master_flat_median_pre = float(np.median(master_flat))
if master_flat_median_pre <= 0:
    print("ERROR: Master flat median is non-positive.")
    raise SystemExit

master_flat = master_flat / master_flat_median_pre

master_flat_median = float(np.median(master_flat))
master_flat_mean = float(np.mean(master_flat))
master_flat_std = float(np.std(master_flat))
flat_rms_median = float(np.median(flat_rms))
flat_rms_mean = float(np.mean(flat_rms))

# ============================================================
# 5. PREPARE HEADERS
# ============================================================
if header_for_output is None:
    header_for_output = fits.Header()

header_for_output["HISTORY"] = "Master flat built from clean Ks flats"
header_for_output["HISTORY"] = f"Number of flats used: {stack.shape[0]}"
header_for_output["HISTORY"] = "Combination method: median of per-frame median-normalised flats"
header_for_output["IMAGETYP"] = "MASTER_FLAT"

rms_header = header_for_output.copy()
rms_header["HISTORY"] = "Per-pixel RMS map from normalised flat stack"
rms_header["IMAGETYP"] = "FLAT_RMS"

# ============================================================
# 6. SAVE FITS
# ============================================================
fits.writeto(MASTER_FLAT_FITS, master_flat.astype(np.float32), header_for_output, overwrite=True)
fits.writeto(FLAT_RMS_FITS, flat_rms.astype(np.float32), rms_header, overwrite=True)

# ============================================================
# 7. SAVE REPORT
# ============================================================
with open(REPORT_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("MASTER FLAT REPORT\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Input folder: {FLAT_FOLDER}\n")
    f.write(f"Total flat files found: {len(flat_files)}\n")
    f.write(f"Flat files used: {len(used_files)}\n")
    f.write(f"Flat files skipped: {len(skipped_files)}\n\n")

    f.write(f"Master flat output: {MASTER_FLAT_FITS}\n")
    f.write(f"Flat RMS output:    {FLAT_RMS_FITS}\n\n")

    f.write(f"Master flat shape: {master_flat.shape}\n")
    f.write(f"Master flat median: {master_flat_median}\n")
    f.write(f"Master flat mean:   {master_flat_mean}\n")
    f.write(f"Master flat std:    {master_flat_std}\n\n")

    f.write(f"Flat RMS median: {flat_rms_median}\n")
    f.write(f"Flat RMS mean:   {flat_rms_mean}\n\n")

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
print(f"Total flat files found : {len(flat_files)}")
print(f"Flat files used        : {len(used_files)}")
print(f"Flat files skipped     : {len(skipped_files)}")
print(f"Master flat shape      : {master_flat.shape}")
print(f"Master flat median     : {master_flat_median}")
print(f"Master flat mean       : {master_flat_mean}")
print(f"Master flat std        : {master_flat_std}")
print(f"Flat RMS median        : {flat_rms_median}")
print(f"Flat RMS mean          : {flat_rms_mean}")

print(f"\nMaster flat saved to:\n{MASTER_FLAT_FITS}")
print(f"\nFlat RMS map saved to:\n{FLAT_RMS_FITS}")
print(f"\nReport saved to:\n{REPORT_TXT}")