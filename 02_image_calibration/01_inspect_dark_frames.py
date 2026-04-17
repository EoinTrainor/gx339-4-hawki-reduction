# Inspect Matched Dark Frames

from pathlib import Path
import csv
import numpy as np
from astropy.io import fits

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
DARK_FOLDER = ROOT / "03_Ks_Working_Subset" / "Darks_match_10.0_NDIT9"
CSV_OUT = ROOT / "03_Ks_Working_Subset" / "dark_frame_statistics.csv"
TXT_OUT = ROOT / "03_Ks_Working_Subset" / "dark_frame_statistics_report.txt"

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

print("=" * 90)
print("INSPECT MATCHED DARK FRAMES")
print("=" * 90)
print(f"Dark files found: {len(dark_files)}")
print(f"Folder: {DARK_FOLDER}")
print()

# ============================================================
# 3. HELPERS
# ============================================================
def safe_header_get(header, keys):
    for key in keys:
        try:
            if key in header:
                return header[key]
        except Exception:
            pass
    return ""

def robust_sigma(data):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return 1.4826 * mad

# ============================================================
# 4. INSPECT FRAMES
# ============================================================
rows = []
errors = []

for i, file in enumerate(dark_files, start=1):
    print(f"[{i}/{len(dark_files)}] Reading: {file.name}")

    row = {
        "filename": file.name,
        "shape": "",
        "dtype": "",
        "date_obs": "",
        "exptime": "",
        "dit": "",
        "ndit": "",
        "filter": "",
        "median": "",
        "mean": "",
        "std": "",
        "robust_sigma": "",
        "min": "",
        "max": "",
        "nan_count": "",
        "error": "",
    }

    try:
        with fits.open(file, ignore_missing_simple=True) as hdul:
            data = None
            header = None

            for hdu in hdul:
                if getattr(hdu, "data", None) is not None:
                    data = hdu.data
                    header = hdu.header
                    break

            if data is None:
                raise ValueError("No image data found")

            data = np.asarray(data, dtype=np.float64)

            row["shape"] = str(data.shape)
            row["dtype"] = str(data.dtype)
            row["date_obs"] = safe_header_get(header, ["DATE-OBS"])
            row["exptime"] = safe_header_get(header, ["EXPTIME", "TEXPTIME"])
            row["dit"] = safe_header_get(header, ["HIERARCH ESO DET DIT", "DIT"])
            row["ndit"] = safe_header_get(header, ["HIERARCH ESO DET NDIT", "NDIT"])
            row["filter"] = safe_header_get(header, [
                "FILTER",
                "FILTER1",
                "HIERARCH ESO INS FILT1 NAME",
                "HIERARCH ESO INS FILT NAME"
            ])

            finite_mask = np.isfinite(data)
            finite_data = data[finite_mask]

            if finite_data.size == 0:
                raise ValueError("No finite pixels found")

            row["median"] = float(np.median(finite_data))
            row["mean"] = float(np.mean(finite_data))
            row["std"] = float(np.std(finite_data))
            row["robust_sigma"] = float(robust_sigma(finite_data))
            row["min"] = float(np.min(finite_data))
            row["max"] = float(np.max(finite_data))
            row["nan_count"] = int(np.size(data) - finite_data.size)

    except Exception as e:
        row["error"] = str(e)
        errors.append((file.name, str(e)))

    rows.append(row)

# ============================================================
# 5. OUTLIER CHECKS
# ============================================================
good_rows = [r for r in rows if r["error"] == ""]

medians = np.array([r["median"] for r in good_rows], dtype=float) if good_rows else np.array([])
sigmas = np.array([r["robust_sigma"] for r in good_rows], dtype=float) if good_rows else np.array([])

median_of_medians = float(np.median(medians)) if medians.size else np.nan
sigma_of_medians = float(robust_sigma(medians)) if medians.size else np.nan

median_of_sigmas = float(np.median(sigmas)) if sigmas.size else np.nan
sigma_of_sigmas = float(robust_sigma(sigmas)) if sigmas.size else np.nan

for r in good_rows:
    flags = []

    if sigma_of_medians > 0:
        if abs(r["median"] - median_of_medians) > 3 * sigma_of_medians:
            flags.append("median_outlier")

    if sigma_of_sigmas > 0:
        if abs(r["robust_sigma"] - median_of_sigmas) > 3 * sigma_of_sigmas:
            flags.append("noise_outlier")

    r["flags"] = ",".join(flags)

for r in rows:
    if "flags" not in r:
        r["flags"] = ""

# ============================================================
# 6. SAVE CSV
# ============================================================
fieldnames = list(rows[0].keys())

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ============================================================
# 7. SAVE REPORT
# ============================================================
unique_shapes = sorted(set(r["shape"] for r in good_rows))
unique_exptime = sorted(set(str(r["exptime"]) for r in good_rows))
unique_dit = sorted(set(str(r["dit"]) for r in good_rows))
unique_ndit = sorted(set(str(r["ndit"]) for r in good_rows))
flagged = [r for r in good_rows if r["flags"]]

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("DARK FRAME STATISTICS REPORT\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Total dark files: {len(dark_files)}\n")
    f.write(f"Readable dark files: {len(good_rows)}\n")
    f.write(f"Errored dark files: {len(errors)}\n\n")

    f.write(f"Unique shapes   : {unique_shapes}\n")
    f.write(f"Unique EXPTIME  : {unique_exptime}\n")
    f.write(f"Unique DIT      : {unique_dit}\n")
    f.write(f"Unique NDIT     : {unique_ndit}\n\n")

    f.write(f"Median of medians      : {median_of_medians}\n")
    f.write(f"Robust sigma of median : {sigma_of_medians}\n")
    f.write(f"Median of noise        : {median_of_sigmas}\n")
    f.write(f"Robust sigma of noise  : {sigma_of_sigmas}\n\n")

    f.write("FLAGGED FILES\n")
    f.write("-" * 90 + "\n")
    for r in flagged:
        f.write(f"{r['filename']} -> {r['flags']}\n")

    f.write("\nERRORS\n")
    f.write("-" * 90 + "\n")
    for name, err in errors:
        f.write(f"{name} -> {err}\n")

# ============================================================
# 8. PRINT SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"Total dark files     : {len(dark_files)}")
print(f"Readable dark files  : {len(good_rows)}")
print(f"Errored dark files   : {len(errors)}")
print(f"Unique shapes        : {unique_shapes}")
print(f"Unique EXPTIME       : {unique_exptime}")
print(f"Unique DIT           : {unique_dit}")
print(f"Unique NDIT          : {unique_ndit}")
print(f"Flagged dark frames  : {len(flagged)}")

if flagged:
    print("\nFirst flagged files:")
    for r in flagged[:10]:
        print(f"  {r['filename']} -> {r['flags']}")

if errors:
    print("\nErrors:")
    for name, err in errors[:10]:
        print(f"  {name} -> {err}")

print(f"\nCSV saved to:\n{CSV_OUT}")
print(f"\nReport saved to:\n{TXT_OUT}")