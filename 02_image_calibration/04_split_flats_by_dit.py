# Split and Inspect Ks Flats by DIT using the CSV inventory

from pathlib import Path
import shutil
import csv
import numpy as np
from astropy.io import fits

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
FLAT_FOLDER = ROOT / "03_Ks_Working_Subset" / "Flats_Ks"
CSV_IN = ROOT / "hawki_header_inventory_final.csv"

OUT_ROOT = ROOT / "04_Calibration_Products" / "Ks_Flat_Groups"
GROUP_A = OUT_ROOT / "DIT_1p676206"
GROUP_B = OUT_ROOT / "DIT_3p5"

CSV_OUT = OUT_ROOT / "ks_flat_group_statistics.csv"
TXT_OUT = OUT_ROOT / "ks_flat_group_statistics_report.txt"

REBUILD_OUTPUT = True

# ============================================================
# 2. HELPERS
# ============================================================
def robust_sigma(data):
    med = np.median(data)
    mad = np.median(np.abs(data - med))
    return 1.4826 * mad

def first_data_hdu(hdul):
    for hdu in hdul:
        if getattr(hdu, "data", None) is not None:
            return hdu.data, hdu.header
    return None, None

def classify_from_row(row):
    dpr_type = str(row.get("dpr_type", "")).strip().upper()
    frame_guess = str(row.get("frame_guess", "")).strip()

    if dpr_type == "STD":
        return "Standard"
    if frame_guess in ["Science", "Flat", "Dark", "Bias", "Sky", "Standard"]:
        return frame_guess
    if dpr_type == "OBJECT":
        return "Science"
    if dpr_type == "FLAT":
        return "Flat"
    if dpr_type == "DARK":
        return "Dark"
    return "Unclear"

def classify_dit(text):
    text = str(text).strip()
    if text == "1.676206":
        return "DIT_1p676206"
    if text == "3.5":
        return "DIT_3p5"
    return "OTHER"

# ============================================================
# 3. CHECK INPUTS
# ============================================================
if not FLAT_FOLDER.exists():
    print(f"ERROR: Flat folder not found:\n{FLAT_FOLDER}")
    raise SystemExit

if not CSV_IN.exists():
    print(f"ERROR: CSV inventory not found:\n{CSV_IN}")
    raise SystemExit

flat_files = sorted(FLAT_FOLDER.glob("HAWKI*.fits"))

if not flat_files:
    print(f"ERROR: No flat FITS files found in:\n{FLAT_FOLDER}")
    raise SystemExit

if REBUILD_OUTPUT and OUT_ROOT.exists():
    shutil.rmtree(OUT_ROOT)

OUT_ROOT.mkdir(parents=True, exist_ok=True)
GROUP_A.mkdir(parents=True, exist_ok=True)
GROUP_B.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SPLIT AND INSPECT KS FLATS BY DIT")
print("=" * 90)
print(f"Input flat files: {len(flat_files)}")
print(f"CSV inventory:    {CSV_IN}")
print()

# ============================================================
# 4. MAP FILENAMES TO DIT FROM CSV
# ============================================================
dit_lookup = {}

with open(CSV_IN, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        filename = str(row.get("filename", "")).strip()
        if not filename:
            continue
        if classify_from_row(row) == "Flat" and str(row.get("filter", "")).strip() == "Ks":
            dit_lookup[filename] = str(row.get("dit", "")).strip()

# ============================================================
# 5. SPLIT FILES USING CSV
# ============================================================
copied_a = 0
copied_b = 0
other = 0
rows = []

for i, file in enumerate(flat_files, start=1):
    print(f"[{i}/{len(flat_files)}] Reading: {file.name}")

    row = {
        "filename": file.name,
        "group": "",
        "dit": "",
        "shape": "",
        "median": "",
        "mean": "",
        "std": "",
        "robust_sigma": "",
        "min": "",
        "max": "",
        "nan_count": "",
        "flags": "",
        "error": "",
    }

    try:
        dit = dit_lookup.get(file.name, "")
        row["dit"] = dit
        group = classify_dit(dit)
        row["group"] = group

        with fits.open(file, ignore_missing_simple=True) as hdul:
            data, hdr = first_data_hdu(hdul)
            if data is None:
                raise ValueError("No image data found")

            data = np.asarray(data, dtype=np.float64)
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                raise ValueError("No finite pixels found")

            row["shape"] = str(data.shape)
            row["median"] = float(np.median(finite))
            row["mean"] = float(np.mean(finite))
            row["std"] = float(np.std(finite))
            row["robust_sigma"] = float(robust_sigma(finite))
            row["min"] = float(np.min(finite))
            row["max"] = float(np.max(finite))
            row["nan_count"] = int(data.size - finite.size)

        if group == "DIT_1p676206":
            shutil.copy2(file, GROUP_A / file.name)
            copied_a += 1
        elif group == "DIT_3p5":
            shutil.copy2(file, GROUP_B / file.name)
            copied_b += 1
        else:
            other += 1

    except Exception as e:
        row["error"] = str(e)

    rows.append(row)

# ============================================================
# 6. OUTLIER FLAGS
# ============================================================
for group_name in ["DIT_1p676206", "DIT_3p5"]:
    subset = [r for r in rows if r["group"] == group_name and r["error"] == ""]
    if not subset:
        continue

    medians = np.array([r["median"] for r in subset], dtype=float)
    noises = np.array([r["robust_sigma"] for r in subset], dtype=float)

    med0 = np.median(medians)
    sig_med = robust_sigma(medians)
    noise0 = np.median(noises)
    sig_noise = robust_sigma(noises)

    for r in subset:
        flags = []
        if sig_med > 0 and abs(r["median"] - med0) > 3 * sig_med:
            flags.append("median_outlier")
        if sig_noise > 0 and abs(r["robust_sigma"] - noise0) > 3 * sig_noise:
            flags.append("noise_outlier")
        r["flags"] = ",".join(flags)

def summarise_group(group_name):
    subset = [r for r in rows if r["group"] == group_name and r["error"] == ""]
    flagged = [r for r in subset if r["flags"]]
    if not subset:
        return {"count": 0, "shape_set": [], "median_of_medians": "", "median_of_noise": "", "flagged": []}
    return {
        "count": len(subset),
        "shape_set": sorted(set(r["shape"] for r in subset)),
        "median_of_medians": float(np.median([r["median"] for r in subset])),
        "median_of_noise": float(np.median([r["robust_sigma"] for r in subset])),
        "flagged": flagged,
    }

summary_a = summarise_group("DIT_1p676206")
summary_b = summarise_group("DIT_3p5")

# ============================================================
# 7. SAVE OUTPUTS
# ============================================================
fieldnames = list(rows[0].keys())
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("KS FLAT GROUP STATISTICS REPORT\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"Total flat files found: {len(flat_files)}\n")
    f.write(f"DIT 1.676206 copied: {copied_a}\n")
    f.write(f"DIT 3.5 copied: {copied_b}\n")
    f.write(f"Other/unknown DIT: {other}\n\n")

    f.write("GROUP: DIT_1p676206\n")
    f.write("-" * 90 + "\n")
    f.write(f"Count              : {summary_a['count']}\n")
    f.write(f"Shape set          : {summary_a['shape_set']}\n")
    f.write(f"Median of medians  : {summary_a['median_of_medians']}\n")
    f.write(f"Median of noise    : {summary_a['median_of_noise']}\n")
    f.write(f"Flagged frames     : {len(summary_a['flagged'])}\n")

    f.write("\nGROUP: DIT_3p5\n")
    f.write("-" * 90 + "\n")
    f.write(f"Count              : {summary_b['count']}\n")
    f.write(f"Shape set          : {summary_b['shape_set']}\n")
    f.write(f"Median of medians  : {summary_b['median_of_medians']}\n")
    f.write(f"Median of noise    : {summary_b['median_of_noise']}\n")
    f.write(f"Flagged frames     : {len(summary_b['flagged'])}\n")

# ============================================================
# 8. PRINT SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
print(f"DIT 1.676206 copied : {copied_a}")
print(f"DIT 3.5 copied      : {copied_b}")
print(f"Other/unknown DIT   : {other}")

print("\nDIT_1p676206")
print(f"  Count             : {summary_a['count']}")
print(f"  Shape set         : {summary_a['shape_set']}")
print(f"  Median of medians : {summary_a['median_of_medians']}")
print(f"  Median of noise   : {summary_a['median_of_noise']}")
print(f"  Flagged frames    : {len(summary_a['flagged'])}")

print("\nDIT_3p5")
print(f"  Count             : {summary_b['count']}")
print(f"  Shape set         : {summary_b['shape_set']}")
print(f"  Median of medians : {summary_b['median_of_medians']}")
print(f"  Median of noise   : {summary_b['median_of_noise']}")
print(f"  Flagged frames    : {len(summary_b['flagged'])}")

print(f"\nCSV saved to:\n{CSV_OUT}")
print(f"\nReport saved to:\n{TXT_OUT}")
print(f"\nFlat groups saved in:\n{OUT_ROOT}")