# Sort HAWKI FITS files into subcategories by type and filter

from pathlib import Path
import csv
import shutil

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
FITS_FOLDER = ROOT / "01_Raw_HAWKI_Decompressed"
CSV_IN = ROOT / "hawki_header_inventory_final.csv"
SORTED_ROOT = ROOT / "02_Sorted_By_Type_And_Filter"

# If True, deletes the old sorted folder and rebuilds it cleanly
REBUILD_SORTED_FOLDER = True

# ============================================================
# 2. HELPERS
# ============================================================
def clean_name(value):
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return "Unknown"
    return text.replace("/", "_").replace("\\", "_").replace(" ", "_")

def classify_from_row(row):
    dpr_type = str(row.get("dpr_type", "")).strip().upper()
    frame_guess = str(row.get("frame_guess", "")).strip()

    # Force STD to Standard
    if dpr_type == "STD":
        return "Standard"

    # Trust existing good labels
    if frame_guess in ["Science", "Flat", "Dark", "Bias", "Sky", "Standard"]:
        return frame_guess

    # Fallback from DPR TYPE
    if dpr_type == "OBJECT":
        return "Science"
    if dpr_type == "FLAT":
        return "Flat"
    if dpr_type == "DARK":
        return "Dark"
    if dpr_type == "BIAS":
        return "Bias"
    if dpr_type == "SKY":
        return "Sky"

    return "Unclear"

def destination_folder(frame_class, filt):
    filt = clean_name(filt)

    if frame_class == "Science":
        return SORTED_ROOT / "Science" / filt

    if frame_class == "Flat":
        return SORTED_ROOT / "Flats" / filt

    if frame_class == "Standard":
        return SORTED_ROOT / "Standards" / filt

    if frame_class == "Dark":
        return SORTED_ROOT / "Darks"

    if frame_class == "Bias":
        return SORTED_ROOT / "Bias"

    if frame_class == "Sky":
        return SORTED_ROOT / "Sky" / filt

    return SORTED_ROOT / "Unclear"

# ============================================================
# 3. CHECK INPUTS
# ============================================================
if not FITS_FOLDER.exists():
    print(f"ERROR: FITS folder not found:\n{FITS_FOLDER}")
    raise SystemExit

if not CSV_IN.exists():
    print(f"ERROR: CSV inventory not found:\n{CSV_IN}")
    raise SystemExit

# ============================================================
# 4. OPTIONAL CLEAN REBUILD
# ============================================================
if REBUILD_SORTED_FOLDER and SORTED_ROOT.exists():
    print(f"Removing old sorted folder:\n{SORTED_ROOT}\n")
    shutil.rmtree(SORTED_ROOT)

SORTED_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 5. SORT FILES
# ============================================================
copied = 0
missing = 0
skipped_existing = 0

print("=" * 90)
print("SORT HAWKI FITS FILES INTO SUBCATEGORIES")
print("=" * 90)
print(f"FITS folder:   {FITS_FOLDER}")
print(f"CSV inventory: {CSV_IN}")
print(f"Output folder: {SORTED_ROOT}")
print()

with open(CSV_IN, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename = row["filename"]
        filt = row.get("filter", "")
        frame_class = classify_from_row(row)

        src = FITS_FOLDER / filename

        if not src.exists():
            print(f"MISSING: {filename}")
            missing += 1
            continue

        dest_folder = destination_folder(frame_class, filt)
        dest_folder.mkdir(parents=True, exist_ok=True)
        dest = dest_folder / filename

        if dest.exists():
            print(f"SKIP already exists: {dest}")
            skipped_existing += 1
            continue

        shutil.copy2(src, dest)
        copied += 1
        print(f"COPIED: {filename} -> {dest_folder}")

# ============================================================
# 6. SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print(f"Files copied:      {copied}")
print(f"Missing files:     {missing}")
print(f"Skipped existing:  {skipped_existing}")
print(f"\nSorted folder created at:\n{SORTED_ROOT}")