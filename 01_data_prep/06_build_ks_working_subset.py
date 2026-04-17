# Build Ks Working Subset for Reduction

from pathlib import Path
import csv
import shutil

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
FITS_FOLDER = ROOT / "01_Raw_HAWKI_Decompressed"
CSV_IN = ROOT / "hawki_header_inventory_final.csv"
WORK_ROOT = ROOT / "03_Ks_Working_Subset"

REBUILD_WORK_FOLDER = True

# ============================================================
# 2. HELPERS
# ============================================================
def norm(x):
    return str(x).strip()

def classify_from_row(row):
    dpr_type = norm(row.get("dpr_type", "")).upper()
    frame_guess = norm(row.get("frame_guess", ""))

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
    if dpr_type == "BIAS":
        return "Bias"
    if dpr_type == "SKY":
        return "Sky"
    return "Unclear"

def matches_science_ks(row):
    return (
        classify_from_row(row) == "Science"
        and norm(row.get("filter", "")) == "Ks"
    )

def matches_flat_ks(row):
    return (
        classify_from_row(row) == "Flat"
        and norm(row.get("filter", "")) == "Ks"
    )

def matches_dark_for_science(row):
    return (
        classify_from_row(row) == "Dark"
        and norm(row.get("dit", "")) == "10.0"
        and norm(row.get("ndit", "")) == "9"
    )

def safe_copy(src: Path, dest_folder: Path):
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest = dest_folder / src.name

    if dest.exists():
        return False

    shutil.copy2(src, dest)
    return True

# ============================================================
# 3. CHECK INPUTS
# ============================================================
if not FITS_FOLDER.exists():
    print(f"ERROR: FITS folder not found:\n{FITS_FOLDER}")
    raise SystemExit

if not CSV_IN.exists():
    print(f"ERROR: CSV file not found:\n{CSV_IN}")
    raise SystemExit

if REBUILD_WORK_FOLDER and WORK_ROOT.exists():
    print(f"Removing old work folder:\n{WORK_ROOT}\n")
    shutil.rmtree(WORK_ROOT)

SCIENCE_OUT = WORK_ROOT / "Science_Ks"
DARK_OUT = WORK_ROOT / "Darks_match_10.0_NDIT9"
FLAT_OUT = WORK_ROOT / "Flats_Ks"

# ============================================================
# 4. COPY MATCHING FILES
# ============================================================
science_count = 0
dark_count = 0
flat_count = 0
missing_count = 0

print("=" * 90)
print("BUILD KS WORKING SUBSET")
print("=" * 90)
print(f"Input FITS folder: {FITS_FOLDER}")
print(f"CSV inventory:     {CSV_IN}")
print(f"Output root:       {WORK_ROOT}")
print()

with open(CSV_IN, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)

    for row in reader:
        filename = norm(row["filename"])
        src = FITS_FOLDER / filename

        if not src.exists():
            print(f"MISSING: {filename}")
            missing_count += 1
            continue

        if matches_science_ks(row):
            if safe_copy(src, SCIENCE_OUT):
                science_count += 1
                print(f"SCIENCE -> {filename}")
            continue

        if matches_dark_for_science(row):
            if safe_copy(src, DARK_OUT):
                dark_count += 1
                print(f"DARK    -> {filename}")
            continue

        if matches_flat_ks(row):
            if safe_copy(src, FLAT_OUT):
                flat_count += 1
                print(f"FLAT    -> {filename}")
            continue

# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print(f"Science_Ks copied          : {science_count}")
print(f"Darks_match_10.0_NDIT9     : {dark_count}")
print(f"Flats_Ks copied            : {flat_count}")
print(f"Missing source files       : {missing_count}")
print(f"\nWorking subset created at:\n{WORK_ROOT}")