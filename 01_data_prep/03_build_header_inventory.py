# HAWKI FITS Header Inventory to CSV

from pathlib import Path
import csv
from collections import Counter
from astropy.io import fits

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
FITS_FOLDER = ROOT / "01_Raw_HAWKI_Decompressed"
CSV_OUT = ROOT / "hawki_header_inventory_final.csv"
TXT_OUT = ROOT / "hawki_header_inventory_final_report.txt"

# ============================================================
# 2. HELPERS
# ============================================================
def safe_header_get(header, keys):
    for key in keys:
        try:
            if key in header:
                return header[key]
        except Exception:
            pass
    return ""

def format_size(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

def classify_frame(dpr_cat, dpr_type, dpr_tech, obj):
    combo = f"{dpr_cat} {dpr_type} {dpr_tech} {obj}".lower()

    if "science" in combo or "object" in combo:
        return "Science"
    if "flat" in combo:
        return "Flat"
    if "dark" in combo:
        return "Dark"
    if "bias" in combo:
        return "Bias"
    if "sky" in combo:
        return "Sky"
    if "standard" in combo:
        return "Standard"
    return "Unclear"

def best_header_from_hdul(hdul):
    hdr0 = hdul[0].header
    if len(hdr0) > 0:
        return hdr0, 0

    for idx, hdu in enumerate(hdul):
        try:
            if len(hdu.header) > 0:
                return hdu.header, idx
        except Exception:
            pass

    return hdul[0].header, 0

def first_data_shape(hdul):
    for idx, hdu in enumerate(hdul):
        data = getattr(hdu, "data", None)
        if data is not None:
            try:
                return str(data.shape), idx
            except Exception:
                pass
    return "", ""

# ============================================================
# 3. CHECK INPUT
# ============================================================
if not FITS_FOLDER.exists():
    print(f"ERROR: FITS folder not found:\n{FITS_FOLDER}")
    raise SystemExit

fits_files = sorted(FITS_FOLDER.glob("HAWKI*.fits"))

print("=" * 90)
print("HAWKI HEADER INVENTORY")
print("=" * 90)
print(f"FITS files found: {len(fits_files)}")
print(f"Input folder:     {FITS_FOLDER}")
print()

if not fits_files:
    print("ERROR: No HAWKI*.fits files found.")
    raise SystemExit

rows = []
frame_counts = Counter()
filter_counts = Counter()
dpr_cat_counts = Counter()
dpr_type_counts = Counter()
error_count = 0
example_errors = []

# ============================================================
# 4. READ HEADERS
# ============================================================
for i, fits_file in enumerate(fits_files, start=1):
    print(f"Reading header [{i}/{len(fits_files)}]: {fits_file.name}")

    row = {
        "filename": fits_file.name,
        "size_bytes": fits_file.stat().st_size,
        "size_readable": format_size(fits_file.stat().st_size),
        "readable": False,
        "header_hdu_index": "",
        "data_hdu_index": "",
        "n_hdus": "",
        "shape": "",
        "object": "",
        "date_obs": "",
        "mjd_obs": "",
        "exptime": "",
        "filter": "",
        "instrument": "",
        "telescope": "",
        "det_name": "",
        "ra": "",
        "dec": "",
        "obs_id": "",
        "dp_id": "",
        "prog_id": "",
        "dpr_cat": "",
        "dpr_type": "",
        "dpr_tech": "",
        "gain": "",
        "read_noise": "",
        "ndit": "",
        "dit": "",
        "airmass": "",
        "frame_guess": "",
        "error": "",
    }

    try:
        with fits.open(fits_file, ignore_missing_simple=True) as hdul:
            row["readable"] = True
            row["n_hdus"] = len(hdul)

            shape, data_hdu_index = first_data_shape(hdul)
            row["shape"] = shape
            row["data_hdu_index"] = data_hdu_index

            hdr, header_hdu_index = best_header_from_hdul(hdul)
            row["header_hdu_index"] = header_hdu_index

            row["object"] = safe_header_get(hdr, ["OBJECT", "HIERARCH ESO OBS TARG NAME"])
            row["date_obs"] = safe_header_get(hdr, ["DATE-OBS"])
            row["mjd_obs"] = safe_header_get(hdr, ["MJD-OBS"])
            row["exptime"] = safe_header_get(hdr, ["EXPTIME", "TEXPTIME"])
            row["filter"] = safe_header_get(hdr, [
                "FILTER",
                "FILTER1",
                "HIERARCH ESO INS FILT1 NAME",
                "HIERARCH ESO INS FILT NAME"
            ])
            row["instrument"] = safe_header_get(hdr, ["INSTRUME"])
            row["telescope"] = safe_header_get(hdr, ["TELESCOP"])
            row["det_name"] = safe_header_get(hdr, ["DETECTOR", "HIERARCH ESO DET NAME"])
            row["ra"] = safe_header_get(hdr, ["RA", "CRVAL1"])
            row["dec"] = safe_header_get(hdr, ["DEC", "CRVAL2"])
            row["obs_id"] = safe_header_get(hdr, ["OBID1", "HIERARCH ESO OBS ID"])
            row["dp_id"] = safe_header_get(hdr, ["DP.ID", "HIERARCH ESO DP ID"])
            row["prog_id"] = safe_header_get(hdr, ["PROG_ID", "HIERARCH ESO OBS PROG ID"])
            row["dpr_cat"] = safe_header_get(hdr, ["HIERARCH ESO DPR CAT"])
            row["dpr_type"] = safe_header_get(hdr, ["HIERARCH ESO DPR TYPE"])
            row["dpr_tech"] = safe_header_get(hdr, ["HIERARCH ESO DPR TECH"])
            row["gain"] = safe_header_get(hdr, ["GAIN", "HIERARCH ESO DET OUT1 GAIN"])
            row["read_noise"] = safe_header_get(hdr, ["RDNOISE", "RON", "HIERARCH ESO DET OUT1 RON"])
            row["ndit"] = safe_header_get(hdr, ["HIERARCH ESO DET NDIT", "NDIT"])
            row["dit"] = safe_header_get(hdr, ["HIERARCH ESO DET DIT", "DIT"])
            row["airmass"] = safe_header_get(hdr, ["AIRMASS", "HIERARCH ESO TEL AIRM START"])

            row["frame_guess"] = classify_frame(
                str(row["dpr_cat"]),
                str(row["dpr_type"]),
                str(row["dpr_tech"]),
                str(row["object"])
            )

            frame_counts[row["frame_guess"]] += 1
            filter_counts[str(row["filter"])] += 1
            dpr_cat_counts[str(row["dpr_cat"])] += 1
            dpr_type_counts[str(row["dpr_type"])] += 1

    except Exception as e:
        row["error"] = str(e)
        error_count += 1
        if len(example_errors) < 15:
            example_errors.append((fits_file.name, str(e)))

    rows.append(row)

# ============================================================
# 5. SAVE CSV
# ============================================================
fieldnames = list(rows[0].keys())

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

# ============================================================
# 6. SAVE TEXT REPORT
# ============================================================
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("HAWKI HEADER INVENTORY REPORT\n")
    f.write("=" * 90 + "\n\n")
    f.write(f"FITS files found: {len(fits_files)}\n")
    f.write(f"Errors: {error_count}\n\n")

    f.write("FRAME TYPE COUNTS\n")
    f.write("-" * 90 + "\n")
    for key, value in frame_counts.most_common():
        f.write(f"{key:20} : {value}\n")

    f.write("\nDPR CAT COUNTS\n")
    f.write("-" * 90 + "\n")
    for key, value in dpr_cat_counts.most_common():
        f.write(f"{key:20} : {value}\n")

    f.write("\nDPR TYPE COUNTS\n")
    f.write("-" * 90 + "\n")
    for key, value in dpr_type_counts.most_common():
        f.write(f"{key:20} : {value}\n")

    f.write("\nFILTER COUNTS\n")
    f.write("-" * 90 + "\n")
    for key, value in filter_counts.most_common():
        f.write(f"{key:20} : {value}\n")

    f.write("\nEXAMPLE ERRORS\n")
    f.write("-" * 90 + "\n")
    for name, err in example_errors:
        f.write(f"{name} -> {err}\n")

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"\nTotal FITS files: {len(fits_files)}")
print(f"Errors: {error_count}")

print("\nFrame type counts:")
for key, value in frame_counts.most_common():
    print(f"  {key:20} : {value}")

print("\nDPR CAT counts:")
for key, value in dpr_cat_counts.most_common():
    print(f"  {key:20} : {value}")

print("\nDPR TYPE counts:")
for key, value in dpr_type_counts.most_common():
    print(f"  {key:20} : {value}")

print("\nFilter counts:")
for key, value in filter_counts.most_common():
    print(f"  {key:20} : {value}")

if example_errors:
    print("\nExample errors:")
    for name, err in example_errors[:10]:
        print(f"  {name} -> {err}")

print(f"\nCSV saved to:\n{CSV_OUT}")
print(f"\nText report saved to:\n{TXT_OUT}")