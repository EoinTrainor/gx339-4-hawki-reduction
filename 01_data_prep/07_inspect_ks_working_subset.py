# Inspect Ks Working Subset for Calibration Consistency

from pathlib import Path
from collections import Counter, defaultdict
from astropy.io import fits

# ============================================================
# 1. PATHS
# ============================================================
ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
WORK_ROOT = ROOT / "03_Ks_Working_Subset"

GROUPS = {
    "Science_Ks": WORK_ROOT / "Science_Ks",
    "Darks_match_10.0_NDIT9": WORK_ROOT / "Darks_match_10.0_NDIT9",
    "Flats_Ks": WORK_ROOT / "Flats_Ks",
}

TXT_OUT = WORK_ROOT / "ks_working_subset_inspection.txt"

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

def sorted_nonempty(values):
    vals = [str(v).strip() for v in values if str(v).strip() != ""]
    return sorted(set(vals))

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
# 3. CHECK INPUTS
# ============================================================
for label, folder in GROUPS.items():
    if not folder.exists():
        print(f"ERROR: Missing folder for {label}:\n{folder}")
        raise SystemExit

report_lines = []
print("=" * 90)
print("KS WORKING SUBSET INSPECTION")
print("=" * 90)

# ============================================================
# 4. INSPECT EACH GROUP
# ============================================================
for label, folder in GROUPS.items():
    fits_files = sorted(folder.glob("HAWKI*.fits"))

    exptime_vals = []
    dit_vals = []
    ndit_vals = []
    filter_vals = []
    shape_vals = []
    instrument_vals = []
    detector_vals = []
    dpr_type_vals = []
    errors = []

    for fits_file in fits_files:
        try:
            with fits.open(fits_file, ignore_missing_simple=True) as hdul:
                hdr = hdul[0].header
                shape, _ = first_data_shape(hdul)

                exptime_vals.append(safe_header_get(hdr, ["EXPTIME", "TEXPTIME"]))
                dit_vals.append(safe_header_get(hdr, ["HIERARCH ESO DET DIT", "DIT"]))
                ndit_vals.append(safe_header_get(hdr, ["HIERARCH ESO DET NDIT", "NDIT"]))
                filter_vals.append(safe_header_get(hdr, [
                    "FILTER",
                    "FILTER1",
                    "HIERARCH ESO INS FILT1 NAME",
                    "HIERARCH ESO INS FILT NAME"
                ]))
                shape_vals.append(shape)
                instrument_vals.append(safe_header_get(hdr, ["INSTRUME"]))
                detector_vals.append(safe_header_get(hdr, ["DETECTOR", "HIERARCH ESO DET NAME"]))
                dpr_type_vals.append(safe_header_get(hdr, ["HIERARCH ESO DPR TYPE"]))
        except Exception as e:
            errors.append(f"{fits_file.name} -> {e}")

    exptime_unique = sorted_nonempty(exptime_vals)
    dit_unique = sorted_nonempty(dit_vals)
    ndit_unique = sorted_nonempty(ndit_vals)
    filter_unique = sorted_nonempty(filter_vals)
    shape_unique = sorted_nonempty(shape_vals)
    instrument_unique = sorted_nonempty(instrument_vals)
    detector_unique = sorted_nonempty(detector_vals)
    dpr_type_unique = sorted_nonempty(dpr_type_vals)

    print(f"\n{label}")
    print("-" * 90)
    print(f"Folder         : {folder}")
    print(f"File count     : {len(fits_files)}")
    print(f"EXPTIME unique : {exptime_unique}")
    print(f"DIT unique     : {dit_unique}")
    print(f"NDIT unique    : {ndit_unique}")
    print(f"Filter unique  : {filter_unique}")
    print(f"Shape unique   : {shape_unique}")
    print(f"Instrument     : {instrument_unique}")
    print(f"Detector       : {detector_unique}")
    print(f"DPR TYPE       : {dpr_type_unique}")

    if fits_files:
        print("Sample files   :")
        for f in fits_files[:5]:
            print(f"  {f.name}")

    if errors:
        print(f"Errors         : {len(errors)}")
        for err in errors[:5]:
            print(f"  {err}")
    else:
        print("Errors         : 0")

    report_lines.append(f"{label}\n")
    report_lines.append("-" * 90 + "\n")
    report_lines.append(f"Folder         : {folder}\n")
    report_lines.append(f"File count     : {len(fits_files)}\n")
    report_lines.append(f"EXPTIME unique : {exptime_unique}\n")
    report_lines.append(f"DIT unique     : {dit_unique}\n")
    report_lines.append(f"NDIT unique    : {ndit_unique}\n")
    report_lines.append(f"Filter unique  : {filter_unique}\n")
    report_lines.append(f"Shape unique   : {shape_unique}\n")
    report_lines.append(f"Instrument     : {instrument_unique}\n")
    report_lines.append(f"Detector       : {detector_unique}\n")
    report_lines.append(f"DPR TYPE       : {dpr_type_unique}\n")
    report_lines.append("Sample files   :\n")
    for f in fits_files[:5]:
        report_lines.append(f"  {f.name}\n")
    if errors:
        report_lines.append(f"Errors         : {len(errors)}\n")
        for err in errors[:10]:
            report_lines.append(f"  {err}\n")
    else:
        report_lines.append("Errors         : 0\n")
    report_lines.append("\n")

# ============================================================
# 5. SAVE REPORT
# ============================================================
with open(TXT_OUT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("KS WORKING SUBSET INSPECTION\n")
    f.write("=" * 90 + "\n\n")
    f.writelines(report_lines)

print("\n" + "=" * 90)
print("DONE")
print("=" * 90)
print(f"Report saved to:\n{TXT_OUT}")