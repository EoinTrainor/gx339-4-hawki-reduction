# HAWKI Calibration Compatibility Summary

from pathlib import Path
import csv
from collections import defaultdict, Counter

ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4")
CSV_IN = ROOT / "hawki_header_inventory_final.csv"
OUT_TXT = ROOT / "hawki_calibration_summary.txt"

if not CSV_IN.exists():
    print(f"ERROR: CSV inventory not found:\n{CSV_IN}")
    raise SystemExit

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
    if dpr_type == "BIAS":
        return "Bias"
    if dpr_type == "SKY":
        return "Sky"
    return "Unclear"

summary = defaultdict(lambda: {
    "count": 0,
    "exptime": set(),
    "dit": set(),
    "ndit": set(),
    "files": []
})

with open(CSV_IN, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        frame_class = classify_from_row(row)
        filt = str(row.get("filter", "")).strip() or "Unknown"

        key = (frame_class, filt)

        summary[key]["count"] += 1
        summary[key]["exptime"].add(str(row.get("exptime", "")).strip())
        summary[key]["dit"].add(str(row.get("dit", "")).strip())
        summary[key]["ndit"].add(str(row.get("ndit", "")).strip())
        summary[key]["files"].append(row["filename"])

print("=" * 90)
print("HAWKI CALIBRATION COMPATIBILITY SUMMARY")
print("=" * 90)

with open(OUT_TXT, "w", encoding="utf-8") as f:
    f.write("=" * 90 + "\n")
    f.write("HAWKI CALIBRATION COMPATIBILITY SUMMARY\n")
    f.write("=" * 90 + "\n\n")

    for key in sorted(summary.keys()):
        frame_class, filt = key
        block = summary[key]

        exptime_vals = sorted(v for v in block["exptime"] if v)
        dit_vals = sorted(v for v in block["dit"] if v)
        ndit_vals = sorted(v for v in block["ndit"] if v)

        print(f"\n{frame_class} | Filter = {filt}")
        print(f"  Count   : {block['count']}")
        print(f"  EXPTIME : {exptime_vals}")
        print(f"  DIT     : {dit_vals}")
        print(f"  NDIT    : {ndit_vals}")

        f.write(f"{frame_class} | Filter = {filt}\n")
        f.write(f"  Count   : {block['count']}\n")
        f.write(f"  EXPTIME : {exptime_vals}\n")
        f.write(f"  DIT     : {dit_vals}\n")
        f.write(f"  NDIT    : {ndit_vals}\n\n")

print(f"\nSummary saved to:\n{OUT_TXT}")