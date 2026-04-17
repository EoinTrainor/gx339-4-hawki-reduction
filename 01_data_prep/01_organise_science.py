# ESO Science-Only Organiser for Calibration and ZOGY Work

from pathlib import Path
import shutil

# ============================================================
# 1. SET YOUR PATH
# ============================================================
ROOT = Path(r"C:\Users\user\OneDrive - University College Cork\Desktop\archive")
OUT = ROOT / "SCIENCE_ONLY"

# safety: force Path objects
ROOT = Path(ROOT)
OUT = Path(OUT)

# ============================================================
# 2. HELPERS
# ============================================================
def safe_copy(src: Path, dest_folder: Path):
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    dest = dest_folder / src.name
    counter = 1

    while dest.exists():
        dest = dest_folder / f"{src.stem}_{counter}{src.suffix}"
        counter += 1

    shutil.copy2(src, dest)

def classify_file(p: Path):
    name = p.name
    lower = name.lower()

    # Main raw science products
    if lower.startswith("hawki.") and lower.endswith(".fits.z"):
        return "01_Raw_HAWKI_Compressed"

    # Calibration / association metadata
    if lower.endswith("_raw2raw.xml"):
        return "02_Association_raw2raw_XML"

    if lower.endswith("_raw2master.xml"):
        return "03_Association_raw2master_XML"

    if lower.endswith(".nl.txt"):
        return "04_Metadata_NL_TXT"

    # Optional archive products
    if lower.startswith("adp.") and lower.endswith(".fits"):
        return "05_ADP_Processed_FITS"

    if lower.startswith("adp.") and lower.endswith(".png"):
        return "06_ADP_Preview_PNG"

    # everything else gets ignored
    return None

# ============================================================
# 3. SCAN AND COPY
# ============================================================
if not ROOT.exists():
    print(f"ERROR: Folder does not exist:\n{ROOT}")
    raise SystemExit

all_files = [p for p in ROOT.rglob("*") if p.is_file() and OUT not in p.parents]

copied_counts = {}
ignored_count = 0
ignored_examples = []
copied_log = []
ignored_log = []

print("=" * 80)
print("SCIENCE ONLY ORGANISER")
print("=" * 80)
print(f"Root folder: {ROOT}")
print(f"Files scanned: {len(all_files)}")
print()

for file_path in all_files:
    category = classify_file(file_path)

    if category is None:
        ignored_count += 1
        ignored_log.append(str(file_path.relative_to(ROOT)))
        if len(ignored_examples) < 25:
            ignored_examples.append(str(file_path.relative_to(ROOT)))
        continue

    dest_folder = OUT / Path(category)
    safe_copy(file_path, dest_folder)

    copied_counts[category] = copied_counts.get(category, 0) + 1
    copied_log.append(f"{file_path.relative_to(ROOT)}  -->  {category}")
    print(f"COPIED: {file_path.name}  -->  {category}")

# ============================================================
# 4. WRITE REPORT
# ============================================================
OUT.mkdir(parents=True, exist_ok=True)
report_path = OUT / "science_only_report.txt"

with open(report_path, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("SCIENCE ONLY ORGANISER REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Root folder: {ROOT}\n")
    f.write(f"Files scanned: {len(all_files)}\n\n")

    f.write("COPIED COUNTS\n")
    f.write("-" * 80 + "\n")
    for key in sorted(copied_counts):
        f.write(f"{key:35} : {copied_counts[key]}\n")

    f.write(f"\nIgnored files: {ignored_count}\n\n")

    f.write("COPIED FILES\n")
    f.write("-" * 80 + "\n")
    for line in copied_log:
        f.write(line + "\n")

    f.write("\nIGNORED FILES\n")
    f.write("-" * 80 + "\n")
    for line in ignored_log:
        f.write(line + "\n")

# ============================================================
# 5. SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

for key in sorted(copied_counts):
    print(f"{key:35} : {copied_counts[key]}")

print(f"\nIgnored files: {ignored_count}")

if ignored_examples:
    print("\nExamples of ignored files:")
    for ex in ignored_examples:
        print(f" - {ex}")

print(f"\nOrganised science-only folder created at:\n{OUT}")
print(f"Report saved to:\n{report_path}")