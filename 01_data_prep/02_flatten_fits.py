# Flatten HAWKI FITS files into one folder

from pathlib import Path

ROOT = Path(r"C:\Astronomy\GX 339-4 Raw Data\ESO_RAW_GX339_4\01_Raw_HAWKI_Decompressed")

REMOVE_EMPTY_FOLDERS = True

if not ROOT.exists():
    print(f"ERROR: Folder not found:\n{ROOT}")
    raise SystemExit

# Find FITS files below ROOT, but not already in ROOT itself
nested_fits = [p for p in ROOT.rglob("*.fits") if p.parent != ROOT]

print("=" * 80)
print("FLATTEN DECOMPRESSED HAWKI FITS FILES")
print("=" * 80)
print(f"Root folder: {ROOT}")
print(f"Nested FITS files found: {len(nested_fits)}")
print()

moved = 0
skipped = 0

for src in nested_fits:
    dest = ROOT / src.name

    # Handle duplicate names safely
    if dest.exists():
        if src.resolve() == dest.resolve():
            skipped += 1
            print(f"SKIPPED (same file): {src}")
            continue

        counter = 1
        while True:
            new_dest = ROOT / f"{src.stem}_{counter}{src.suffix}"
            if not new_dest.exists():
                dest = new_dest
                break
            counter += 1

    print(f"MOVE:\n  {src}\n  -> {dest}\n")
    src.rename(dest)
    moved += 1

# Remove empty folders afterward
removed_dirs = 0
if REMOVE_EMPTY_FOLDERS:
    # deepest first
    all_dirs = sorted([p for p in ROOT.rglob("*") if p.is_dir()], key=lambda x: len(x.parts), reverse=True)
    for d in all_dirs:
        try:
            if d != ROOT and not any(d.iterdir()):
                d.rmdir()
                removed_dirs += 1
        except Exception:
            pass

print("=" * 80)
print("DONE")
print("=" * 80)
print(f"Moved FITS files:     {moved}")
print(f"Skipped FITS files:   {skipped}")
print(f"Removed empty folders:{removed_dirs}")