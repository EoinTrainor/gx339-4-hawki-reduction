# title: Extract ESO ZIP, fix Windows filenames, decompress .fits.Z to .fits (with timer + progress bar)

import zipfile
import os
from pathlib import Path
import shutil
import time

# ---- REQUIRED for .Z decompression ----
# Install once in your environment: pip install unlzw3
import unlzw3

# ---- OPTIONAL (nice progress bar) ----
# Install once: pip install tqdm
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

# TODO: change this to the path of your ESO zip file:
ZIP_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images/archive (1).zip"

# TODO: choose output folder (your choice)
OUTPUT_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images/Science Images FITS (Download)"

# If True: delete the compressed .Z after successful decompression
DELETE_Z_AFTER = False

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def make_windows_safe(name: str) -> str:
    """
    Replace characters that Windows does not allow in filenames.
    Here we mainly care about ':' in the ESO timestamps.
    """
    bad_chars = {
        ":": "-",   # 2025-06-04T07:48:45.939 → 2025-06-04T07-48-45.939
        "*": "_",
        "?": "_",
        "\"": "_",
        "<": "_",
        ">": "_",
        "|": "_"
    }
    for bad, repl in bad_chars.items():
        name = name.replace(bad, repl)
    return name


def is_z_file(path: Path) -> bool:
    """Return True for .Z or .fits.Z"""
    name = path.name.lower()
    return name.endswith(".z") or name.endswith(".fits.z")


def decompressed_fits_path(z_path: Path) -> Path:
    """
    Convert:
      something.fits.Z  -> something.fits
      something.Z       -> something   (but we will append .fits if missing)
    """
    name = z_path.name
    if name.lower().endswith(".z"):
        name_noz = name[:-2]  # strip trailing ".Z"
    else:
        name_noz = name

    out = z_path.with_name(name_noz)

    # If after removing .Z it doesn't end in .fits, enforce .fits
    if not out.name.lower().endswith((".fits", ".fit")):
        out = out.with_suffix(out.suffix + ".fits")

    return out


def decompress_z_file(z_path: Path) -> Path:
    """
    Decompress a Unix-compress .Z file to a FITS file.
    Returns output .fits path.
    """
    out_path = decompressed_fits_path(z_path)

    # If already exists, don’t redo unless you want to.
    if out_path.exists():
        return out_path

    with open(z_path, "rb") as f_in:
        comp = f_in.read()

    raw = unlzw3.unlzw(comp)

    # Write decompressed bytes
    with open(out_path, "wb") as f_out:
        f_out.write(raw)

    # Optionally delete compressed .Z after success
    if DELETE_Z_AFTER:
        try:
            z_path.unlink()
        except Exception as e:
            print(f"[WARN] Could not delete {z_path.name}: {e}")

    return out_path


def _format_seconds(sec: float) -> str:
    sec = float(sec)
    if sec < 60:
        return f"{sec:.1f} s"
    if sec < 3600:
        return f"{sec/60:.2f} min"
    return f"{sec/3600:.2f} hr"


def _iter_progress(iterable, desc: str, total: int = None):
    """
    Progress wrapper:
    - uses tqdm if installed
    - otherwise prints a simple percentage progress
    """
    if _HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, unit="file")

    # fallback simple text progress
    total = total if total is not None else len(iterable)
    def gen():
        for i, item in enumerate(iterable, start=1):
            pct = 100.0 * i / max(1, total)
            print(f"{desc}: {i}/{total} ({pct:5.1f}%)", end="\r")
            yield item
        print()  # newline after done
    return gen()


# -------------------- TIMING START --------------------
t0 = time.perf_counter()

# -------------------- 1) EXTRACT ZIP (SAFE FILENAMES) --------------------
extract_start = time.perf_counter()
extracted_paths = []

with zipfile.ZipFile(ZIP_FILE, "r") as z:
    members = [m for m in z.infolist() if not m.is_dir()]

    for member in _iter_progress(members, desc="Extracting", total=len(members)):
        original_name = member.filename
        safe_name = make_windows_safe(original_name)

        rel_path = Path(safe_name)
        target_path = Path(OUTPUT_DIR) / rel_path
        target_path.parent.mkdir(parents=True, exist_ok=True)

        with z.open(member, "r") as src, open(target_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        extracted_paths.append(target_path)

extract_end = time.perf_counter()
print("\nExtraction complete.")
print(f"Files extracted to: {OUTPUT_DIR}")
print(f"Extraction time: {_format_seconds(extract_end - extract_start)}")


# -------------------- 2) DECOMPRESS ANY .Z FILES --------------------
decomp_start = time.perf_counter()
z_files = [p for p in extracted_paths if is_z_file(p)]

if not z_files:
    print("\nNo .Z files found to decompress.")
else:
    print(f"\nFound {len(z_files)} .Z file(s). Decompressing...")

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for zp in _iter_progress(z_files, desc="Decompressing", total=len(z_files)):
        try:
            out_path = decompressed_fits_path(zp)
            if out_path.exists():
                n_skipped += 1
                continue

            decompress_z_file(zp)
            n_done += 1

        except Exception as e:
            n_failed += 1
            print(f"\n[ERROR] Failed to decompress {zp.name}: {e}")

    print(f"\nDecompression summary: done={n_done}, skipped={n_skipped}, failed={n_failed}")

decomp_end = time.perf_counter()
print(f"Decompression time: {_format_seconds(decomp_end - decomp_start)}")


# -------------------- TOTAL TIME --------------------
t1 = time.perf_counter()
print(f"\nAll done. Total processing time: {_format_seconds(t1 - t0)}")
