"""
00_diagnostics.py
-----------------
Step 0: Data diagnostics — run this FIRST before any calibration.

Inspects your raw FITS files and prints a structured summary report covering:
  - FITS file structure (single extension vs multi-extension MEF)
  - Image dimensions and data types per detector
  - Key header keywords (EXPTIME, filter, readout mode, dither offsets)
  - Inventory of darks, flats and science frames
  - Consistency checks across the dataset

Usage:
    python pipeline/00_diagnostics.py

Output:
    Printed report to terminal + saved to logs/diagnostics.txt
"""

import sys
from pathlib import Path

# Allow running from repo root or pipeline/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from astropy.io import fits
from astropy.table import Table
import config

# ─── Header keywords to extract (HAWK-I specific) ────────────────────────────
KEYWORDS = [
    "NAXIS1", "NAXIS2",          # Image dimensions
    "EXPTIME",                    # Exposure time
    "HIERARCH ESO INS FILT1 NAME",  # Filter name
    "HIERARCH ESO DET READ MODE",   # Readout mode
    "HIERARCH ESO SEQ CUMOFFSETX",  # Dither X offset
    "HIERARCH ESO SEQ CUMOFFSETY",  # Dither Y offset
    "HIERARCH ESO OBS NAME",        # OB name
    "HIERARCH ESO OBS ID",          # OB ID
    "DATE-OBS",                     # Observation date/time
    "MJD-OBS",                      # Modified Julian Date
    "OBJECT",                       # Object name
]


def inspect_file(filepath: Path) -> dict:
    """Open a FITS file and extract structural info and header keywords."""
    info = {"filename": filepath.name, "path": str(filepath)}

    try:
        with fits.open(filepath, memmap=True) as hdul:
            info["n_extensions"] = len(hdul)
            info["is_mef"] = len(hdul) > 1

            # Inspect primary HDU or first image extension
            for i, hdu in enumerate(hdul):
                if hdu.data is not None:
                    info["first_image_ext"] = i
                    info["shape"] = hdu.data.shape
                    info["dtype"] = str(hdu.data.dtype)
                    break
            else:
                info["first_image_ext"] = None
                info["shape"] = None
                info["dtype"] = None

            # Extract keywords from primary header
            primary_header = hdul[0].header
            for kw in KEYWORDS:
                try:
                    info[kw] = primary_header[kw]
                except KeyError:
                    # Try first image extension header if not in primary
                    try:
                        ext = info.get("first_image_ext", 1) or 1
                        info[kw] = hdul[ext].header[kw]
                    except (KeyError, IndexError):
                        info[kw] = "NOT FOUND"

    except Exception as e:
        info["error"] = str(e)

    return info


def summarise_folder(folder: Path, label: str) -> list:
    """Inspect all FITS files in a folder and return list of info dicts."""
    fits_files = sorted(
        list(folder.glob("*.fits")) +
        list(folder.glob("*.fit")) +
        list(folder.glob("*.fts")) +
        list(folder.glob("*.FITS"))
    )

    if not fits_files:
        print(f"\n  [WARNING] No FITS files found in {folder}")
        return []

    print(f"\n  Found {len(fits_files)} FITS files in {label}")
    results = []
    for f in fits_files:
        results.append(inspect_file(f))

    return results


def print_folder_report(results: list, label: str):
    """Print a structured summary for one folder's worth of files."""
    if not results:
        return

    print(f"\n{'='*70}")
    print(f"  {label.upper()}")
    print(f"{'='*70}")

    # Check for errors
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\n  [!] {len(errors)} file(s) could not be opened:")
        for e in errors:
            print(f"      {e['filename']}: {e['error']}")

    good = [r for r in results if "error" not in r]
    if not good:
        return

    # MEF vs single extension
    mef_count = sum(1 for r in good if r.get("is_mef"))
    print(f"\n  File structure:")
    print(f"    Multi-extension (MEF): {mef_count} / {len(good)}")
    print(f"    Single extension     : {len(good) - mef_count} / {len(good)}")
    if good[0].get("n_extensions"):
        ext_counts = set(r["n_extensions"] for r in good)
        print(f"    Extensions per file  : {ext_counts}")

    # Image shapes
    shapes = set(str(r.get("shape")) for r in good)
    print(f"\n  Image shapes (all extensions): {shapes}")

    dtypes = set(r.get("dtype") for r in good)
    print(f"  Data types: {dtypes}")

    # Exposure times
    exptimes = set(r.get("EXPTIME") for r in good)
    print(f"\n  Exposure times (s): {exptimes}")

    # Filters
    filters = set(r.get("HIERARCH ESO INS FILT1 NAME") for r in good)
    print(f"  Filters: {filters}")

    # Readout modes
    modes = set(r.get("HIERARCH ESO DET READ MODE") for r in good)
    print(f"  Readout modes: {modes}")

    # OB names (science only)
    ob_names = set(r.get("HIERARCH ESO OBS NAME") for r in good
                   if r.get("HIERARCH ESO OBS NAME") not in ("NOT FOUND", None))
    if ob_names:
        print(f"\n  OB names ({len(ob_names)} unique):")
        for ob in sorted(ob_names):
            print(f"    {ob}")

    # Date range
    dates = [r.get("DATE-OBS") for r in good if r.get("DATE-OBS") != "NOT FOUND"]
    if dates:
        print(f"\n  Date range:")
        print(f"    First : {min(dates)}")
        print(f"    Last  : {max(dates)}")

    # Dither offsets (science)
    x_offsets = [r.get("HIERARCH ESO SEQ CUMOFFSETX") for r in good
                 if r.get("HIERARCH ESO SEQ CUMOFFSETX") not in ("NOT FOUND", None)]
    y_offsets = [r.get("HIERARCH ESO SEQ CUMOFFSETY") for r in good
                 if r.get("HIERARCH ESO SEQ CUMOFFSETY") not in ("NOT FOUND", None)]
    if x_offsets:
        unique_x = sorted(set(x_offsets))
        unique_y = sorted(set(y_offsets))
        print(f"\n  Dither offsets:")
        print(f"    Unique X offsets: {unique_x}")
        print(f"    Unique Y offsets: {unique_y}")
        print(f"    Dither pattern size: {len(unique_x)} x {len(unique_y)}")

    # First file detailed header dump
    print(f"\n  --- First file header sample: {good[0]['filename']} ---")
    for kw in KEYWORDS:
        val = good[0].get(kw, "NOT FOUND")
        short_kw = kw.replace("HIERARCH ESO ", "")
        print(f"    {short_kw:<35} = {val}")


def main():
    config.make_output_dirs()
    log_lines = []

    print("\n" + "="*70)
    print("  GX 339-4 HAWK-I Pipeline — Step 0: Data Diagnostics")
    print("="*70)
    print(f"\n  Data root   : {config.DATA_ROOT}")
    print(f"  Darks dir   : {config.DARKS_DIR}")
    print(f"  Flats dir   : {config.FLATS_DIR}")
    print(f"  Science dir : {config.SCIENCE_DIR}")

    # Check directories exist
    for label, path in [("Darks", config.DARKS_DIR),
                        ("Flats", config.FLATS_DIR),
                        ("Science", config.SCIENCE_DIR)]:
        if not path.exists():
            print(f"\n  [ERROR] {label} directory not found: {path}")
            print("  Please update config.py with your correct data paths.")
            sys.exit(1)

    # Inspect each folder
    dark_results    = summarise_folder(config.DARKS_DIR, "Darks")
    flat_results    = summarise_folder(config.FLATS_DIR, "Flats")
    science_results = summarise_folder(config.SCIENCE_DIR, "Science")

    print_folder_report(dark_results,    "Darks (Ks)")
    print_folder_report(flat_results,    "Flats (Ks)")
    print_folder_report(science_results, "Science frames (Ks)")

    # Cross-checks
    print(f"\n{'='*70}")
    print("  CROSS-CHECKS")
    print(f"{'='*70}")

    all_good = dark_results + flat_results + science_results
    good = [r for r in all_good if "error" not in r]

    exptimes_dark    = set(r.get("EXPTIME") for r in dark_results if "error" not in r)
    exptimes_science = set(r.get("EXPTIME") for r in science_results if "error" not in r)
    exptimes_flat    = set(r.get("EXPTIME") for r in flat_results if "error" not in r)

    print(f"\n  Exposure time consistency:")
    print(f"    Dark exptimes    : {exptimes_dark}")
    print(f"    Flat exptimes    : {exptimes_flat}")
    print(f"    Science exptimes : {exptimes_science}")

    # Check if dark exptimes cover science exptimes
    missing_darks = exptimes_science - exptimes_dark
    if missing_darks:
        print(f"\n  [WARNING] No darks found for science exposure times: {missing_darks}")
    else:
        print(f"\n  [OK] Dark exposure times cover all science exposure times.")

    print(f"\n{'='*70}")
    print("  Diagnostics complete. Review the report above before proceeding.")
    print("  Next step: python pipeline/01_master_dark.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
