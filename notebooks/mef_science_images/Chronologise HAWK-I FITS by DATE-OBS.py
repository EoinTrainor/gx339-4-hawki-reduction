# title: Chronologise HAWK-I MEF Science FITS by DATE-OBS (preserve multi-extension structure)

import os
import csv
import shutil
from datetime import datetime

from astropy.io import fits
from tqdm import tqdm


# ---------------- USER INPUTS ----------------
INPUT_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images/Science Images FITS (Download)"

OUTPUT_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images/SI_Chronologic_DATE_OBS"

# If True: copy into OUTPUT_DIR (recommended, non-destructive)
# If False: move (rename) into OUTPUT_DIR
COPY_INSTEAD_OF_MOVE = True

# If True: keep original ADP stem in filename after DATE-OBS (recommended)
APPEND_ORIGINAL_STEM = True
# --------------------------------------------


def find_date_obs_any_hdu(hdul):
    """
    Find DATE-OBS, preferring primary header, else searching extensions.
    Returns raw DATE-OBS string or None.
    """
    if "DATE-OBS" in hdul[0].header:
        return str(hdul[0].header["DATE-OBS"]).strip()

    for hdu in hdul[1:]:
        if hdu.header is not None and "DATE-OBS" in hdu.header:
            return str(hdu.header["DATE-OBS"]).strip()

    return None


def parse_date_obs(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str)


def safe_dateobs_filename(dt: datetime) -> str:
    # Windows safe sortable timestamp
    return dt.strftime("%Y-%m-%d_%H-%M-%S.%f")


def unique_path(path: str) -> str:
    """
    If the target filename already exists, append _001, _002, etc.
    """
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path

    k = 1
    while True:
        candidate = f"{base}_{k:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


def is_mef(hdul) -> bool:
    """
    A practical MEF check: primary HDU usually has no data,
    and there are multiple image extensions with data.
    """
    img_ext_count = 0
    for i, hdu in enumerate(hdul):
        if hdu.data is None:
            continue
        try:
            if hasattr(hdu.data, "ndim") and hdu.data.ndim >= 2:
                img_ext_count += 1
        except Exception:
            pass
    return img_ext_count >= 2


def main():
    print("\n=== HAWK-I MEF Science FITS DATE-OBS Chronologiser (preserve MEF) ===\n")

    if not os.path.isdir(INPUT_DIR):
        raise NotADirectoryError(f"INPUT_DIR not found:\n{INPUT_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fits_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".fits", ".fit", ".fts"))
        and os.path.isfile(os.path.join(INPUT_DIR, f))
    ]

    if not fits_files:
        raise RuntimeError(f"No FITS files found in:\n{INPUT_DIR}")

    print(f"🔍 Found {len(fits_files)} FITS file(s) in input folder.\n")

    records = []
    problems = []

    print("📖 Reading DATE-OBS from FITS headers (no rewriting, just reading)...")
    for fname in tqdm(fits_files, desc="Reading headers", unit="file"):
        fpath = os.path.join(INPUT_DIR, fname)

        try:
            with fits.open(fpath, memmap=False) as hdul:
                date_obs_raw = find_date_obs_any_hdu(hdul)

                if date_obs_raw is None:
                    problems.append((fname, "Missing DATE-OBS"))
                    continue

                dt = parse_date_obs(date_obs_raw)

                # For your sanity, also note whether file is MEF-like
                mef_flag = is_mef(hdul)

            records.append((dt, fname, date_obs_raw, mef_flag))

        except Exception as e:
            problems.append((fname, f"Error reading/parsing DATE-OBS: {e}"))

    print(f"\n   ✅ DATE-OBS read for {len(records)} file(s).")
    if problems:
        print(f"   ⚠️  {len(problems)} file(s) had issues.\n")

    print("🗂️  Sorting chronologically by DATE-OBS...")
    records.sort(key=lambda t: t[0])

    log_path = os.path.join(OUTPUT_DIR, "dateobs_rename_log.csv")
    print("💾 Copying/moving MEF FITS files in DATE-OBS order (structure preserved)...")

    with open(log_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "order",
            "date_obs_raw",
            "date_obs_parsed",
            "old_filename",
            "new_filename",
            "action",
            "mef_detected"
        ])

        for i, (dt, old_name, date_obs_raw, mef_flag) in enumerate(
                tqdm(records, desc="Writing chronological files", unit="file"), start=1):

            old_path = os.path.join(INPUT_DIR, old_name)
            date_tag = safe_dateobs_filename(dt)
            old_stem, old_ext = os.path.splitext(old_name)

            if APPEND_ORIGINAL_STEM:
                new_name = f"{date_tag}__{old_stem}{old_ext}"
            else:
                new_name = f"{date_tag}{old_ext}"

            new_path = unique_path(os.path.join(OUTPUT_DIR, new_name))
            new_name_final = os.path.basename(new_path)

            # IMPORTANT: We are NOT touching the FITS content.
            # We are only copying/moving the whole file as-is.
            if COPY_INSTEAD_OF_MOVE:
                shutil.copy2(old_path, new_path)
                action = "copied"
            else:
                shutil.move(old_path, new_path)
                action = "moved/renamed"

            writer.writerow([
                i,
                date_obs_raw,
                dt.isoformat(),
                old_name,
                new_name_final,
                action,
                mef_flag
            ])

    if problems:
        prob_path = os.path.join(OUTPUT_DIR, "dateobs_problems.csv")
        with open(prob_path, "w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(["filename", "issue"])
            for row in problems:
                writer.writerow(row)

        print(f"\n⚠️  Problem report written to:\n{prob_path}")

    print("\n✅ Done.")
    print(f"   Output folder: {OUTPUT_DIR}")
    print(f"   Log file:      {log_path}")
    print(f"   Processed:     {len(records)} file(s)")
    print(f"   Problems:      {len(problems)} file(s)\n")


if __name__ == "__main__":
    main()
