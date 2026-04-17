# title: Chronologise HAWK-I FITS by DATE-OBS (with progress bar + logging)

import os
import csv
import shutil
from datetime import datetime

from astropy.io import fits
from tqdm import tqdm


# ---------------- USER INPUTS ----------------
INPUT_DIR  = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images"
OUTPUT_DIR = os.path.join("C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Chronological DATE-OBS")

COPY_INSTEAD_OF_RENAME = True
APPEND_ORIGINAL_STEM = True
# --------------------------------------------


def find_date_obs_in_hdul(hdul):
    if "DATE-OBS" in hdul[0].header:
        return str(hdul[0].header["DATE-OBS"]).strip()

    for hdu in hdul[1:]:
        if hdu.header is not None and "DATE-OBS" in hdu.header:
            return str(hdu.header["DATE-OBS"]).strip()

    return None


def parse_date_obs(date_str):
    return datetime.fromisoformat(date_str)


def safe_dateobs_filename(dt):
    return dt.strftime("%Y-%m-%d_%H-%M-%S.%f")


def unique_path(path):
    base, ext = os.path.splitext(path)
    if not os.path.exists(path):
        return path

    k = 1
    while True:
        candidate = f"{base}_{k:03d}{ext}"
        if not os.path.exists(candidate):
            return candidate
        k += 1


def main():
    print("\n=== HAWK-I FITS DATE-OBS Chronologiser ===\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("🔍 Scanning input directory...")
    fits_files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".fits", ".fit", ".fts"))
        and os.path.isfile(os.path.join(INPUT_DIR, f))
    ]

    if not fits_files:
        raise RuntimeError(f"No FITS files found in:\n{INPUT_DIR}")

    print(f"   Found {len(fits_files)} FITS files.\n")

    records = []
    problems = []

    print("📖 Reading DATE-OBS from FITS headers...")
    for fname in tqdm(fits_files, desc="Reading headers", unit="file"):
        fpath = os.path.join(INPUT_DIR, fname)

        try:
            with fits.open(fpath, memmap=False) as hdul:
                date_obs_raw = find_date_obs_in_hdul(hdul)

            if date_obs_raw is None:
                problems.append((fname, "Missing DATE-OBS"))
                continue

            dt = parse_date_obs(date_obs_raw)
            records.append((dt, fname, date_obs_raw))

        except Exception as e:
            problems.append((fname, f"Error reading/parsing DATE-OBS: {e}"))

    print(f"\n   Successfully read DATE-OBS from {len(records)} file(s).")
    if problems:
        print(f"   ⚠️  {len(problems)} file(s) had issues.")

    print("\n🗂️  Sorting files chronologically...")
    records.sort(key=lambda t: t[0])

    print("💾 Writing files in DATE-OBS order...")
    log_path = os.path.join(OUTPUT_DIR, "dateobs_rename_log.csv")

    with open(log_path, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "order",
            "date_obs_raw",
            "date_obs_parsed",
            "old_filename",
            "new_filename",
            "action"
        ])

        for i, (dt, old_name, date_obs_raw) in enumerate(
                tqdm(records, desc="Copying files", unit="file"), start=1):

            old_path = os.path.join(INPUT_DIR, old_name)
            date_tag = safe_dateobs_filename(dt)
            old_stem, old_ext = os.path.splitext(old_name)

            if APPEND_ORIGINAL_STEM:
                new_name = f"{date_tag}__{old_stem}{old_ext}"
            else:
                new_name = f"{date_tag}{old_ext}"

            new_path = unique_path(os.path.join(OUTPUT_DIR, new_name))
            new_name_final = os.path.basename(new_path)

            if COPY_INSTEAD_OF_RENAME:
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
                action
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
