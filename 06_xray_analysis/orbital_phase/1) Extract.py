####         1         #####

# title: Extract DATE-OBS and EXPTIME from HAWK-I FITS headers

import os
import csv
from astropy.io import fits

# --- USER PATHS ---
FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS" 
OUTPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/1) eso_extracted_timestamps.csv"

# Which extensions / header keywords to use
FITS_EXTENSIONS = (".fits", ".fit", ".fz")  # adjust if needed
DATE_KEY = "DATE-OBS"
EXPTIME_KEY = "EXPTIME"

rows = []

for root, _, files in os.walk(FITS_DIR):
    for fname in files:
        if not fname.lower().endswith(FITS_EXTENSIONS):
            continue

        full_path = os.path.join(root, fname)

        try:
            with fits.open(full_path) as hdul:
                hdr = hdul[0].header

                date_obs = hdr.get(DATE_KEY, None)
                exptime = hdr.get(EXPTIME_KEY, None)

                if date_obs is None:
                    print(f"[WARN] {fname}: missing {DATE_KEY}, skipping.")
                    continue

                # Normalise DATE-OBS → FULL_TIMESTAMP, DATE, TIME
                # Typical ESO format: 'YYYY-MM-DDThh:mm:ss.sss'
                date_obs = str(date_obs).strip()

                if "T" in date_obs:
                    date_part, time_part = date_obs.split("T", 1)
                else:
                    # Fallback if it's 'YYYY-MM-DD hh:mm:ss' or just a date
                    parts = date_obs.replace(" ", "T").split("T", 1)
                    date_part = parts[0]
                    time_part = parts[1] if len(parts) > 1 else ""

                rows.append(
                    {
                        "FILENAME": fname,
                        "FULL_TIMESTAMP": date_obs,
                        "DATE": date_part,
                        "TIME": time_part,
                        "EXPTIME": exptime,
                    }
                )

        except Exception as e:
            print(f"[ERROR] Could not read FITS file {full_path}: {e}")

# Write CSV
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

with open(OUTPUT_FILE, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["FILENAME", "FULL_TIMESTAMP", "DATE", "TIME", "EXPTIME"],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Extracted {len(rows)} rows from FITS headers → {OUTPUT_FILE}")
