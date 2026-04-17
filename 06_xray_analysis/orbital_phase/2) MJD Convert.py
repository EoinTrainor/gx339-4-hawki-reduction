####        2         ####

# title: Convert ESO FITS DATE-OBS timestamps to MJD (robust)

import re
import pandas as pd
from astropy.time import Time

INPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/1) eso_extracted_timestamps.csv"
OUTPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"

# Read the CSV (now created from FITS headers)
df = pd.read_csv(INPUT_FILE)

if "FULL_TIMESTAMP" not in df.columns:
    raise KeyError("Column FULL_TIMESTAMP not found in the CSV file.")

# Clean up the column
df = df.dropna(subset=["FULL_TIMESTAMP"]).copy()
df["FULL_TIMESTAMP"] = df["FULL_TIMESTAMP"].astype(str).str.strip()

# ISO timestamp pattern: YYYY-MM-DDThh:mm:ss(.fraction)?
iso_pattern = re.compile(
    r"^\d{4}-\d{2}-\d{2}T"
    r"\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?$"
)

valid_mask = df["FULL_TIMESTAMP"].apply(lambda x: bool(iso_pattern.match(x)))
bad_rows = df[~valid_mask]

print(f"Total rows in file: {len(df)}")
print(f"Valid ISO timestamps: {valid_mask.sum()}")
print(f"Invalid / skipped rows: {len(bad_rows)}")

if not bad_rows.empty:
    print("\nExamples of invalid rows (first 5):")
    print(bad_rows.head())

df_valid = df[valid_mask].copy()

time_list = df_valid["FULL_TIMESTAMP"].tolist()
times = Time(time_list, format="isot", scale="utc")

df_valid["MJD_UTC"] = times.mjd
df_valid["JD_UTC"] = times.jd

df_valid.to_csv(OUTPUT_FILE, index=False)

print(f"\nSuccessfully converted {len(df_valid)} timestamps to MJD.")
print(f"Output written to: {OUTPUT_FILE}")
