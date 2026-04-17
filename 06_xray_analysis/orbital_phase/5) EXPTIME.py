####        5           ####

# title: Summarise EXPTIME per night for GX 339-4

import pandas as pd
from astropy.time import Time

INPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"
OUTPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/5) gx3394_exptime_by_night.csv"

df = pd.read_csv(INPUT_FILE)

if "MJD_UTC" not in df.columns:
    raise KeyError("Column 'MJD_UTC' not found — run the MJD conversion first.")
if "EXPTIME" not in df.columns:
    raise KeyError("Column 'EXPTIME' not found — make sure it was written from FITS headers.")

# Convert MJD → calendar date
times = Time(df["MJD_UTC"].values, format="mjd", scale="utc").to_datetime()
df["OBS_DATE"] = [t.date().isoformat() for t in times]

# Sum exposure per date (EXPTIME assumed in seconds)
summary = (
    df.groupby("OBS_DATE")["EXPTIME"]
      .sum()
      .reset_index()
      .rename(columns={"EXPTIME": "TOTAL_EXPTIME_S"})
)

summary["TOTAL_EXPTIME_HR"] = summary["TOTAL_EXPTIME_S"] / 3600.0

summary.to_csv(OUTPUT_FILE, index=False)

print(summary)
print()
print(f"Total on-source time: {summary['TOTAL_EXPTIME_HR'].sum():.2f} hours")
print(f"Output written to: {OUTPUT_FILE}")
