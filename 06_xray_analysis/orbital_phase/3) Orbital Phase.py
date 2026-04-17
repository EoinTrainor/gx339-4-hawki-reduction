####         3          ####

# title: Compute orbital phase for GX 339-4 using min(MJD) as T0

import pandas as pd
import numpy as np

INPUT_FILE  = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"
OUTPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/3) eso_times_with_phase.csv"

# ---- Orbital period (days) ----
P_ORB_DAYS = 1.7587   # keep consistent with your analysis

df = pd.read_csv(INPUT_FILE)

if "MJD_UTC" not in df.columns:
    raise KeyError("MJD_UTC not found — run the MJD conversion first.")

# ---- Choose T0 from Heida et al. (2017) ----
T0_MJD = 57529.397  # MJD
T0_ERR = 0.003      # days (1σ)

# Also compute earliest observation time for comparison
t_min = df["MJD_UTC"].min()
print(f"T0 (Heida et al. 2017) = {T0_MJD:.6f} ± {T0_ERR:.3f} MJD")
print(f"t_min (earliest observed MJD) = {t_min:.6f} MJD")
print(f"Offset (t_min − T0) = {t_min - T0_MJD:.6f} days")
# ---- Compute orbital phase in [0, 1) ----
df["PHASE"] = ((df["MJD_UTC"] - T0_MJD) / P_ORB_DAYS) % 1.0

# Store T0 for traceability
df["T0_MJD_REF"] = T0_MJD

# Sort by observation time
df = df.sort_values("MJD_UTC")

df.to_csv(OUTPUT_FILE, index=False)

print(f"Wrote {len(df)} rows with phase values → {OUTPUT_FILE}")

