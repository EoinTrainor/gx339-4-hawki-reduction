# title: Build GX 339-4 observation table with times, phases, and phase coverage

import pandas as pd
import numpy as np

# --- INPUT / OUTPUT FILES ---
INPUT_FILE  = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"
OUTPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/6) gx3394_observation_phase_table.csv"

# --- ORBITAL PERIOD (days) ---
P_ORB_DAYS = 1.7587   # keep consistent with rest of your analysis

# --- LOAD DATA ---
df = pd.read_csv(INPUT_FILE)

# Basic sanity checks
required_cols = ["MJD_UTC", "FULL_TIMESTAMP", "EXPTIME", "FILENAME"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise KeyError(f"Missing required columns in {INPUT_FILE}: {missing}")

# Ensure EXPTIME is numeric (seconds)
df["EXPTIME"] = pd.to_numeric(df["EXPTIME"], errors="coerce")
df = df.dropna(subset=["EXPTIME", "MJD_UTC"]).copy()

# --- DEFINE T0 FROM HEIDA et al. (2017) ---
# Ephemeris reference time (phase zero)
T0_MJD = 57529.397  # MJD (Heida et al. 2017)
T0_ERR = 0.003      # days (1σ)

# Also compute earliest observation time for comparison
t_min = df["MJD_UTC"].min()
print(f"T0 (Heida et al. 2017) = {T0_MJD:.6f} ± {T0_ERR:.3f} MJD")
print(f"t_min (earliest observed MJD) = {t_min:.6f} MJD")
print(f"Offset (t_min − T0) = {t_min - T0_MJD:.6f} days")

# --- TIME AND PHASE CALCULATIONS ---

# Exposure duration in days
dt_days = df["EXPTIME"] / 86400.0

# Start / mid / end times in MJD
df["MJD_START"] = df["MJD_UTC"]
df["MJD_MID"]   = df["MJD_START"] + 0.5 * dt_days
df["MJD_END"]   = df["MJD_START"] + dt_days

# Helper to compute phase in [0,1)
def phase_from_mjd(mjd):
    return ((mjd - T0_MJD) / P_ORB_DAYS) % 1.0

df["PHASE_START"] = phase_from_mjd(df["MJD_START"])
df["PHASE_MID"]   = phase_from_mjd(df["MJD_MID"])
df["PHASE_END"]   = phase_from_mjd(df["MJD_END"])

# Phase span of each exposure (always positive, small)
df["PHASE_SPAN"] = dt_days / P_ORB_DAYS   # Δφ = Δt / P

# For convenience, also express phase span in degrees
df["PHASE_SPAN_DEG"] = df["PHASE_SPAN"] * 360.0

# Optional: also add exposure duration in minutes / hours
df["EXPTIME_MIN"] = df["EXPTIME"] / 60.0
df["EXPTIME_HR"]  = df["EXPTIME"] / 3600.0

# --- BUILD A CLEAN, SUPERVISOR-FRIENDLY TABLE ---

cols_for_supervisor = [
    "FILENAME",
    "FULL_TIMESTAMP",   # UT start time
    "EXPTIME",          # seconds
    "EXPTIME_MIN",
    "PHASE_START",
    "PHASE_MID",
    "PHASE_END",
    "PHASE_SPAN",
    "PHASE_SPAN_DEG",
]

table = df[cols_for_supervisor].sort_values("FULL_TIMESTAMP")

table.to_csv(OUTPUT_FILE, index=False)

print(f"Wrote observation phase table with {len(table)} rows → {OUTPUT_FILE}")
