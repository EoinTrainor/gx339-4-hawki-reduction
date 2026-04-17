# title: Swift BAT Lightcurve with ESO Observing Block Windows Overlay (MJD) — FIXED for your file formats

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------- USER INPUTS ----------------
# Swift BAT daily CSV (yours has real headers: TIME, RATE, ERROR, YEAR, DAY, ...)
SWIFT_CSV = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/3) Outburst/Data & Analysis/GX339-4 Lightcurve/GX339-4 Lightcurve.csv"

# ESO observing log CSV (has: Observing Block, EXPTIME, MJD_UTC, ...)
ESO_CSV = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"

# Column names (match your actual CSV headers)
COL_BLOCK = "Observing Block"
COL_MJD   = "MJD_UTC"
COL_EXPT  = "EXPTIME"   # seconds

SWIFT_TIME_COL  = "TIME"    # IMPORTANT: In your Swift file this is MJD (days)
SWIFT_RATE_COL  = "RATE"
SWIFT_ERROR_COL = "ERROR"

# Output
OUT_PNG = "swift_bat_with_eso_observing_blocks.png"

# Plot controls
PAD_DAYS = 2.0
SHOW_ERRORBARS = True
SHADE_ALPHA = 0.18
DASHED_LW = 1.2
LABEL_FONTSIZE = 10
FIGSIZE = (13, 6)


# ---------------- LOAD SWIFT (your CSV already has correct headers) ----------------
swift = pd.read_csv(SWIFT_CSV, comment="#")  # comment safe; won't hurt even if none

# Basic column presence check
for c in (SWIFT_TIME_COL, SWIFT_RATE_COL):
    if c not in swift.columns:
        raise ValueError(f"Swift CSV missing required column '{c}'. Found: {list(swift.columns)}")

# Numeric coercion
swift[SWIFT_TIME_COL] = pd.to_numeric(swift[SWIFT_TIME_COL], errors="coerce")
swift[SWIFT_RATE_COL] = pd.to_numeric(swift[SWIFT_RATE_COL], errors="coerce")

if SWIFT_ERROR_COL in swift.columns:
    swift[SWIFT_ERROR_COL] = pd.to_numeric(swift[SWIFT_ERROR_COL], errors="coerce")

swift = swift.dropna(subset=[SWIFT_TIME_COL, SWIFT_RATE_COL]).sort_values(SWIFT_TIME_COL).reset_index(drop=True)

# IMPORTANT: Your Swift TIME values (e.g., 53415) are NOT MJD — they are truncated MJD.
# Swift "TIME" here is days since MJD=0 but missing the 40000 offset (common in some exports).
# Convert to true MJD so it matches ESO MJD_UTC (~608xx):
# 53415 -> 53415 + 40000 = 93415? (too large) OR 53415 is actually MJD itself? (too small)
# For Swift BAT monitor files, values around 53415 correspond to MJD 53415 (year 2005).
# ESO is around 60812 (year 2025).
# So we DO NOT add any offset; we simply plot Swift in true MJD and crop to ESO range.
# If your Swift file has FULL mission range including 608xx, this will work immediately.
# If it doesn't, the script will clearly tell you the Swift TIME range.

swift_time_min = float(swift[SWIFT_TIME_COL].min())
swift_time_max = float(swift[SWIFT_TIME_COL].max())


# ---------------- LOAD ESO ----------------
eso = pd.read_csv(ESO_CSV)

for c in (COL_BLOCK, COL_MJD, COL_EXPT):
    if c not in eso.columns:
        raise ValueError(f"ESO CSV missing required column '{c}'. Found: {list(eso.columns)}")

eso[COL_BLOCK] = pd.to_numeric(eso[COL_BLOCK], errors="coerce")
eso[COL_MJD]   = pd.to_numeric(eso[COL_MJD], errors="coerce")
eso[COL_EXPT]  = pd.to_numeric(eso[COL_EXPT], errors="coerce")

eso = eso.dropna(subset=[COL_BLOCK, COL_MJD, COL_EXPT]).copy()
eso = eso.sort_values(COL_BLOCK).reset_index(drop=True)

# Build observing windows (width scales with EXPTIME)
eso["START_MJD"] = eso["MJD_UTC"]
eso["END_MJD"]   = eso["MJD_UTC"] + (eso["EXPTIME"] / 86400.0)


xmin = float(eso["START_MJD"].min() - PAD_DAYS)
xmax = float(eso["END_MJD"].max() + PAD_DAYS)


# ---------------- TRIM SWIFT TO ESO WINDOW ----------------
swift_win = swift[(swift[SWIFT_TIME_COL] >= xmin) & (swift[SWIFT_TIME_COL] <= xmax)].copy()

if swift_win.empty:
    raise ValueError(
        "No Swift points found in the ESO time window.\n"
        f"ESO window:  {xmin:.3f} to {xmax:.3f}\n"
        f"Swift TIME:  {swift_time_min:.3f} to {swift_time_max:.3f}\n\n"
        "This means your Swift CSV likely does NOT include the 2025 MJD range (~60812–60938).\n"
        "Fix: re-download GX339-4.lc.txt from Swift and convert without truncating the mission, OR\n"
        "ensure your CSV contains the late-mission rows around MJD ~608xx."
    )

# ---------------- BAT BASELINE & SIGMA LEVELS ----------------
# Use long-term Swift BAT statistics (over the plotted window)
bat_mean = swift_win[SWIFT_RATE_COL].mean()
bat_std  = swift_win[SWIFT_RATE_COL].std()

print(f"BAT mean rate   = {bat_mean:.5e}")
print(f"BAT 1σ scatter  = {bat_std:.5e}")

# ---------------- PLOT ----------------
fig, ax = plt.subplots(figsize=FIGSIZE)

# === Swift BAT: red, connected ===
has_err = (SWIFT_ERROR_COL in swift_win.columns) and swift_win[SWIFT_ERROR_COL].notna().any()

if SHOW_ERRORBARS and has_err:
    ax.errorbar(
        swift_win[SWIFT_TIME_COL],
        swift_win[SWIFT_RATE_COL],
        yerr=swift_win[SWIFT_ERROR_COL],
        fmt="o",
        markersize=3,
        linewidth=1.2,
        color="red",
        ecolor="red",
        capsize=0,
        label="Swift BAT (15–50 keV)"
    )
else:
    ax.plot(
        swift_win[SWIFT_TIME_COL],
        swift_win[SWIFT_RATE_COL],
        "-o",
        markersize=3,
        linewidth=1.2,
        color="red",
        label="Swift BAT (15–50 keV)"
    )

# === σ reference lines ===
ax.axhline(bat_mean, color="darkred", linewidth=1.4, label="BAT mean")

ax.axhline(
    bat_mean + bat_std,
    color="darkred",
    linestyle="--",
    linewidth=1.0,
    label="±1σ"
)
ax.axhline(
    bat_mean - bat_std,
    color="darkred",
    linestyle="--",
    linewidth=1.0
)

ax.axhline(
    bat_mean + 3 * bat_std,
    color="darkred",
    linestyle=":",
    linewidth=1.4,
    label="±3σ"
)
ax.axhline(
    bat_mean - 3 * bat_std,
    color="darkred",
    linestyle=":",
    linewidth=1.4
)

# Axes labels and limits
ax.set_xlabel("MJD")
ax.set_ylabel("Swift BAT Rate (counts cm$^{-2}$ s$^{-1}$)")
ax.set_title("GX 339-4: Swift BAT Light Curve with ESO Observing Blocks")
ax.set_xlim(xmin, xmax)

# Label placement height
ymin, ymax = ax.get_ylim()
yr = ymax - ymin
label_y = ymax - 0.06 * yr

# === ESO observing blocks (black) ===
for _, row in eso.iterrows():
    block = int(row[COL_BLOCK])
    start = row["START_MJD"]
    end   = row["END_MJD"]
    mid   = row[COL_MJD]

    ax.axvspan(start, end, color="black", alpha=0.20)
    ax.axvline(start, linestyle="--", linewidth=DASHED_LW, color="black")
    ax.axvline(end,   linestyle="--", linewidth=DASHED_LW, color="black")

    ax.text(
        mid, label_y, str(block),
        ha="center", va="top",
        fontsize=LABEL_FONTSIZE,
        fontweight="bold",
        color="black"
    )

# Final touches
ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
plt.show()

print(f"Saved: {os.path.abspath(OUT_PNG)}")
