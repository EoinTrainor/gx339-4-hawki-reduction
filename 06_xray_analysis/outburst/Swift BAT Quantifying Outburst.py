# title: Swift BAT strict quiescence (1σ high-side clip) + 5 plots + neighbour-aware block states (display only)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# ---------------- USER INPUTS ----------------
SWIFT_CSV = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/3) Outburst/Data & Analysis/GX339-4 Lightcurve/GX339-4 Lightcurve.csv"
ESO_CSV   = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"

COL_BLOCK = "Observing Block"
COL_MJD   = "MJD_UTC"
COL_EXPT  = "EXPTIME"   # seconds

SWIFT_TIME_COL  = "TIME"    # MJD (days)
SWIFT_RATE_COL  = "RATE"
SWIFT_ERROR_COL = "ERROR"   # optional
SWIFT_FLAG_COL  = "FLAG"    # optional

PAD_DAYS = 2.0
SHOW_ERRORBARS = True
DASHED_LW = 1.2
LABEL_FONTSIZE = 10
FIGSIZE = (13, 6)

# ---------------- STRICT QUIESCENCE DEFINITION ----------------
CLIP_SIGMA = 1.0         # strict 1σ cutoff (high-side only)
MAX_ITERS  = 50

# ---------------- NEIGHBOUR-AWARE STATE SETTINGS ----------------
DELTA_NEIGH_DAYS = 2.0   # neighbourhood half-width for coherence (± days)
BLOCK_PAD_DAYS   = 0.5   # include BAT points within ±12h of each observing block

# Diagnostic sigma lines to draw on lightcurves
SIGMAS_LINES = [1, 2, 3]

# Thresholds for neighbour-aware classification
Z_ELEV   = 1.0
Z_STRONG = 2.0
Z_OUT    = 3.0

# Point-level support criteria
F_MIN_OUT_A      = 0.50   # if z>=3: at least 50% of neighbours are elevated (>=1σ)
F2_MIN_OUT_B     = 0.33   # if z>=2 and zbar>=2: at least 33% of neighbours are strong (>=2σ)
ZBAR_MIN_OUT_B   = 2.0

# Block-level classification criteria
MIN_COUNT_OUTBURST_SUPPORTED = 2     # need >=2 supported points in the block window
MIN_COUNT_ELEV_POINTS        = 2     # elevated if >=2 points >=1σ (if not outburst)

S_OUT_BLOCK  = 2.0   # outburst if Smax>=2.0 AND at least 2 points >=2σ
S_ELEV_BLOCK = 1.0   # elevated if Smax>=1.0 (if not outburst)

# Optional BAT quality filtering
FILTER_FLAGGED = False
GOOD_FLAG_VALUES = [0]

# ---------------- HELPERS ----------------
def high_side_sigma_clip(y, sigma=1.0, maxiters=50):
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y)
    for _ in range(maxiters):
        yk = y[keep]
        if yk.size < 10:
            break
        mu = np.mean(yk)
        sd = np.std(yk, ddof=1)
        if (not np.isfinite(sd)) or sd == 0:
            break
        thresh = mu + sigma * sd
        new_keep = keep & (y <= thresh)
        if new_keep.sum() == keep.sum():
            keep = new_keep
            break
        keep = new_keep
    return keep

def compute_local_metrics(t, z, delta_days):
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)
    zbar = np.full_like(z, np.nan, dtype=float)
    f1 = np.full_like(z, np.nan, dtype=float)
    f2 = np.full_like(z, np.nan, dtype=float)
    n = np.zeros_like(z, dtype=int)

    for i in range(len(z)):
        idx = (t >= t[i] - delta_days) & (t <= t[i] + delta_days)
        zi = z[idx]
        n[i] = zi.size
        if zi.size == 0:
            continue
        zbar[i] = float(np.mean(zi))
        f1[i] = float(np.mean(zi >= Z_ELEV))
        f2[i] = float(np.mean(zi >= Z_STRONG))
    return zbar, f1, f2, n

def classify_points(z, zbar, f1, f2):
    z = np.asarray(z, dtype=float)
    zbar = np.asarray(zbar, dtype=float)
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    critA = (z >= Z_OUT) & (f1 >= F_MIN_OUT_A)
    critB = (z >= Z_STRONG) & (zbar >= ZBAR_MIN_OUT_B) & (f2 >= F2_MIN_OUT_B)
    return critA | critB

def classify_block(block_df):
    if len(block_df) == 0:
        return "No BAT data", {}
    z = block_df["z"].to_numpy()
    S = block_df["S"].to_numpy()
    supported = block_df["supported_outburst"].to_numpy()

    n_supported = int(np.sum(supported))
    n_elev = int(np.sum(z >= Z_ELEV))
    n_strong = int(np.sum(z >= Z_STRONG))

    Smax = float(np.nanmax(S))
    zmax = float(np.nanmax(z))

    outburst = (n_supported >= MIN_COUNT_OUTBURST_SUPPORTED) or ((Smax >= S_OUT_BLOCK) and (n_strong >= 2))
    if outburst:
        label = "Outburst"
    else:
        elevated = (Smax >= S_ELEV_BLOCK) or (n_elev >= MIN_COUNT_ELEV_POINTS)
        label = "Elevated" if elevated else "Quiescent"

    stats = {
        "n_points": int(len(block_df)),
        "zmax": zmax,
        "Smax": Smax,
        "n_ge_1sigma": n_elev,
        "n_ge_2sigma": n_strong,
        "n_supported_outburst": n_supported,
    }
    return label, stats

def plot_sigma_categories(ax, t, y, z, mu_q, sd_q, sigmas_lines, title):
    """
    Plot points in sigma categories like Plot 4:
      <1σ, 1–2σ, 2–3σ, ≥3σ
    """
    z = np.asarray(z, dtype=float)
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    mask_q  = z < 1.0
    mask_12 = (z >= 1.0) & (z < 2.0)
    mask_23 = (z >= 2.0) & (z < 3.0)
    mask_3p = z >= 3.0

    ax.plot(t[mask_q],  y[mask_q],  marker=".", linestyle="", alpha=0.55, label="< 1σ (quiescent)")
    ax.plot(t[mask_12], y[mask_12], marker=".", linestyle="", alpha=0.75, label="1–2σ (elevated)")
    ax.plot(t[mask_23], y[mask_23], marker=".", linestyle="", alpha=0.85, label="2–3σ (strong)")
    ax.plot(t[mask_3p], y[mask_3p], marker="o", linestyle="", markersize=4, alpha=0.95, label="≥ 3σ (candidate outburst)")

    ax.axhline(mu_q, linewidth=1.4, label="μ (quiescence)")
    for s in sigmas_lines:
        ax.axhline(mu_q + s * sd_q, linestyle="--", linewidth=1.1, label=f"μ + {s}σ" if s == 1 else None)

    ax.set_title(title)
    ax.set_xlabel("MJD")
    ax.set_ylabel("Swift BAT RATE (count/cm$^2$/s)")
    ax.grid(True, alpha=0.25)

# ---------------- LOAD SWIFT ----------------
swift = pd.read_csv(SWIFT_CSV, comment="#")
for c in (SWIFT_TIME_COL, SWIFT_RATE_COL):
    if c not in swift.columns:
        raise ValueError(f"Swift CSV missing required column '{c}'. Found: {list(swift.columns)}")

swift[SWIFT_TIME_COL] = pd.to_numeric(swift[SWIFT_TIME_COL], errors="coerce")
swift[SWIFT_RATE_COL] = pd.to_numeric(swift[SWIFT_RATE_COL], errors="coerce")
if SWIFT_ERROR_COL in swift.columns:
    swift[SWIFT_ERROR_COL] = pd.to_numeric(swift[SWIFT_ERROR_COL], errors="coerce")
if SWIFT_FLAG_COL in swift.columns:
    swift[SWIFT_FLAG_COL] = pd.to_numeric(swift[SWIFT_FLAG_COL], errors="coerce")

swift = swift.dropna(subset=[SWIFT_TIME_COL, SWIFT_RATE_COL]).sort_values(SWIFT_TIME_COL).reset_index(drop=True)
if FILTER_FLAGGED and (SWIFT_FLAG_COL in swift.columns):
    swift = swift[swift[SWIFT_FLAG_COL].isin(GOOD_FLAG_VALUES)].copy()
    swift = swift.sort_values(SWIFT_TIME_COL).reset_index(drop=True)

t_all = swift[SWIFT_TIME_COL].to_numpy(dtype=float)
y_all = swift[SWIFT_RATE_COL].to_numpy(dtype=float)

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
eso["START_MJD"] = eso[COL_MJD]
eso["END_MJD"]   = eso[COL_MJD] + (eso[COL_EXPT] / 86400.0)

xmin = float(eso["START_MJD"].min() - PAD_DAYS)
xmax = float(eso["END_MJD"].max() + PAD_DAYS)

# ---------------- DEFINE QUIESCENCE (STRICT 1σ HIGH-SIDE CLIP ON FULL BAT) ----------------
keep_q = high_side_sigma_clip(y_all, sigma=CLIP_SIGMA, maxiters=MAX_ITERS)
y_q = y_all[keep_q]
mu_q = float(np.mean(y_q))
sd_q = float(np.std(y_q, ddof=1))

print("\n--- Strict quiescence baseline from full BAT (1σ high-side clip) ---")
print(f"Total BAT points: {len(y_all)}")
print(f"Quiescent kept:   {keep_q.sum()} ({keep_q.sum()/len(y_all):.3f})")
print(f"mu_q = {mu_q:.6e}")
print(f"sd_q = {sd_q:.6e}")
print(f"Clipping: high-side only, sigma={CLIP_SIGMA}, maxiters={MAX_ITERS}\n")

# ---------------- NEIGHBOUR METRICS + SUPPORT FLAGS ----------------
swift["z"] = (swift[SWIFT_RATE_COL] - mu_q) / sd_q
z_all = swift["z"].to_numpy(dtype=float)

zbar, f1, f2, nnei = compute_local_metrics(t_all, z_all, delta_days=DELTA_NEIGH_DAYS)
swift["zbar"] = zbar
swift["f1"] = f1
swift["f2"] = f2
swift["nnei"] = nnei
swift["S"] = swift["zbar"] * swift["f1"]
swift["supported_outburst"] = classify_points(swift["z"].to_numpy(), swift["zbar"].to_numpy(), swift["f1"].to_numpy(), swift["f2"].to_numpy())

# ---------------- BLOCK CLASSIFICATION ----------------
rows = []
for _, row in eso.iterrows():
    block = int(row[COL_BLOCK])
    start = float(row["START_MJD"]) - BLOCK_PAD_DAYS
    end   = float(row["END_MJD"]) + BLOCK_PAD_DAYS
    block_swift = swift[(swift[SWIFT_TIME_COL] >= start) & (swift[SWIFT_TIME_COL] <= end)].copy()
    label, stats = classify_block(block_swift)
    out = {"Observing Block": block, "BAT_window_start_MJD": start, "BAT_window_end_MJD": end, "State": label}
    out.update(stats)
    rows.append(out)

block_summary = pd.DataFrame(rows).sort_values("Observing Block").reset_index(drop=True)
print("\n--- ESO Observing Block State Classification (Neighbour-Aware) ---")
print(block_summary.to_string(index=False))
block_to_state = {int(r["Observing Block"]): r["State"] for _, r in block_summary.iterrows()}

# ---------------- PLOT 1: HISTOGRAM ----------------
mu_fit, sd_fit = norm.fit(y_q)

plt.figure(figsize=(9, 5))
plt.hist(y_q, bins=70, density=True)
xs = np.linspace(np.min(y_q), np.max(y_q), 600)
plt.plot(xs, norm.pdf(xs, mu_fit, sd_fit))
plt.axvline(mu_q, linewidth=1.4, label="μ (quiescence)")
for s in [1, 2]:
    plt.axvline(mu_q + s * sd_q, linestyle="--", linewidth=1.1, label=f"μ ± {s}σ" if s == 1 else None)
    plt.axvline(mu_q - s * sd_q, linestyle="--", linewidth=1.1)
plt.title("Plot 1: Swift BAT Quiescent Distribution (Strict 1σ High-Side Clipping)")
plt.xlabel("RATE (count/cm$^2$/s)")
plt.ylabel("Density")
plt.grid(True, alpha=0.25)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- PLOT 2: ESO WINDOW + BLOCK STATES ----------------
swift_win = swift[(swift[SWIFT_TIME_COL] >= xmin) & (swift[SWIFT_TIME_COL] <= xmax)].copy()
if swift_win.empty:
    raise ValueError(
        "No Swift BAT points found in the ESO time window.\n"
        f"ESO window:  {xmin:.3f} to {xmax:.3f}\n"
        f"Swift TIME:  {float(swift[SWIFT_TIME_COL].min()):.3f} to {float(swift[SWIFT_TIME_COL].max()):.3f}\n\n"
        "Likely cause: your Swift CSV does not include the 2025 rows (~608xx MJD)."
    )

fig, ax = plt.subplots(figsize=FIGSIZE)
has_err = (SWIFT_ERROR_COL in swift_win.columns) and swift_win[SWIFT_ERROR_COL].notna().any()
if SHOW_ERRORBARS and has_err:
    ax.errorbar(swift_win[SWIFT_TIME_COL], swift_win[SWIFT_RATE_COL], yerr=swift_win[SWIFT_ERROR_COL],
                fmt="o", markersize=3, linewidth=1.1, label="BAT points")
else:
    ax.plot(swift_win[SWIFT_TIME_COL], swift_win[SWIFT_RATE_COL], "-o", markersize=3, linewidth=1.1, label="BAT points")

ax.axhline(mu_q, linewidth=1.4, label="μ (quiescence)")
for s in SIGMAS_LINES:
    ax.axhline(mu_q + s * sd_q, linestyle="--", linewidth=1.1, label=f"μ + {s}σ" if s == 1 else None)

win_supp = swift_win["supported_outburst"].to_numpy(dtype=bool)
ax.plot(swift_win.loc[win_supp, SWIFT_TIME_COL], swift_win.loc[win_supp, SWIFT_RATE_COL],
        marker="o", linestyle="", markersize=6, label="Neighbour-supported outburst pts")

ax.set_xlabel("MJD")
ax.set_ylabel("Swift BAT RATE (count/cm$^2$/s)")
ax.set_title("Plot 2: Swift BAT in ESO Observing Window (Strict Baseline + Block States)")
ax.set_xlim(xmin, xmax)

state_to_alpha_color = {
    "Quiescent": ("grey", 0.18),
    "Elevated": ("orange", 0.18),
    "Outburst": ("red", 0.18),
    "No BAT data": ("grey", 0.08)
}

ymin, ymax = ax.get_ylim()
label_y = ymax - 0.06 * (ymax - ymin)

for _, row in eso.iterrows():
    block = int(row[COL_BLOCK])
    start0 = float(row["START_MJD"])
    end0   = float(row["END_MJD"])
    mid    = float(row[COL_MJD])

    state = block_to_state.get(block, "No BAT data")
    color, alpha = state_to_alpha_color.get(state, ("grey", 0.18))

    ax.axvspan(start0, end0, color=color, alpha=alpha)
    ax.axvline(start0, linestyle="--", linewidth=DASHED_LW, color="black")
    ax.axvline(end0,   linestyle="--", linewidth=DASHED_LW, color="black")

    ax.text(mid, label_y, f"{block}\n{state}", ha="center", va="top",
            fontsize=LABEL_FONTSIZE, fontweight="bold", color="black")

ax.grid(True, alpha=0.25)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---------------- PLOT 3: FULL BAT + σ THRESHOLDS + NEIGHBOUR SUPPORT ----------------
plt.figure(figsize=FIGSIZE)
plt.plot(t_all, y_all, marker=".", linestyle="", alpha=0.55, label="BAT points")
plt.plot(t_all[~keep_q], y_all[~keep_q], marker=".", linestyle="", alpha=0.95, label="Clipped (non-quiescent)")
supp = swift["supported_outburst"].to_numpy(dtype=bool)
plt.plot(t_all[supp], y_all[supp], marker="o", linestyle="", markersize=5, label="Neighbour-supported outburst pts")
plt.axhline(mu_q, linewidth=1.4, label="μ (quiescence)")
for s in SIGMAS_LINES:
    plt.axhline(mu_q + s * sd_q, linestyle="--", linewidth=1.1, label=f"μ + {s}σ" if s == 1 else None)
plt.title("Plot 3: Swift BAT Full Lightcurve with Quiescence σ Thresholds + Neighbour Support")
plt.xlabel("MJD")
plt.ylabel("Swift BAT RATE (count/cm$^2$/s)")
plt.grid(True, alpha=0.25)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---------------- PLOT 4: FULL BAT CATEGORISED BY σ LEVEL ----------------
fig, ax = plt.subplots(figsize=FIGSIZE)
plot_sigma_categories(
    ax=ax,
    t=t_all,
    y=y_all,
    z=swift["z"].to_numpy(dtype=float),
    mu_q=mu_q,
    sd_q=sd_q,
    sigmas_lines=SIGMAS_LINES,
    title="Plot 4: Swift BAT Full Dataset Categorised by σ Level (Strict Baseline)"
)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---------------- PLOT 5: *YOUR DATA RANGE* CATEGORISED BY σ LEVEL (SAME STYLE AS PLOT 4) ----------------
# This is exactly Plot 4 formatting but restricted to the ESO padded window [xmin, xmax].
mask_range = (t_all >= xmin) & (t_all <= xmax)
t_range = t_all[mask_range]
y_range = y_all[mask_range]
z_range = swift.loc[mask_range, "z"].to_numpy(dtype=float)

fig, ax = plt.subplots(figsize=FIGSIZE)
plot_sigma_categories(
    ax=ax,
    t=t_range,
    y=y_range,
    z=z_range,
    mu_q=mu_q,
    sd_q=sd_q,
    sigmas_lines=SIGMAS_LINES,
    title="Plot 5: Swift BAT in ESO Time Range Categorised by σ Level (Strict Baseline)"
)
ax.set_xlim(xmin, xmax)
ax.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ---------------- NUMERIC WINDOW SUMMARY ----------------
z_win = swift_win["z"].to_numpy(dtype=float)
S_win = swift_win["S"].to_numpy(dtype=float)

print("\n--- Outburst quantification within ESO window (strict baseline + neighbour support) ---")
print(f"Max sigma excursion: {float(np.nanmax(z_win)):.2f}σ")
print(f"Max coherence score S: {float(np.nanmax(S_win)):.2f}")
print(f"Neighbour-supported outburst points in window: {int(np.sum(win_supp))}/{len(swift_win)}")
for s in [1, 2, 3]:
    print(f"Points ≥ {s}σ: {int(np.sum(z_win >= s))}/{len(z_win)}")
