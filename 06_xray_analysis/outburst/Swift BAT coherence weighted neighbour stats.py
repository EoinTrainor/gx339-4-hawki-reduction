# title: Swift BAT coherence weighted neighbour stats (time + amplitude) WITH uncertainties + ESO observing blocks (true windows) + non-overlapping OB label boxes

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from scipy.stats import norm


# ============================== USER INPUTS ==============================
SWIFT_CSV = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/3) Outburst/Data & Analysis/GX339-4 Lightcurve/GX339-4 Lightcurve.csv"
ESO_CSV   = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"

# ESO column names
COL_BLOCK = "Observing Block"
COL_MJD   = "MJD_UTC"
COL_EXPT  = "EXPTIME"  # seconds

# Swift BAT column names
SWIFT_TIME_COL  = "TIME"   # MJD days
SWIFT_RATE_COL  = "RATE"
SWIFT_ERROR_COL = "ERROR"  # optional but strongly recommended
SWIFT_FLAG_COL  = "FLAG"   # optional

# Optional BAT quality filtering (only if FLAG column exists and you want it)
FILTER_FLAGGED = False
GOOD_FLAG_VALUES = [0]

# ============================== QUIESCENCE MODEL ==============================
CLIP_SIGMA = 1.0
MAX_ITERS  = 50

# ============================== COHERENCE WEIGHTING MODEL ==============================
DELTA_NEIGH_DAYS = 2.0
TAU_DAYS         = 2.0
SIGMA_C          = 1.0
TIME_KERNEL      = "exp"     # "exp" or "gauss"
INCLUDE_SELF     = True

# Block padding (days): applied to block start/end when scoring the block evidence
BLOCK_PAD_DAYS   = 0.5

# Evidence thresholds (tune later)
ELEV_EPS = 3.0
OUT_EPS  = 6.0

# log safety
P_FLOOR = 1e-300

# Plot window around ESO span
PAD_DAYS = 2.0


# ============================== HELPERS ==============================
def high_side_sigma_clip(y, sigma=1.0, maxiters=50):
    """
    Iteratively remove points above mu + sigma*sd (high side only).
    Returns boolean mask for retained points.
    """
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(y)

    for _ in range(maxiters):
        yk = y[keep]
        if yk.size < 10:
            break

        mu = float(np.mean(yk))
        sd = float(np.std(yk, ddof=1))
        if (not np.isfinite(sd)) or sd <= 0:
            break

        thresh = mu + sigma * sd
        new_keep = keep & (y <= thresh)

        if new_keep.sum() == keep.sum():
            keep = new_keep
            break

        keep = new_keep

    return keep


def compute_z_scores_with_uncertainty(rate, mu_q, sd_q, err=None):
    """
    If err is provided, use:
        z_i = (rate_i - mu_q) / sqrt(sd_q^2 + err_i^2)
    This makes a point with large measurement uncertainty less significant.
    If err missing, fall back to:
        z_i = (rate_i - mu_q) / sd_q
    """
    rate = np.asarray(rate, dtype=float)
    if err is None:
        return (rate - mu_q) / sd_q

    err = np.asarray(err, dtype=float)
    denom = np.sqrt(sd_q**2 + np.where(np.isfinite(err), err, np.nan)**2)
    return (rate - mu_q) / denom


def compute_neighbour_supported_stats_coherent(
    t,
    z,
    delta_days: float,
    tau_days: float = 2.0,
    sigma_c: float = 1.0,
    time_kernel: str = "exp",   # "exp" or "gauss"
    include_self: bool = True,
    p_floor: float = 1e-300
):
    """
    Coherence-weighted neighbour statistic.

    Neighbourhood N(i): |t_j - t_i| <= delta_days.

    Weights:
      w_time = exp(-|dt|/tau) OR exp(-(dt^2)/(2 tau^2))
      w_amp  = exp(-(z_j - z_i)^2/(2 sigma_c^2))
      w      = w_time * w_amp

    Statistic:
      Zcoh_i = (sum_j w_j z_j) / sqrt(sum_j w_j^2)

    Returns:
      nnei : number of neighbours in window
      neff : effective neighbour count = (sum w)^2 / (sum w^2)
      Zcoh : coherence-weighted combined Z
      p    : one-sided p (elevated)
      E    : -log10(p)
    """
    t = np.asarray(t, dtype=float)
    z = np.asarray(z, dtype=float)

    n = len(z)
    nnei = np.zeros(n, dtype=int)
    neff = np.full(n, np.nan, dtype=float)
    Zcoh = np.full(n, np.nan, dtype=float)

    finite_mask = np.isfinite(t) & np.isfinite(z)

    tau_days = float(tau_days)
    sigma_c = float(sigma_c)
    delta_days = float(delta_days)
    if tau_days <= 0:
        raise ValueError("tau_days must be > 0")
    if sigma_c <= 0:
        raise ValueError("sigma_c must be > 0")
    if delta_days <= 0:
        raise ValueError("delta_days must be > 0")

    kernel = time_kernel.lower().strip()
    if kernel not in ("exp", "gauss"):
        raise ValueError("time_kernel must be 'exp' or 'gauss'")

    for i in range(n):
        if not finite_mask[i]:
            continue

        dt = t - t[i]
        idx = finite_mask & (np.abs(dt) <= delta_days)
        if not include_self:
            idx[i] = False

        zi = z[i]
        zj = z[idx]
        dtj = dt[idx]

        nnei_i = int(zj.size)
        nnei[i] = nnei_i
        if nnei_i == 0:
            continue

        # time weights
        if kernel == "exp":
            w_time = np.exp(-np.abs(dtj) / tau_days)
        else:
            w_time = np.exp(-(dtj ** 2) / (2.0 * tau_days ** 2))

        # amplitude similarity weights
        w_amp = np.exp(-((zj - zi) ** 2) / (2.0 * sigma_c ** 2))

        w = w_time * w_amp
        w2_sum = float(np.sum(w ** 2))
        if (not np.isfinite(w2_sum)) or w2_sum <= 0:
            continue

        num = float(np.sum(w * zj))
        Zcoh[i] = num / np.sqrt(w2_sum)

        w_sum = float(np.sum(w))
        if np.isfinite(w_sum) and w_sum > 0:
            neff[i] = (w_sum ** 2) / w2_sum

    p = 1.0 - norm.cdf(Zcoh)
    p = np.clip(p, p_floor, 1.0)
    E = -np.log10(p)

    return nnei, neff, Zcoh, p, E


def point_state_from_E(E, elev_eps=ELEV_EPS, out_eps=OUT_EPS):
    if not np.isfinite(E):
        return "No data"
    if E >= out_eps:
        return "Outburst"
    if E >= elev_eps:
        return "Elevated"
    return "Quiescent"


def classify_block_from_epsilon(epsilon_block, elev_eps=ELEV_EPS, out_eps=OUT_EPS):
    if not np.isfinite(epsilon_block):
        return "No BAT data"
    if epsilon_block >= out_eps:
        return "Outburst"
    if epsilon_block >= elev_eps:
        return "Elevated"
    return "Quiescent"


def annotate_sigma_lines(ax, mu_q, sd_q, sigmas=(1, 2, 3), x_frac=0.995):
    """
    Draw and label mu_q and mu_q + k*sd_q.
    """
    x0, x1 = ax.get_xlim()
    x_text = x0 + x_frac * (x1 - x0)

    ax.axhline(mu_q, linestyle="-", linewidth=1.4, alpha=0.85)
    ax.text(
        x_text, mu_q, "μq",
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0)
    )

    for k in sigmas:
        y = mu_q + k * sd_q
        ax.axhline(y, linestyle="--", linewidth=1.1, alpha=0.65)
        ax.text(
            x_text, y, f"μq + {k}σq",
            ha="right", va="bottom",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0)
        )


def assign_overlap_lanes(intervals):
    """
    Greedy lane assignment so label boxes do not overlap in x.

    intervals: list of dicts with keys: start, end, and any metadata.
    Returns: list of lane indices aligned with intervals order.
    """
    # Sort by start, then end
    order = np.argsort([(iv["start"], iv["end"]) for iv in intervals], axis=0)[:, 0] \
        if len(intervals) > 1 else np.array([0])

    # lane_end[lane] = last end in that lane
    lane_end = []
    lanes = [None] * len(intervals)

    for idx in order:
        s = intervals[idx]["start"]
        e = intervals[idx]["end"]

        placed = False
        for lane, last_end in enumerate(lane_end):
            if s >= last_end:  # no overlap with last interval in lane
                lanes[idx] = lane
                lane_end[lane] = e
                placed = True
                break

        if not placed:
            lanes[idx] = len(lane_end)
            lane_end.append(e)

    return lanes, len(lane_end)


def shade_and_label_blocks(ax, block_stats, show_eps=True):
    """
    Shade each OB span (true start/end), and label each OB.
    Label boxes are placed on different heights (lanes) when OB windows overlap in time.
    """
    intervals = []
    for _, r in block_stats.iterrows():
        intervals.append({
            "start": float(r["block_start"]),
            "end": float(r["block_end"]),
            "block": int(r[COL_BLOCK]),
            "state": r["state"],
            "eps": r.get("epsilon_block", np.nan)
        })

    lanes, nlanes = assign_overlap_lanes(intervals)

    # Build lane y positions (axes fraction)
    # top lane near 0.985, next down by step; clamp so it doesn't go off plot
    y_top = 0.985
    step = 0.085
    y_positions = [max(0.55, y_top - lane * step) for lane in range(nlanes)]

    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    state_to_alpha = {"Quiescent": 0.10, "Elevated": 0.18, "Outburst": 0.28, "No BAT data": 0.08}

    for iv, lane in zip(intervals, lanes):
        s = iv["state"]
        a = state_to_alpha.get(s, 0.12)
        x0, x1 = iv["start"], iv["end"]
        xm = 0.5 * (x0 + x1)

        # shaded window (the actual timeframe)
        ax.axvspan(x0, x1, alpha=a)

        # label box in its lane
        bnum = iv["block"]
        eps = iv["eps"]
        if show_eps and np.isfinite(eps):
            txt = f"OB {bnum}\n{s}\nε={eps:.2f}"
        else:
            txt = f"OB {bnum}\n{s}"

        ax.text(
            xm, y_positions[lane], txt,
            transform=trans,
            ha="center", va="top",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.20", alpha=0.20, linewidth=0)
        )


def safe_float(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan


# ============================== LOAD SWIFT BAT ==============================
bat = pd.read_csv(SWIFT_CSV)

for c in (SWIFT_TIME_COL, SWIFT_RATE_COL):
    if c not in bat.columns:
        raise ValueError(f"Swift CSV missing required column '{c}'. Found: {list(bat.columns)}")

bat[SWIFT_TIME_COL] = pd.to_numeric(bat[SWIFT_TIME_COL], errors="coerce")
bat[SWIFT_RATE_COL] = pd.to_numeric(bat[SWIFT_RATE_COL], errors="coerce")

has_err = SWIFT_ERROR_COL in bat.columns
if has_err:
    bat[SWIFT_ERROR_COL] = pd.to_numeric(bat[SWIFT_ERROR_COL], errors="coerce")

if SWIFT_FLAG_COL in bat.columns:
    bat[SWIFT_FLAG_COL] = pd.to_numeric(bat[SWIFT_FLAG_COL], errors="coerce")

bat = bat.dropna(subset=[SWIFT_TIME_COL, SWIFT_RATE_COL]).copy()
if FILTER_FLAGGED and (SWIFT_FLAG_COL in bat.columns):
    bat = bat[bat[SWIFT_FLAG_COL].isin(GOOD_FLAG_VALUES)].copy()

bat = bat.sort_values(SWIFT_TIME_COL).reset_index(drop=True)

# ============================== QUIESCENCE BASELINE ==============================
keep_q = high_side_sigma_clip(bat[SWIFT_RATE_COL].to_numpy(), sigma=CLIP_SIGMA, maxiters=MAX_ITERS)
mu_q = float(np.mean(bat.loc[keep_q, SWIFT_RATE_COL].to_numpy()))
sd_q = float(np.std(bat.loc[keep_q, SWIFT_RATE_COL].to_numpy(), ddof=1))

print("\n================= QUIESCENCE MODEL =================")
print(f"Quiescent mu_q = {mu_q:.6g}")
print(f"Quiescent sd_q = {sd_q:.6g}")
print(f"Quiescent points kept = {int(np.sum(keep_q))} / {len(bat)}")
print(f"Using measurement uncertainties in z: {has_err}")
print("===================================================\n")

# ============================== POINT METRICS (FULL LIGHTCURVE) ==============================
err_arr = bat[SWIFT_ERROR_COL].to_numpy() if has_err else None
bat["z"] = compute_z_scores_with_uncertainty(
    rate=bat[SWIFT_RATE_COL].to_numpy(),
    mu_q=mu_q,
    sd_q=sd_q,
    err=err_arr
)

nnei_all, neff_all, Zcoh_all, p_all, E_all = compute_neighbour_supported_stats_coherent(
    t=bat[SWIFT_TIME_COL].to_numpy(),
    z=bat["z"].to_numpy(),
    delta_days=DELTA_NEIGH_DAYS,
    tau_days=TAU_DAYS,
    sigma_c=SIGMA_C,
    time_kernel=TIME_KERNEL,
    include_self=INCLUDE_SELF,
    p_floor=P_FLOOR
)

bat["nnei"] = nnei_all
bat["neff"] = neff_all
bat["Zcoh"] = Zcoh_all
bat["p"] = p_all
bat["E"] = E_all
bat["point_state"] = [point_state_from_E(v, ELEV_EPS, OUT_EPS) for v in bat["E"].to_numpy()]

# ============================== LOAD ESO OBSERVING BLOCKS ==============================
eso = pd.read_csv(ESO_CSV)
for c in (COL_BLOCK, COL_MJD, COL_EXPT):
    if c not in eso.columns:
        raise ValueError(f"ESO CSV missing required column '{c}'. Found: {list(eso.columns)}")

eso[COL_BLOCK] = pd.to_numeric(eso[COL_BLOCK], errors="coerce")
eso[COL_MJD]   = pd.to_numeric(eso[COL_MJD], errors="coerce")
eso[COL_EXPT]  = pd.to_numeric(eso[COL_EXPT], errors="coerce")

eso = eso.dropna(subset=[COL_BLOCK, COL_MJD, COL_EXPT]).copy()
eso = eso.sort_values([COL_BLOCK, COL_MJD]).reset_index(drop=True)

# Each exposure is a true time window: [START_MJD, END_MJD]
eso["START_MJD"] = eso[COL_MJD]
eso["END_MJD"]   = eso[COL_MJD] + (eso[COL_EXPT] / 86400.0)

# Each OB is the union window across its exposures
blocks = (
    eso.groupby(COL_BLOCK)
       .agg(block_start=("START_MJD", "min"),
            block_end=("END_MJD", "max"),
            n_exposures=(COL_BLOCK, "size"))
       .reset_index()
       .sort_values(COL_BLOCK)
       .reset_index(drop=True)
)

xmin = float(eso["START_MJD"].min() - PAD_DAYS)
xmax = float(eso["END_MJD"].max() + PAD_DAYS)
bat_win = bat[(bat[SWIFT_TIME_COL] >= xmin) & (bat[SWIFT_TIME_COL] <= xmax)].copy()

# ============================== BLOCK SCORING + OB STATS (PADDED WINDOWS) ==============================
block_rows = []
ob_stats_rows = []

for _, row in blocks.iterrows():
    b = int(row[COL_BLOCK])
    b_start = float(row["block_start"])
    b_end   = float(row["block_end"])

    # padded block window for scoring
    w0 = b_start - BLOCK_PAD_DAYS
    w1 = b_end   + BLOCK_PAD_DAYS
    sub = bat[(bat[SWIFT_TIME_COL] >= w0) & (bat[SWIFT_TIME_COL] <= w1)].copy()

    if len(sub) == 0:
        pmin = np.nan
        eps  = np.nan
        state = "No BAT data"
        n_bat = 0
        z_peak = np.nan; z_peak_time = np.nan
        Z_peak = np.nan; Z_peak_time = np.nan
        E_peak = np.nan; E_peak_time = np.nan
    else:
        pmin = float(np.nanmin(sub["p"].to_numpy()))
        pmin = max(pmin, P_FLOOR)
        eps  = float(-np.log10(pmin))
        state = classify_block_from_epsilon(eps, ELEV_EPS, OUT_EPS)
        n_bat = int(len(sub))

        zvals = sub["z"].to_numpy()
        i_z = int(np.nanargmax(zvals))
        z_peak = float(zvals[i_z])
        z_peak_time = float(sub.iloc[i_z][SWIFT_TIME_COL])

        Zvals = sub["Zcoh"].to_numpy()
        i_Z = int(np.nanargmax(Zvals))
        Z_peak = float(Zvals[i_Z])
        Z_peak_time = float(sub.iloc[i_Z][SWIFT_TIME_COL])

        Evals = sub["E"].to_numpy()
        i_E = int(np.nanargmax(Evals))
        E_peak = float(Evals[i_E])
        E_peak_time = float(sub.iloc[i_E][SWIFT_TIME_COL])

    block_rows.append({
        COL_BLOCK: b,
        "block_start": b_start,
        "block_end": b_end,
        "n_exposures": int(row["n_exposures"]),
        "n_bat_points": n_bat,
        "pmin": safe_float(pmin),
        "epsilon_block": safe_float(eps),
        "state": state,
    })

    ob_stats_rows.append({
        COL_BLOCK: b,
        "block_start": b_start,
        "block_end": b_end,
        "n_bat_points": n_bat,
        "z_peak": safe_float(z_peak),
        "z_peak_time": safe_float(z_peak_time),
        "Zcoh_peak": safe_float(Z_peak),
        "Zcoh_peak_time": safe_float(Z_peak_time),
        "E_peak": safe_float(E_peak),
        "E_peak_time": safe_float(E_peak_time),
        "pmin": safe_float(pmin),
        "epsilon_block": safe_float(eps),
        "state": state,
    })

block_stats = pd.DataFrame(block_rows).sort_values(COL_BLOCK).reset_index(drop=True)
ob_stats = pd.DataFrame(ob_stats_rows).sort_values(COL_BLOCK).reset_index(drop=True)

print("================= BLOCK CLASSIFICATION =================")
print(block_stats[[COL_BLOCK, "n_exposures", "n_bat_points", "epsilon_block", "state"]].to_string(index=False))
print("========================================================\n")

print("================= OB WINDOW Z-STATS (PADDED) =================")
print(ob_stats[[COL_BLOCK, "n_bat_points", "z_peak", "Zcoh_peak", "epsilon_block", "state"]].to_string(index=False))
print("=============================================================\n")


# ============================== PLOTS (with uncertainties + non-overlapping OB label boxes) ==============================
coh_txt = f"Δ={DELTA_NEIGH_DAYS:g} d, τ={TAU_DAYS:g} d, σc={SIGMA_C:g}, kernel={TIME_KERNEL}; z uses sqrt(σq²+err²)={has_err}"

# ---------- Plot 1: FULL LIGHTCURVE, POINT STATE (with uncertainties) ----------
fig, ax = plt.subplots(figsize=(14, 6))

# Plot error bars first (light), then state-coloured points on top
if has_err and np.isfinite(bat[SWIFT_ERROR_COL].to_numpy()).any():
    ax.errorbar(
        bat[SWIFT_TIME_COL].to_numpy(),
        bat[SWIFT_RATE_COL].to_numpy(),
        yerr=bat[SWIFT_ERROR_COL].to_numpy(),
        fmt="none",
        elinewidth=0.8,
        alpha=0.25
    )

for s in ["Quiescent", "Elevated", "Outburst"]:
    m = bat["point_state"].to_numpy() == s
    ax.scatter(
        bat.loc[m, SWIFT_TIME_COL].to_numpy(),
        bat.loc[m, SWIFT_RATE_COL].to_numpy(),
        s=12, alpha=0.80, label=f"Point state: {s}"
    )

ax.set_title("Swift BAT full lightcurve: coherence weighted neighbour supported point state (with uncertainties)")
ax.set_xlabel("MJD")
ax.set_ylabel("BAT rate")

ax.set_xlim(float(bat[SWIFT_TIME_COL].min()), float(bat[SWIFT_TIME_COL].max()))
annotate_sigma_lines(ax, mu_q, sd_q, sigmas=(1, 2, 3))

ax.text(
    0.01, 0.02, coh_txt,
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", alpha=0.15, linewidth=0)
)

ax.legend(loc="best")
plt.tight_layout()
plt.show()

# ---------- Plot 2: FULL LIGHTCURVE, E(t) ----------
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(bat[SWIFT_TIME_COL].to_numpy(), bat["E"].to_numpy(), ".", markersize=2.6, alpha=0.85)
ax.axhline(ELEV_EPS, linestyle="--", linewidth=1.1, alpha=0.7)
ax.axhline(OUT_EPS,  linestyle="--", linewidth=1.1, alpha=0.7)

x0, x1 = ax.get_xlim()
x_text = x0 + 0.995 * (x1 - x0)
ax.text(x_text, ELEV_EPS, f"ELEV_EPS={ELEV_EPS:g}", ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0))
ax.text(x_text, OUT_EPS,  f"OUT_EPS={OUT_EPS:g}",  ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0))

ax.set_title("Neighbour supported evidence per BAT point: E(t) = -log10(1 - Φ(Zcoh(t)))")
ax.set_xlabel("MJD")
ax.set_ylabel("E(t)")

ax.text(
    0.01, 0.06, coh_txt,
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", alpha=0.15, linewidth=0)
)

plt.tight_layout()
plt.show()

# ---------- Plot 3: OB TIMEFRAME, POINT STATE + TRUE OB WINDOWS (label boxes stacked) ----------
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(xmin, xmax)

# shade + label every OB; label boxes stacked to avoid overlap
shade_and_label_blocks(ax, block_stats, show_eps=True)

# Uncertainty bars (light)
if has_err and np.isfinite(bat_win[SWIFT_ERROR_COL].to_numpy()).any():
    ax.errorbar(
        bat_win[SWIFT_TIME_COL].to_numpy(),
        bat_win[SWIFT_RATE_COL].to_numpy(),
        yerr=bat_win[SWIFT_ERROR_COL].to_numpy(),
        fmt="none",
        elinewidth=0.9,
        alpha=0.30
    )

# State points on top
for s in ["Quiescent", "Elevated", "Outburst"]:
    m = bat_win["point_state"].to_numpy() == s
    ax.scatter(
        bat_win.loc[m, SWIFT_TIME_COL].to_numpy(),
        bat_win.loc[m, SWIFT_RATE_COL].to_numpy(),
        s=18, alpha=0.85, label=f"Point state: {s}"
    )

ax.set_title("Swift BAT during ESO timeframe: point state + true observing block windows (label boxes stacked)")
ax.set_xlabel("MJD")
ax.set_ylabel("BAT rate")

annotate_sigma_lines(ax, mu_q, sd_q, sigmas=(1, 2, 3))

ax.text(
    0.01, 0.02, coh_txt,
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", alpha=0.15, linewidth=0)
)

ax.legend(loc="best")
plt.tight_layout()
plt.show()

# ---------- Plot 4: epsilon_block per OB (label each) ----------
fig, ax = plt.subplots(figsize=(12, 4))

xB = block_stats[COL_BLOCK].to_numpy()
yEps = block_stats["epsilon_block"].to_numpy()
ax.plot(xB, yEps, "o", alpha=0.9)

ax.axhline(ELEV_EPS, linestyle="--", linewidth=1.1, alpha=0.7)
ax.axhline(OUT_EPS,  linestyle="--", linewidth=1.1, alpha=0.7)

x0, x1 = ax.get_xlim()
x_text = x0 + 0.995 * (x1 - x0)
ax.text(x_text, ELEV_EPS, f"ELEV_EPS={ELEV_EPS:g}", ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0))
ax.text(x_text, OUT_EPS,  f"OUT_EPS={OUT_EPS:g}",  ha="right", va="bottom",
        fontsize=9, bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0))

for xb, ye, st in zip(xB, yEps, block_stats["state"].to_numpy()):
    if np.isfinite(ye):
        ax.text(
            xb, ye, f"OB {int(xb)}\n{st}",
            ha="center", va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0)
        )

ax.set_title("ESO observing blocks: ε_block = -log10(min p(t) in padded block window)")
ax.set_xlabel("Observing Block")
ax.set_ylabel("ε_block")
plt.tight_layout()
plt.show()

# ---------- Plot 5: OB TIMEFRAME raw z(t) with labelled sigma lines ----------
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(bat_win[SWIFT_TIME_COL].to_numpy(), bat_win["z"].to_numpy(), ".", markersize=3.0, alpha=0.85)

for k in [1, 2, 3, 4]:
    ax.axhline(k, linestyle="--", linewidth=1.1, alpha=0.6)
    x0, x1 = ax.get_xlim()
    x_text = x0 + 0.995 * (x1 - x0)
    ax.text(
        x_text, k, f"{k}σ",
        ha="right", va="bottom",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.15", alpha=0.15, linewidth=0)
    )

ax.set_xlim(xmin, xmax)
ax.set_title("Swift BAT during ESO timeframe: raw z(t) (includes measurement uncertainties if available)")
ax.set_xlabel("MJD")
ax.set_ylabel("z(t)")

ax.text(
    0.01, 0.06, f"z_i = (rate_i - μq) / sqrt(σq² + err_i²)  |  err available={has_err}",
    transform=ax.transAxes,
    ha="left", va="bottom",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.25", alpha=0.15, linewidth=0)
)

plt.tight_layout()
plt.show()

# Optional saves (uncomment if you want files)
# block_stats.to_csv("eso_block_bat_epsilon_classification.csv", index=False)
# ob_stats.to_csv("eso_ob_window_bat_stats.csv", index=False)
# bat.to_csv("swift_bat_with_coherence_stats.csv", index=False)
