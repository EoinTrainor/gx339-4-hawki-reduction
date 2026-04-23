"""
10_lightcurve.py
----------------
Stage 10: Phase-fold the GX 339-4 differential lightcurve from ZOGY S statistic.

Inputs:
  lightcurve_raw.csv  — output of 09_zogy.py
  reference_stars.json

Steps:
  1. Load S_target per frame (MJD, S_target, flux_diff, F_D, ref star S values)
  2. Sigma-clip per-OB outliers (cosmic rays / bad frames)
  3. Per-OB detrending — remove OB-to-OB flux offset before phase-folding
  4. Compute orbital phase using Heida+2017 ephemeris
  5. Phase-fold and bin the lightcurve
  6. Save figures and calibrated lightcurve CSV

Ephemeris (Heida et al. 2017):
  T0 = 57529.397 MJD  (inferior conjunction of donor star)
  P  = 1.7587 d

Phase convention:
  phi=0.0  inferior conjunction  (donor closest to observer) -> ellipsoidal minimum
  phi=0.25 quadrature            (donor side-on)             -> ellipsoidal maximum
  phi=0.5  superior conjunction  (donor behind compact obj)  -> ellipsoidal minimum
  phi=0.75 quadrature                                        -> ellipsoidal maximum
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.stats import sigma_clipped_stats

import config

# ── Ephemeris ─────────────────────────────────────────────────────────────────
T0 = 57529.397   # MJD — inferior conjunction of donor (Heida+2017)
P  = 1.7587      # days

# ── Paths ─────────────────────────────────────────────────────────────────────
LC_RAW_CSV   = config.DIFF_DIR / "lightcurve_raw.csv"
REF_STARS_JSON = config.LOGS_DIR / "zogy" / "quality" / "reference_stars.json"
OUT_DIR      = config.DIFF_DIR / "lightcurve"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Settings ──────────────────────────────────────────────────────────────────
OUTLIER_SIGMA  = 3.0    # per-OB sigma clip threshold
N_PHASE_BINS   = 20     # bins for phase-folded lightcurve
MIN_BIN_PTS    = 3      # minimum points to plot a bin

OB_COLOURS = {
    "GX339_Ks_Imaging_1":  "#4C72B0",
    "GX339_Ks_Imaging_2":  "#DD8452",
    "GX339_Ks_Imaging_3":  "#55A868",
    "GX339_Ks_Imaging_4":  "#C44E52",
    "GX339_Ks_Imaging_5":  "#8172B3",
    "GX339_Ks_Imaging_6":  "#937860",
    "GX339_Ks_Imaging_7":  "#DA8BC3",
    "GX339_Ks_Imaging_8":  "#8C8C8C",
    "GX339_Ks_Imaging_9":  "#CCB974",
    "GX339_Ks_Imaging_10": "#64B5CD",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("Loading lightcurve_raw.csv ...")
rows = list(csv.DictReader(open(LC_RAW_CSV)))

mjd       = np.array([float(r["mjd"])       for r in rows])
S_target  = np.array([float(r["S_target"])  for r in rows])
flux_diff = np.array([float(r["flux_diff"]) for r in rows])
F_D       = np.array([float(r["F_D"])       for r in rows])
fwhm      = np.array([float(r["fwhm_n"])    for r in rows])
obs       = np.array([r["ob"]               for r in rows])

ref_ids = [k for k in rows[0].keys() if k.startswith("ref_")]
ref_S = {}
for rid in ref_ids:
    vals = []
    for r in rows:
        v = r.get(rid, "")
        vals.append(float(v) if v not in ("", "nan") else np.nan)
    ref_S[rid] = np.array(vals)

print(f"  {len(mjd)} frames,  {len(set(obs))} OBs,  {len(ref_ids)} reference stars")

# Load reference star catalogue info
ref_meta = {}
if REF_STARS_JSON.exists():
    for rs in json.load(open(REF_STARS_JSON)):
        ref_meta[rs["id"]] = rs


# =============================================================================
# 2. PER-OB SIGMA CLIPPING
# =============================================================================
print("Per-OB sigma clipping ...")
good = np.ones(len(mjd), dtype=bool)
unique_obs = sorted(set(obs), key=lambda o: int(o.split("_")[-1]))

for ob in unique_obs:
    mask = obs == ob
    s    = S_target[mask]
    _, med, std = sigma_clipped_stats(s, sigma=OUTLIER_SIGMA)
    outliers = np.abs(s - med) > OUTLIER_SIGMA * std
    n_clip = outliers.sum()
    if n_clip:
        idx = np.where(mask)[0][outliers]
        good[idx] = False
        print(f"  {ob[-2:]}: clipped {n_clip} frame(s)")

print(f"  Kept {good.sum()}/{len(good)} frames after clipping")

mjd_g  = mjd[good];   S_g  = S_target[good]
obs_g  = obs[good];   fwhm_g = fwhm[good]
F_D_g  = F_D[good];   flux_g = flux_diff[good]


# =============================================================================
# 3. PER-OB DETRENDING
# =============================================================================
# Subtract the per-OB mean of S_target. This removes OB-to-OB flux offsets
# (e.g. slow accretion variability, photometric calibration drift) while
# preserving intra-OB variability. Each OB spans ~52 min << P=1.76 d, so the
# ellipsoidal signal within a single OB is approximately linear and small.
print("Per-OB detrending ...")
ob_means = {}
S_detrend = S_g.copy()

for ob in unique_obs:
    mask = obs_g == ob
    ob_mean = np.mean(S_g[mask])
    ob_means[ob] = ob_mean
    S_detrend[mask] -= ob_mean

print("  Per-OB means (raw S_target):")
for ob in unique_obs:
    print(f"    {ob[-2:]}: {ob_means[ob]:.3e}")

# Normalise by global std so units are comparable
S_norm_std = np.std(S_detrend)
S_detrend_norm = S_detrend / S_norm_std
print(f"  Global std after detrending: {S_norm_std:.3e}")


# =============================================================================
# 4. ORBITAL PHASE
# =============================================================================
phase_g = ((mjd_g - T0) / P) % 1.0


# =============================================================================
# 5. PHASE-FOLDED BINNING
# =============================================================================
bin_edges  = np.linspace(0, 1, N_PHASE_BINS + 1)
bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bin_mean    = np.full(N_PHASE_BINS, np.nan)
bin_err     = np.full(N_PHASE_BINS, np.nan)
bin_n       = np.zeros(N_PHASE_BINS, dtype=int)

for i in range(N_PHASE_BINS):
    in_bin = (phase_g >= bin_edges[i]) & (phase_g < bin_edges[i + 1])
    n = in_bin.sum()
    bin_n[i] = n
    if n >= MIN_BIN_PTS:
        vals = S_detrend_norm[in_bin]
        bin_mean[i] = np.mean(vals)
        bin_err[i]  = np.std(vals) / np.sqrt(n)


# =============================================================================
# 6. FIGURES
# =============================================================================

# ── Figure 1: Raw S_target vs MJD (long-term trend) ─────────────────────────
print("Figure 1: raw lightcurve vs MJD ...")
fig, ax = plt.subplots(figsize=(13, 4))
for ob in unique_obs:
    mask = obs_g == ob
    col  = OB_COLOURS.get(ob, "grey")
    ax.scatter(mjd_g[mask], S_g[mask], color=col, s=12, alpha=0.7, zorder=3,
               label=ob.replace("GX339_Ks_Imaging_", "OB "))
    ax.axhline(ob_means[ob], color=col, lw=0.8, ls="--", alpha=0.5)

ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)
ax.set_xlabel("MJD")
ax.set_ylabel("S_target  [ZOGY matched-filter units]")
ax.set_title("GX 339-4  —  Raw differential S_target vs time  (dashed = per-OB mean)")
ax.legend(fontsize=8, ncol=5, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "lc01_raw_vs_mjd.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "lc01_raw_vs_mjd.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ── Figure 2: Reference star S values vs MJD ─────────────────────────────────
print("Figure 2: reference star S values ...")
fig, ax = plt.subplots(figsize=(13, 4))
colours_ref = plt.cm.tab10(np.linspace(0, 1, len(ref_ids)))
for rid, col in zip(ref_ids, colours_ref):
    v = ref_S[rid][good]
    ks = ref_meta.get(rid, {}).get("Kmag", "?")
    ax.scatter(mjd_g, v, color=col, s=8, alpha=0.5, zorder=3,
               label=f"{rid}  Ks={ks:.1f}" if isinstance(ks, float) else rid)

ax.axhline(0, color="black", lw=1.0, ls="-")
ax.set_xlabel("MJD")
ax.set_ylabel("S  [matched-filter units]")
ax.set_title("Reference stars — S should be ~0 for non-variable sources")
ax.legend(fontsize=8, ncol=4, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "lc02_reference_stars.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "lc02_reference_stars.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ── Figure 3: Detrended S vs MJD ─────────────────────────────────────────────
print("Figure 3: detrended lightcurve vs MJD ...")
fig, ax = plt.subplots(figsize=(13, 4))
for ob in unique_obs:
    mask = obs_g == ob
    col  = OB_COLOURS.get(ob, "grey")
    ax.scatter(mjd_g[mask], S_detrend_norm[mask], color=col, s=12, alpha=0.7,
               zorder=3, label=ob.replace("GX339_Ks_Imaging_", "OB "))
ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.4)
ax.set_xlabel("MJD")
ax.set_ylabel("Detrended S  (per-OB mean subtracted, / global std)")
ax.set_title("GX 339-4  —  Detrended differential lightcurve vs time")
ax.legend(fontsize=8, ncol=5, loc="upper left")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "lc03_detrended_vs_mjd.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "lc03_detrended_vs_mjd.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ── Figure 4: Phase-folded lightcurve ────────────────────────────────────────
print("Figure 4: phase-folded lightcurve ...")
fig = plt.figure(figsize=(11, 7))
gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Individual points (shown twice: phi and phi+1 for continuity)
for ob in unique_obs:
    mask = obs_g == ob
    col  = OB_COLOURS.get(ob, "grey")
    ph   = phase_g[mask]
    sv   = S_detrend_norm[mask]
    ax1.scatter(ph,     sv, color=col, s=10, alpha=0.4, zorder=2)
    ax1.scatter(ph + 1, sv, color=col, s=10, alpha=0.4, zorder=2)

# Binned mean (shown twice)
valid_bins = bin_n >= MIN_BIN_PTS
for offset in [0, 1]:
    ax1.errorbar(bin_centres[valid_bins] + offset,
                 bin_mean[valid_bins],
                 yerr=bin_err[valid_bins],
                 fmt="ko", ms=7, capsize=4, lw=1.5, zorder=5,
                 label="Binned mean ± SE" if offset == 0 else None)

# Mark expected ellipsoidal extrema
for phi_min in [0.0, 0.5, 1.0, 1.5]:
    ax1.axvline(phi_min, color="blue", lw=0.7, ls=":", alpha=0.6,
                label="Expected minimum (phi=0, 0.5)" if phi_min == 0.0 else None)
for phi_max in [0.25, 0.75, 1.25, 1.75]:
    ax1.axvline(phi_max, color="red", lw=0.7, ls=":", alpha=0.6,
                label="Expected maximum (phi=0.25, 0.75)" if phi_max == 0.25 else None)

ax1.axhline(0, color="black", lw=0.6, alpha=0.4)
ax1.set_ylabel("Detrended S  (normalised)")
ax1.set_title(
    f"GX 339-4 — Phase-folded differential lightcurve\n"
    f"T$_0$ = {T0} MJD (inf. conj., Heida+2017),  P = {P} d,  "
    f"{good.sum()} frames,  {N_PHASE_BINS} bins"
)
ax1.legend(fontsize=9, loc="upper right")
ax1.set_xlim(0, 2)
ax1.grid(alpha=0.2)
plt.setp(ax1.get_xticklabels(), visible=False)

# Bottom panel: points per bin
ax2.bar(bin_centres, bin_n, width=1/N_PHASE_BINS * 0.85,
        color="steelblue", alpha=0.7, align="center")
ax2.bar(bin_centres + 1, bin_n, width=1/N_PHASE_BINS * 0.85,
        color="steelblue", alpha=0.7, align="center")
ax2.axhline(MIN_BIN_PTS, color="red", lw=0.8, ls="--", label=f"Min pts ({MIN_BIN_PTS})")
ax2.set_xlabel("Orbital phase  (phi=0: inferior conjunction of donor)")
ax2.set_ylabel("N per bin")
ax2.set_xlim(0, 2)
ax2.grid(alpha=0.2)

fig.savefig(OUT_DIR / "lc04_phase_folded.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "lc04_phase_folded.png", bbox_inches="tight", dpi=150)
plt.close(fig)

# ── Figure 5: Phase-folded (single cycle, cleaner for paper) ─────────────────
print("Figure 5: single-cycle phase-folded (report quality) ...")
fig, ax = plt.subplots(figsize=(8, 5))
for ob in unique_obs:
    mask = obs_g == ob
    col  = OB_COLOURS.get(ob, "grey")
    ax.scatter(phase_g[mask], S_detrend_norm[mask],
               color=col, s=14, alpha=0.5, zorder=2)

ax.errorbar(bin_centres[valid_bins], bin_mean[valid_bins],
            yerr=bin_err[valid_bins],
            fmt="ko", ms=8, capsize=4, lw=1.8, zorder=5, label="Binned mean ± SE")

ax.axvline(0.0,  color="blue", lw=0.8, ls=":", alpha=0.7, label="Inferior conjunction (phi=0)")
ax.axvline(0.5,  color="blue", lw=0.8, ls=":", alpha=0.7)
ax.axvline(0.25, color="red",  lw=0.8, ls=":", alpha=0.7, label="Quadrature (phi=0.25, 0.75)")
ax.axvline(0.75, color="red",  lw=0.8, ls=":", alpha=0.7)
ax.axhline(0,    color="black", lw=0.6, alpha=0.4)

ax.set_xlabel(r"Orbital phase  ($\phi=0$: inferior conjunction of donor)")
ax.set_ylabel("Detrended $S$ (normalised)")
ax.set_title(
    "GX 339-4 Ks-band ellipsoidal lightcurve\n"
    f"P = {P} d,  T$_0$ = {T0} MJD (Heida+2017)"
)
ax.set_xlim(0, 1)
ax.legend(fontsize=9)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT_DIR / "lc05_phase_folded_paper.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "lc05_phase_folded_paper.png", bbox_inches="tight", dpi=150)
plt.close(fig)


# =============================================================================
# 7. OUTPUT CSV
# =============================================================================
print("Writing lightcurve_phasefolded.csv ...")
lc_out = OUT_DIR / "lightcurve_phasefolded.csv"
with open(lc_out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mjd", "ob", "phase", "S_target", "S_detrend", "S_detrend_norm",
                "flux_diff", "F_D", "fwhm_n", "good"])
    for i in range(len(mjd)):
        ph = ((mjd[i] - T0) / P) % 1.0
        ob_mean_i = ob_means.get(obs[i], 0.0)
        s_det_i   = (S_target[i] - ob_mean_i) / S_norm_std if good[i] else np.nan
        w.writerow([
            f"{mjd[i]:.8f}", obs[i], f"{ph:.6f}",
            f"{S_target[i]:.6e}", f"{(S_target[i] - ob_mean_i):.6e}",
            f"{s_det_i:.6f}", f"{flux_diff[i]:.6e}", f"{F_D[i]:.6e}",
            f"{fwhm[i]:.3f}", int(good[i]),
        ])

print(f"  Saved: {lc_out}")

# Print binned summary
print("\n=== Phase-binned lightcurve ===")
print(f"  {'Bin centre':>12}  {'Mean':>10}  {'Err':>8}  {'N':>5}")
for i in range(N_PHASE_BINS):
    if bin_n[i] >= MIN_BIN_PTS:
        print(f"  phi={bin_centres[i]:.3f}     {bin_mean[i]:>10.4f}  {bin_err[i]:>8.4f}  {bin_n[i]:>5}")

print(f"\nOutputs saved to: {OUT_DIR}")
print("Done.")
