# GX 339-4 Robust Aperture Fitting Batch Runner (WCS Tracked Comp Stars + Per-Frame Logs + Plot Saving)

"""
Batch runner version of the single-frame script.

For each FITS frame in FITS_FOLDER:
  1) Creates a per-frame output folder named by DATE-OBS (preferred) or filename.
  2) Saves the full console output to console_log.txt inside that folder.
  3) Saves each plot as PNG with filenames:
       OB01_comp_star_snr_vs_radius.png
       OB02_gx_net_flux_vs_radius.png
       OB03_gx_stability_test.png
       OB04_gx_annulus_histogram.png
       OB05_gx_background_structure_inferno.png
       OB06_gx_overlay_core_mask.png
       OB07_panel_2x2_diagnostics.png
       OB08_best_comp_overlay.png
     And plot titles are prefixed like:
       "OB 02: GX Net Flux vs aperture radius"

Additionally:
  - Writes/updates OUT_ROOT/summary_all_frames.csv (one row per FITS frame).

Notes:
  - Comparison stars are tracked by WCS (RA/Dec) each frame and refined via centroiding + Gaussian fit.
  - GX aperture radius is locked to best comparison star peak-SNR aperture radius (r_opt).
  - Inferno plot uses BLACK GX aperture circle for visibility.
  - Computes GX relative brightness = GX_net / Comp_net with propagated uncertainty.
"""

import os
import re
import csv
import glob
import contextlib
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.modeling import models, fitting
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry


# ---------------- USER INPUTS ----------------
FITS_FOLDER = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"
OUT_ROOT = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/6) Aperture Outputs"

FITS_GLOB = "*.fits"   # or "*.fit" depending on your files

# GX 339-4 sky coordinates (ICRS, degrees)
GX_RA_DEG = 255.7057818297
GX_DEC_DEG = -48.7897466540

# Comparison stars WCS coordinates (ICRS, degrees)
STAR_WCS_DEG = {"Star 3": (255.66889394, -48.78490688)}

# Display / cutouts
CUTOUT_HALF_SIZE = 80
PERCENTILE_STRETCH = (5, 99.7)

# Background structure view
BG_STRUCT_NSIG = 3.0
BG_STRUCT_CMAP = "inferno"

# Centroiding + Gaussian fit window
CENTROID_HALF_SIZE = 12
GAUSS_FIT_HALF_SIZE = 12

# Saturation screening cutout size around WCS guess for comp stars
SAT_CUTOUT_HALF_SIZE = 18

# Background model: truncated core
CORE_K = 2.5

# Annulus geometry in multiples of global median FWHM (LOCKED)
ANN_IN_FWHM = 3.0
ANN_OUT_FWHM = 5.0

# Aperture sweep for comparison stars (px)
AP_MIN_PX = 2.0
AP_MAX_PX = 16.0
AP_STEP_PX = 0.25

# Aperture sweep for GX (multiples of median FWHM)
GX_AP_MIN_FWHM = 0.35
GX_AP_MAX_FWHM = 2.5
GX_AP_STEP_FWHM = 0.03

# Stability criterion (diagnostic only; selection locked-to-comp)
STAB_CONSECUTIVE = 2
STAB_SIGMA = 2.0

# Star ranking weights
W_SNR = 1.0
W_SENS = 0.7
W_CORE = 0.2

# Screening thresholds
SAT_ADU_ABS = None
SAT_FRAME_PCTL = 99.999

MAX_AXIS_RATIO = 1.35
MAX_RESID_RMS = 500.0

# Output
WRITE_SUMMARY_CSV = True
SUMMARY_ALL_FRAMES = os.path.join(OUT_ROOT, "summary_all_frames.csv")

# Plot saving
PLOT_DPI = 200
# ---------------- END INPUTS ----------------


# ---------------- I/O HELPERS ----------------

class Tee:
    """Write to multiple file-like objects."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

def safe_name(s: str) -> str:
    s = str(s)
    s = s.strip()
    s = s.replace(":", "-")
    s = s.replace("/", "-")
    s = s.replace("\\", "-")
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s[:180] if len(s) > 180 else s

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_figure(fig, outdir: str, ob_num: int, slug: str, title: str, date_obs: str, fits_base: str):
    fn = f"OB{ob_num:02d}_{slug}.png"
    path = os.path.join(outdir, fn)

    # Put OB title as a figure-level title (keeps axes titles readable if you want to keep them)
    fig.suptitle(title, y=0.995)
    # Add small metadata footer
    footer = f"DATE-OBS: {date_obs} | FITS: {fits_base}"
    fig.text(0.01, 0.01, footer, fontsize=8, alpha=0.85)

    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved Plot] {fn}")

def get_date_obs_quick(fp: str) -> str:
    try:
        with fits.open(fp) as hdul:
            for h in hdul:
                hdr = getattr(h, "header", None)
                if hdr is None:
                    continue
                if "DATE-OBS" in hdr:
                    return str(hdr.get("DATE-OBS"))
    except Exception:
        pass
    return "UNKNOWN_DATE"


# ---------------- SCIENCE HELPERS ----------------

def load_first_2d_hdu(fp: str):
    with fits.open(fp) as hdul:
        data = None
        header = None
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data is not None and np.ndim(h.data) == 2:
                data = h.data.astype(float)
                header = h.header
                break
    if data is None:
        raise ValueError("No 2D image HDU found in FITS file.")
    data[~np.isfinite(data)] = np.nan
    return data, WCS(header), header

def wcs_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(c)
    return float(np.squeeze(x)), float(np.squeeze(y))

def extract_cutout(data, x, y, half):
    ny, nx = data.shape
    x0, y0 = int(round(float(x))), int(round(float(y)))
    x1, x2 = max(0, x0 - half), min(nx, x0 + half + 1)
    y1, y2 = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = data[y1:y2, x1:x2]
    return sub, x1, y1

def safe_percentile_limits(arr, lo, hi):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.nanpercentile(finite, [lo, hi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(finite))
        vmax = float(np.nanmax(finite))
        if vmin == vmax:
            vmax = vmin + 1.0
    return float(vmin), float(vmax)

def robust_sigma_mad(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(v)
    return float(sig)

def trimmed_std(v, trim_frac=0.05):
    v = v[np.isfinite(v)]
    if v.size < 10:
        return float(np.nanstd(v))
    lo = np.quantile(v, trim_frac)
    hi = np.quantile(v, 1 - trim_frac)
    vv = v[(v >= lo) & (v <= hi)]
    return float(np.nanstd(vv)) if vv.size > 5 else float(np.nanstd(v))

def centroid_local_fluxweighted(data, x0, y0, half):
    x0i, y0i = int(round(float(x0))), int(round(float(y0)))
    ny, nx = data.shape
    x1, x2 = max(0, x0i - half), min(nx, x0i + half + 1)
    y1, y2 = max(0, y0i - half), min(ny, y0i + half + 1)
    cut = data[y1:y2, x1:x2]
    if cut.size == 0 or not np.isfinite(cut).any():
        return np.nan, np.nan
    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = np.isfinite(cut)
    flux = cut[mask]
    if flux.size == 0 or np.nansum(flux) == 0:
        return np.nan, np.nan
    xc = np.sum(xx[mask] * flux) / np.sum(flux)
    yc = np.sum(yy[mask] * flux) / np.sum(flux)
    return float(xc), float(yc)

def fit_fwhm_2d_gaussian(data, x, y, half):
    sub, xoff, yoff = extract_cutout(data, x, y, half)
    if sub.size == 0 or not np.isfinite(sub).any():
        return np.nan, np.nan, np.nan, x, y, np.nan, np.nan, False

    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]
    z = sub.copy()
    mask = np.isfinite(z)
    if mask.sum() < 30:
        return np.nan, np.nan, np.nan, x, y, np.nan, np.nan, False

    z_med = np.nanmedian(z[mask])
    z_max = np.nanmax(z[mask])
    amp0 = max(1.0, float(z_max - z_med))
    x0 = float(x - xoff)
    y0 = float(y - yoff)

    g0 = (
        models.Gaussian2D(
            amplitude=amp0,
            x_mean=x0,
            y_mean=y0,
            x_stddev=3.0,
            y_stddev=3.0,
            theta=0.0
        )
        + models.Const2D(z_med)
    )

    fitter = fitting.LevMarLSQFitter()
    with np.errstate(all="ignore"):
        gfit = fitter(g0, xx[mask], yy[mask], z[mask])

    if hasattr(fitter, "fit_info"):
        msg = fitter.fit_info.get("message", "")
        if msg and "unsuccessful" in str(msg).lower():
            print(
                f"WARNING: Gaussian fit may be unsuccessful at "
                f"x={x:.2f}, y={y:.2f} (half={half}, cutout shape={sub.shape}) | message: {msg}"
            )

    try:
        x_fit = float(gfit[0].x_mean.value + xoff)
        y_fit = float(gfit[0].y_mean.value + yoff)
        sx = float(abs(gfit[0].x_stddev.value))
        sy = float(abs(gfit[0].y_stddev.value))
    except Exception:
        return np.nan, np.nan, np.nan, x, y, np.nan, np.nan, False

    if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
        return np.nan, np.nan, np.nan, x, y, np.nan, np.nan, False

    fwhm_x = 2.35482 * sx
    fwhm_y = 2.35482 * sy
    fwhm_mean = float(np.mean([fwhm_x, fwhm_y]))
    axis_ratio = float(max(fwhm_x, fwhm_y) / max(1e-9, min(fwhm_x, fwhm_y)))

    model = np.full_like(z, np.nan, dtype=float)
    model[mask] = gfit(xx[mask], yy[mask])
    resid = (z - model)
    resid_rms = float(np.sqrt(np.nanmean(resid[mask] ** 2)))

    return float(fwhm_mean), float(fwhm_x), float(fwhm_y), float(x_fit), float(y_fit), float(resid_rms), axis_ratio, True

def annulus_values(data, x, y, r_in, r_out):
    ann = CircularAnnulus([(x, y)], r_in=r_in, r_out=r_out)
    m = ann.to_mask(method="exact")[0]
    arr = m.multiply(data)
    vals = arr[m.data > 0]
    vals = vals[np.isfinite(vals)]
    return vals, m

def truncated_core_background(vals, k=2.5):
    v = vals[np.isfinite(vals)]
    if v.size < 50:
        return np.nan, np.nan, (np.nan, np.nan), np.nan, np.nan, 0, 0

    med = float(np.nanmedian(v))
    sig = robust_sigma_mad(v)
    if not np.isfinite(sig) or sig <= 0:
        sig = float(np.nanstd(v))
        if not np.isfinite(sig) or sig <= 0:
            return np.nan, np.nan, (np.nan, np.nan), med, sig, int(v.size), 0

    lo = med - k * sig
    hi = med + k * sig
    core = v[(v >= lo) & (v <= hi)]
    if core.size < 30:
        core = v.copy()

    B = float(np.nanmean(core))
    sigma_sky = robust_sigma_mad(core)

    n_raw = int(v.size)
    n_core = int(core.size)
    return B, sigma_sky, (float(lo), float(hi)), med, float(sig), n_raw, n_core

def compute_snr3(raw_sum, ap_area, B, sigma_sky, n_ann_core):
    if not (np.isfinite(raw_sum) and np.isfinite(ap_area) and np.isfinite(B) and np.isfinite(sigma_sky)):
        return np.nan, np.nan, np.nan
    if ap_area <= 0 or sigma_sky <= 0 or n_ann_core <= 0:
        return np.nan, np.nan, np.nan

    F_net = float(raw_sum - B * ap_area)
    var = (sigma_sky ** 2) * (ap_area + (ap_area ** 2) / float(n_ann_core))
    if not np.isfinite(var) or var <= 0:
        return F_net, np.nan, np.nan

    F_err = float(np.sqrt(var))
    snr = float(F_net / F_err) if F_err > 0 else np.nan
    return F_net, F_err, snr

def sweep_snr_vs_aperture(data, x, y, ann_in, ann_out, ap_radii):
    ann_vals, _ = annulus_values(data, x, y, ann_in, ann_out)
    B, sigma_sky, core_window, med, sig_rob, n_raw, n_core = truncated_core_background(ann_vals, k=CORE_K)

    out = []
    for r in ap_radii:
        ap = CircularAperture([(x, y)], r=float(r))
        tbl = aperture_photometry(data, ap)
        raw_sum = float(tbl["aperture_sum"][0])
        ap_area = float(ap.area)
        F_net, F_err, snr = compute_snr3(raw_sum, ap_area, B, sigma_sky, n_core)
        out.append((float(r), raw_sum, ap_area, F_net, F_err, snr))

    out = np.array(out, dtype=float)

    raw_std = float(np.nanstd(ann_vals)) if ann_vals.size else np.nan
    raw_tstd = trimmed_std(ann_vals) if ann_vals.size else np.nan
    in_core = (ann_vals >= core_window[0]) & (ann_vals <= core_window[1]) if ann_vals.size else np.array([], bool)
    core_vals = ann_vals[in_core] if ann_vals.size else np.array([], float)
    core_std = float(np.nanstd(core_vals)) if core_vals.size else np.nan
    core_tstd = trimmed_std(core_vals) if core_vals.size else np.nan

    meta = {
        "B": B,
        "sigma_sky": sigma_sky,
        "core_window": core_window,
        "raw_med": med,
        "raw_sigma_rob": sig_rob,
        "n_raw": n_raw,
        "n_core": n_core,
        "raw_std_diag": raw_std,
        "raw_trimstd_diag": raw_tstd,
        "core_std_diag": core_std,
        "core_trimstd_diag": core_tstd,
        "ann_vals": ann_vals,
    }
    return out, meta

def snr_sensitivity_fraction(snr_arr, idx_best, frac_window=0.10):
    if snr_arr.size < 5 or idx_best is None:
        return np.inf
    n = snr_arr.size
    w = max(1, int(frac_window * n))
    i1 = max(0, idx_best - w)
    i2 = min(n - 1, idx_best + w)
    peak = snr_arr[idx_best]
    local = snr_arr[i1:i2 + 1]
    if not np.isfinite(peak) or peak <= 0:
        return np.inf
    drop = (peak - np.nanmean(local)) / peak
    return float(drop) if np.isfinite(drop) else np.inf

def maybe_add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(frameon=False)


# ---------------- PER-FRAME PROCESSOR ----------------

def process_one_frame(fits_path: str, outdir: str):
    fits_base = os.path.basename(fits_path)

    data, wcs, header = load_first_2d_hdu(fits_path)
    date_obs = str(header.get("DATE-OBS", "UNKNOWN_DATE"))

    print("\n==============================")
    print("Processing:", fits_path)
    print("==============================")

    frame_finite = data[np.isfinite(data)]
    frame_sat_level = float(np.nanpercentile(frame_finite, SAT_FRAME_PCTL)) if frame_finite.size else np.nan

    # -------- Stage 1: Comp stars (WCS-tracked) --------
    star_results = []
    fwhm_list = []

    for name, (ra_deg, dec_deg) in STAR_WCS_DEG.items():
        x_guess, y_guess = wcs_to_pixel(wcs, ra_deg, dec_deg)

        x_c, y_c = centroid_local_fluxweighted(data, x_guess, y_guess, CENTROID_HALF_SIZE)
        if not (np.isfinite(x_c) and np.isfinite(y_c)):
            x_c, y_c = x_guess, y_guess

        fwhm_mean, fx, fy, x_fit, y_fit, resid_rms, axis_ratio, fit_ok = fit_fwhm_2d_gaussian(
            data, x_c, y_c, GAUSS_FIT_HALF_SIZE
        )
        if np.isfinite(fwhm_mean):
            fwhm_list.append(fwhm_mean)

        sat_cut, _, _ = extract_cutout(data, x_guess, y_guess, SAT_CUTOUT_HALF_SIZE)
        peak_box = float(np.nanmax(sat_cut)) if (sat_cut.size > 0 and np.isfinite(sat_cut).any()) else np.nan

        sat_flag = False
        if SAT_ADU_ABS is not None and np.isfinite(peak_box) and peak_box >= SAT_ADU_ABS:
            sat_flag = True
        if SAT_ADU_ABS is None and np.isfinite(peak_box) and np.isfinite(frame_sat_level) and peak_box >= frame_sat_level:
            sat_flag = True

        fit_flag = False
        if (not fit_ok) or (not np.isfinite(resid_rms)) or (resid_rms > MAX_RESID_RMS):
            fit_flag = True
        if (not np.isfinite(axis_ratio)) or (axis_ratio > MAX_AXIS_RATIO):
            fit_flag = True

        star_results.append({
            "name": name,
            "ra_deg": ra_deg,
            "dec_deg": dec_deg,
            "x": x_fit if np.isfinite(x_fit) else x_c,
            "y": y_fit if np.isfinite(y_fit) else y_c,
            "fwhm": fwhm_mean,
            "sat_flag": sat_flag,
            "fit_flag": fit_flag,
        })

    fwhm_med = float(np.nanmedian(fwhm_list)) if len(fwhm_list) else np.nan
    if not np.isfinite(fwhm_med):
        raise RuntimeError("FWHM estimation failed for comparison stars (WCS tracking).")

    ann_in = ANN_IN_FWHM * fwhm_med
    ann_out = ANN_OUT_FWHM * fwhm_med

    print("\n================= PSF / ANNULUS GEOMETRY =================")
    print(f"Median FWHM from comp stars = {fwhm_med:.3f} px")
    print("Annulus locked from median FWHM scaling:")
    print(f"  r_in  = {ANN_IN_FWHM:.1f} x FWHM = {ann_in:.1f} px")
    print(f"  r_out = {ANN_OUT_FWHM:.1f} x FWHM = {ann_out:.1f} px")
    print("Reason: consistent geometry for all SNR and background estimates in this frame.")
    print("==========================================================\n")

    # -------- Stage 2: Rank comp stars --------
    ap_radii = np.arange(AP_MIN_PX, AP_MAX_PX + 1e-9, AP_STEP_PX)

    ranking_rows = []
    for s in star_results:
        name, x, y = s["name"], s["x"], s["y"]
        sweep, meta = sweep_snr_vs_aperture(data, x, y, ann_in, ann_out, ap_radii)

        radii = sweep[:, 0]
        snr = sweep[:, 5]
        idx_best = int(np.nanargmax(snr)) if np.isfinite(snr).any() else None
        max_snr = float(np.nanmax(snr)) if np.isfinite(snr).any() else np.nan
        r_opt = float(radii[idx_best]) if idx_best is not None else np.nan
        sens = snr_sensitivity_fraction(snr, idx_best, frac_window=0.10)
        coreN = int(meta["n_core"])

        score = (W_SNR * max_snr) * (1.0 / (1.0 + W_SENS * sens)) * (1.0 + W_CORE * np.log10(max(coreN, 10)))

        penalty = 1.0
        if s["sat_flag"]:
            penalty *= 0.1
        if s["fit_flag"]:
            penalty *= 0.2
        score = float(score * penalty)

        ranking_rows.append({
            "name": name,
            "x": x,
            "y": y,
            "fwhm": s["fwhm"],
            "ra_deg": s["ra_deg"],
            "dec_deg": s["dec_deg"],
            "sat_flag": s["sat_flag"],
            "fit_flag": s["fit_flag"],
            "max_snr": max_snr,
            "r_opt": r_opt,
            "sens": sens,
            "coreN": coreN,
            "score": score,
            "meta": meta,
            "sweep": sweep
        })

    ranking_rows = sorted(ranking_rows, key=lambda r: r["score"], reverse=True)
    best = ranking_rows[0]

    print("================= COMPARISON STAR RANKING (Single Frame) =================")
    print("Metric notes:")
    print("  maxSNR3             higher is better (pure max)")
    print("  sens                lower is better (diagnostic stability around optimum)")
    print("  coreN               higher is better (background core sampling)")
    print("  sat, fit            flagged stars are downweighted")
    print("  score               combined quick rank score with penalties\n")

    print(f"{'Star':<7} {'FWHM(px)':>9} {'maxSNR3':>10} {'opt r(px)':>10} {'sens':>8} {'coreN':>8} {'sat':>5} {'fit':>5} {'ann(px)':>12} {'score':>12}")
    for r in ranking_rows:
        ann_str = f"{ann_in:.0f}-{ann_out:.0f}"
        print(
            f"{r['name']:<7} {r['fwhm']:>9.2f} {r['max_snr']:>10.3f} {r['r_opt']:>10.2f} {r['sens']:>8.3f} {r['coreN']:>8d} "
            f"{int(r['sat_flag']):>5d} {int(r['fit_flag']):>5d} {ann_str:>12} {r['score']:>12.2f}"
        )

    print("\nBest comparison star (by score):", best["name"])
    print(f"Suggested optimum aperture radius (px): {best['r_opt']:.2f} (pure max SNR)")
    print(f"Annulus (px): {ann_in:.1f}–{ann_out:.1f} (locked)")
    print("==========================================================================\n")

    # comp measurement at its chosen radius
    comp_flux_out = np.nan
    comp_err_out = np.nan
    comp_snr_out = np.nan
    comp_r_ap_out = np.nan

    best_sweep = best["sweep"]
    best_r = best_sweep[:, 0]
    best_net = best_sweep[:, 3]
    best_err = best_sweep[:, 4]
    best_snr = best_sweep[:, 5]

    comp_idx = None
    if np.isfinite(best.get("r_opt", np.nan)):
        comp_idx = int(np.argmin(np.abs(best_r - best["r_opt"])))
    if comp_idx is not None:
        comp_r_ap_out = float(best_r[comp_idx])
        comp_flux_out = float(best_net[comp_idx])
        comp_err_out = float(best_err[comp_idx])
        comp_snr_out = float(best_snr[comp_idx])

        print("================= COMPARISON STAR MEASUREMENT AT CHOSEN APERTURE =================")
        print(f"DATE OBS = {date_obs}")
        print(f"Star     = {best['name']}")
        print(f"r_ap     = {comp_r_ap_out:.2f} px")
        print(f"net flux = {comp_flux_out:.3f} ADU")
        print(f"err      = {comp_err_out:.3f} ADU")
        print(f"SNR3     = {comp_snr_out:.3f}")
        print("====================================================================\n")

    # -------- Stage 3: GX --------
    gx_x_guess, gx_y_guess = wcs_to_pixel(wcs, GX_RA_DEG, GX_DEC_DEG)
    gx_x, gx_y = centroid_local_fluxweighted(data, gx_x_guess, gx_y_guess, CENTROID_HALF_SIZE)

    print("================= GX 339-4 CENTROID =================")
    print(f"WCS guess (px): x={gx_x_guess:.2f}, y={gx_y_guess:.2f}")
    print(f"Centroid  (px): x={gx_x:.2f}, y={gx_y:.2f}")
    print(f"Offset    (px): dx={gx_x-gx_x_guess:+.2f}, dy={gx_y-gx_y_guess:+.2f}")
    print("Reason: recentering reduces aperture loss and makes stability test meaningful.")
    print("=====================================================\n")

    gx_radii = np.arange(GX_AP_MIN_FWHM, GX_AP_MAX_FWHM + 1e-9, GX_AP_STEP_FWHM) * fwhm_med
    gx_sweep, gx_meta = sweep_snr_vs_aperture(data, gx_x, gx_y, ann_in, ann_out, gx_radii)

    gx_r = gx_sweep[:, 0]
    gx_net = gx_sweep[:, 3]
    gx_err = gx_sweep[:, 4]
    gx_snr = gx_sweep[:, 5]

    # peak SNR (for reporting percentage)
    gx_max_snr = np.nan
    gx_r_at_max_snr = np.nan
    gx_idx_max_snr = None
    if np.isfinite(gx_snr).any():
        gx_idx_max_snr = int(np.nanargmax(gx_snr))
        gx_max_snr = float(gx_snr[gx_idx_max_snr])
        gx_r_at_max_snr = float(gx_r[gx_idx_max_snr])

    # lock GX radius to comp r_opt
    gx_r_choice = float(best["r_opt"]) if np.isfinite(best.get("r_opt", np.nan)) else np.nan
    gx_idx_choice = None
    gx_choice_method = "locked_to_comp_star_peak_snr"

    if np.isfinite(gx_r_choice):
        gx_idx_choice = int(np.argmin(np.abs(gx_r - gx_r_choice)))
        print("================= GX APERTURE (LOCKED TO COMPARISON STAR) =================")
        print(f"Using best comparison star: {best['name']} | r_opt (peak SNR) = {best['r_opt']:.2f} px")
        print(f"GX aperture radius locked to {gx_r_choice:.2f} px (nearest sampled radius index = {gx_idx_choice})")
        print("==========================================================================\n")
    else:
        gx_choice_method = "failed"

    print("================= GX BACKGROUND (Truncated Core) =================")
    print(f"Annulus (px): {ann_in:.1f}–{ann_out:.1f} (locked)")
    print(f"Annulus pixels raw  = {gx_meta['n_raw']} px")
    print(f"Annulus pixels core = {gx_meta['n_core']} px")
    print(f"RAW median          = {gx_meta['raw_med']:.3f} ADU")
    print(f"RAW robust sigma    = {gx_meta['raw_sigma_rob']:.3f} ADU (MAD)")
    print(f"Core window         = [{gx_meta['core_window'][0]:.3f}, {gx_meta['core_window'][1]:.3f}] ADU")
    print(f"B (core mean)       = {gx_meta['B']:.3f} ADU")
    print(f"sigma_sky (core)    = {gx_meta['sigma_sky']:.3f} ADU (MAD)")
    print(f"Diagnostics: raw std={gx_meta['raw_std_diag']:.3f} ADU | raw trimmed std={gx_meta['raw_trimstd_diag']:.3f} ADU")
    print("Reason: sky estimated from distribution core; SNR penalty uses N_core to avoid overstating SNR.")
    print("=====================================================\n")

    print("================= GX APERTURE CHOICE =================")
    print(f"Tested radii range: {gx_r.min():.2f}–{gx_r.max():.2f} px ({GX_AP_MIN_FWHM:.2f}–{GX_AP_MAX_FWHM:.2f} x FWHM)")
    print(
        "Stability criterion: smallest r such that |ΔF(r_i→r_{i+1})| ≤ "
        f"{STAB_SIGMA:.1f} × sqrt(σ_i^2 + σ_{{i+1}}^2) for {STAB_CONSECUTIVE} consecutive adjacent steps."
    )
    print("Choice method:", gx_choice_method)
    if np.isfinite(gx_r_choice):
        print(f"Chosen GX aperture radius = {gx_r_choice:.2f} px")
    else:
        print("No radius chosen (lock failed).")
    print("==========================================================================\n")

    gx_flux_out = np.nan
    gx_err_out = np.nan
    gx_snr_out = np.nan
    gx_r_ap_out = np.nan

    if gx_idx_choice is not None:
        gx_r_ap_out = float(gx_r[gx_idx_choice])
        gx_flux_out = float(gx_net[gx_idx_choice])
        gx_err_out = float(gx_err[gx_idx_choice])
        gx_snr_out = float(gx_snr[gx_idx_choice])

        gx_snr_pct_of_max = np.nan
        if np.isfinite(gx_snr_out) and np.isfinite(gx_max_snr) and gx_max_snr > 0:
            gx_snr_pct_of_max = 100.0 * gx_snr_out / gx_max_snr

        print(f"================= GX MEASUREMENT AT {date_obs} =================")
        print(f"r_ap     = {gx_r_ap_out:.2f} px")
        print(f"net flux = {gx_flux_out:.3f} ADU")
        print(f"err      = {gx_err_out:.3f} ADU")
        if np.isfinite(gx_snr_pct_of_max):
            print(f"SNR3     = {gx_snr_out:.3f} ({gx_snr_pct_of_max:.1f}% of max)")
        else:
            print(f"SNR3     = {gx_snr_out:.3f}")
        if np.isfinite(gx_max_snr) and np.isfinite(gx_r_at_max_snr):
            print(f"max_SNR3 = {gx_max_snr:.3f} at r = {gx_r_at_max_snr:.2f} px")
        print("====================================================================\n")

    # GX relative brightness ratio (propagated)
    gx_rel_brightness = np.nan
    gx_rel_err = np.nan
    if np.isfinite(gx_flux_out) and np.isfinite(comp_flux_out) and comp_flux_out != 0:
        gx_rel_brightness = gx_flux_out / comp_flux_out
        if gx_flux_out != 0 and np.isfinite(gx_err_out) and np.isfinite(comp_err_out):
            gx_rel_err = abs(gx_rel_brightness) * np.sqrt(
                (gx_err_out / gx_flux_out) ** 2 + (comp_err_out / comp_flux_out) ** 2
            )

    print("================= GX RELATIVE BRIGHTNESS =================")
    print(f"GX / {best['name']} = {gx_rel_brightness:.6f}")
    if np.isfinite(gx_rel_err):
        print(f"Ratio err   = {gx_rel_err:.6f}")
    print("=========================================================\n")

    # -------- Stage 4: PLOTS (saved; no plt.show) --------

    # OB01: comp star SNR vs radius
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(best_sweep[:, 0], best_sweep[:, 5], marker="o", ms=3)
    ax.axvline(best["r_opt"], linestyle="--", lw=2, label=f"opt r (max SNR) = {best['r_opt']:.2f}px")
    ax.set_xlabel("Aperture radius (px)")
    ax.set_ylabel("SNR3")
    maybe_add_legend(ax)
    plt.tight_layout()
    save_figure(fig, outdir, 1, "comp_star_snr_vs_radius", "OB 01: Comparison star SNR vs aperture radius", date_obs, fits_base)

    # OB02: GX net flux vs radius
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.errorbar(gx_r, gx_net, yerr=gx_err, fmt="o", ms=3, capsize=2)
    if np.isfinite(gx_r_choice):
        ax.axvline(gx_r_choice, linestyle="--", lw=2, label=f"chosen r = {gx_r_choice:.2f}px")
    ax.set_xlabel("Aperture radius (px)")
    ax.set_ylabel("Net flux (ADU)")
    maybe_add_legend(ax)
    plt.tight_layout()
    save_figure(fig, outdir, 2, "gx_net_flux_vs_radius", "OB 02: GX Net Flux vs aperture radius", date_obs, fits_base)

    # OB03: stability diagnostic
    ok = np.isfinite(gx_r) & np.isfinite(gx_net) & np.isfinite(gx_err) & (gx_err > 0)
    r2 = gx_r[ok]
    F2 = gx_net[ok]
    E2 = gx_err[ok]
    if r2.size >= 3:
        dF = np.abs(np.diff(F2))
        dE = np.sqrt(E2[:-1] ** 2 + E2[1:] ** 2)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        ax.plot(r2[1:], dF, marker="o", ms=3, label="|ΔF| between adjacent radii")
        ax.plot(r2[1:], STAB_SIGMA * dE, linestyle="--", lw=2, label=f"{STAB_SIGMA:.1f} x combined uncertainty")
        if np.isfinite(gx_r_choice):
            ax.axvline(gx_r_choice, linestyle=":", lw=2, label="chosen r")
        ax.set_xlabel("Radius (px) (upper of each adjacent pair)")
        ax.set_ylabel("ADU")
        maybe_add_legend(ax)
        plt.tight_layout()
        save_figure(fig, outdir, 3, "gx_stability_test", "OB 03: GX stability diagnostic", date_obs, fits_base)

    # OB04: GX annulus histogram
    ann_vals = gx_meta["ann_vals"]
    B = gx_meta["B"]
    lo, hi = gx_meta["core_window"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.hist(ann_vals, bins=120, alpha=0.60, density=True, label="Annulus values")
    ax.axvline(B, lw=2, label="B (core mean)")
    ax.axvline(lo, lw=1.5, linestyle=":", label="core window")
    ax.axvline(hi, lw=1.5, linestyle=":")
    ax.set_xlabel("Annulus pixel value (ADU)")
    ax.set_ylabel("Density")
    maybe_add_legend(ax)
    plt.tight_layout()
    save_figure(fig, outdir, 4, "gx_annulus_histogram", "OB 04: GX annulus histogram", date_obs, fits_base)

    # OB05: background structure inferno + black aperture circle
    sub_gx, xoff, yoff = extract_cutout(data, gx_x, gx_y, CUTOUT_HALF_SIZE)
    xc, yc = gx_x - xoff, gx_y - yoff
    vmin_star, vmax_star = safe_percentile_limits(sub_gx, *PERCENTILE_STRETCH)

    med_bg = gx_meta["raw_med"]
    sig_bg = gx_meta["raw_sigma_rob"]
    if np.isfinite(med_bg) and np.isfinite(sig_bg) and sig_bg > 0:
        vmin_bg = med_bg - BG_STRUCT_NSIG * sig_bg
        vmax_bg = med_bg + BG_STRUCT_NSIG * sig_bg
    else:
        vmin_bg, vmax_bg = vmin_star, vmax_star

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.imshow(sub_gx, origin="lower", cmap=BG_STRUCT_CMAP, vmin=vmin_bg, vmax=vmax_bg)
    gx_r_inferno = gx_r_choice if np.isfinite(gx_r_choice) else best["r_opt"]
    CircularAperture([(xc, yc)], r=gx_r_inferno).plot(ax=ax, color="black", lw=2.2)
    CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.8)
    ax.plot([xc], [yc], "w+", ms=12, mew=2)
    ax.axis("off")
    plt.tight_layout()
    save_figure(fig, outdir, 5, "gx_background_structure_inferno", "OB 05: GX background structure view (inferno)", date_obs, fits_base)

    # OB06: overlay with core mask
    gx_r_plot = gx_r_choice if np.isfinite(gx_r_choice) else best["r_opt"]

    ann = CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out)
    m = ann.to_mask(method="exact")[0]
    ann_mask_img = m.to_image(sub_gx.shape)
    core_overlay = np.zeros_like(sub_gx, dtype=float)
    if ann_mask_img is not None and np.isfinite(lo) and np.isfinite(hi):
        ann_pixels = (ann_mask_img > 0) & np.isfinite(sub_gx)
        core_pixels = ann_pixels & (sub_gx >= lo) & (sub_gx <= hi)
        core_overlay[core_pixels] = 1.0

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.imshow(sub_gx, origin="lower", cmap="gray", vmin=vmin_star, vmax=vmax_star)
    CircularAperture([(xc, yc)], r=gx_r_plot).plot(ax=ax, color="yellow", lw=2.0)
    CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.6)
    ax.imshow(core_overlay, origin="lower", alpha=0.28, cmap="winter")
    ax.plot([xc], [yc], "r+", ms=12, mew=2)
    ax.axis("off")
    plt.tight_layout()
    save_figure(fig, outdir, 6, "gx_overlay_core_mask", "OB 06: GX overlay with annulus core mask", date_obs, fits_base)

    # OB07: 2x2 diagnostics panel (rebuild quickly)
    def radial_profile_and_growth(sub, xc, yc, r_max, B=None):
        yy, xx = np.indices(sub.shape)
        rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        dr = 0.5
        bins = np.arange(0, r_max + dr, dr)
        prof = np.zeros(len(bins) - 1, dtype=float)
        r_centers = 0.5 * (bins[:-1] + bins[1:])
        for k in range(len(bins) - 1):
            mm = (rr >= bins[k]) & (rr < bins[k + 1]) & np.isfinite(sub)
            prof[k] = np.nanmean(sub[mm]) if np.any(mm) else np.nan
        r_ap = np.arange(0.5, r_max + 0.01, 0.5)
        growth = np.zeros_like(r_ap, dtype=float)
        for j, r in enumerate(r_ap):
            mm = (rr <= r) & np.isfinite(sub)
            raw_sum = np.nansum(sub[mm])
            area = np.sum(mm)
            growth[j] = raw_sum - B * area if (B is not None and np.isfinite(B)) else raw_sum
        return r_centers, prof, r_ap, growth

    def radial_median_profile(sub, xc, yc, r_max, dr=1.0):
        yy, xx = np.indices(sub.shape)
        rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)
        bins = np.arange(0, r_max + dr, dr)
        r_cent = 0.5 * (bins[:-1] + bins[1:])
        med = np.full(r_cent.shape, np.nan, dtype=float)
        for i in range(r_cent.size):
            mm = (rr >= bins[i]) & (rr < bins[i + 1]) & np.isfinite(sub)
            if np.any(mm):
                med[i] = float(np.nanmedian(sub[mm]))
        return r_cent, med

    r_max = min(CUTOUT_HALF_SIZE - 2, ann_out + 5)
    r_cent, prof_mean, r_ap, growth = radial_profile_and_growth(sub_gx, xc, yc, r_max=r_max, B=B)
    r_cent2, prof_med = radial_median_profile(sub_gx, xc, yc, r_max=r_max, dr=1.0)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 9.0))
    ax = axes[0, 0]
    ax.imshow(sub_gx, origin="lower", cmap="gray", vmin=vmin_star, vmax=vmax_star)
    CircularAperture([(xc, yc)], r=gx_r_plot).plot(ax=ax, color="yellow", lw=2.0)
    CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.6)
    ax.plot([xc], [yc], "r+", ms=12, mew=2)
    ax.set_title(f"GX overlay | r_ap={gx_r_plot:.2f}px | ann={ann_in:.0f}-{ann_out:.0f}px")
    ax.axis("off")

    ax = axes[0, 1]
    ax.plot(r_cent, prof_mean, marker="o", ms=2, label="mean ADU")
    if r_cent2.size:
        ax.plot(r_cent2, prof_med, marker="o", ms=2, label="median ADU")
    ax.axvline(gx_r_plot, linestyle="--", lw=2, label="chosen r_ap")
    ax.axvline(ann_in, linestyle=":", lw=2, label="annulus in")
    ax.axvline(ann_out, linestyle=":", lw=2, label="annulus out")
    ax.set_title("Radial profile")
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("ADU")
    maybe_add_legend(ax)

    ax = axes[1, 0]
    ax.plot(r_ap, growth, marker="o", ms=2)
    ax.axvline(gx_r_plot, linestyle="--", lw=2, label="chosen r_ap")
    ax.set_title("Curve of growth (diagnostic)")
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Cumulative net flux (ADU)")
    maybe_add_legend(ax)

    ax = axes[1, 1]
    ax.plot(gx_r, gx_snr, marker="o", ms=3)
    if np.isfinite(gx_r_choice):
        ax.axvline(gx_r_choice, linestyle="--", lw=2, label="chosen r")
    ax.set_title("GX SNR3 vs radius (diagnostic only)")
    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("SNR3")
    maybe_add_legend(ax)

    plt.tight_layout()
    save_figure(fig, outdir, 7, "panel_2x2_diagnostics", "OB 07: Diagnostics panel (2x2)", date_obs, fits_base)

    # OB08: best comp overlay
    bx, by = best["x"], best["y"]
    subb, xoffb, yoffb = extract_cutout(data, bx, by, CUTOUT_HALF_SIZE)
    xcb, ycb = bx - xoffb, by - yoffb
    vminb, vmaxb = safe_percentile_limits(subb, *PERCENTILE_STRETCH)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.imshow(subb, origin="lower", cmap="gray", vmin=vminb, vmax=vmaxb)
    CircularAperture([(xcb, ycb)], r=best["r_opt"]).plot(ax=ax, color="yellow", lw=2.0)
    CircularAnnulus([(xcb, ycb)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.6)
    ax.plot([xcb], [ycb], "r+", ms=12, mew=2)
    ax.axis("off")
    plt.tight_layout()
    save_figure(fig, outdir, 8, "best_comp_overlay", "OB 08: Best comparison star overlay", date_obs, fits_base)

    # -------- Stage 5: write summary row (master CSV) --------
    if WRITE_SUMMARY_CSV:
        row = {
            "fits_file": fits_base,
            "date_obs": date_obs,
            "fwhm_median_px": fwhm_med,
            "ann_in_px": ann_in,
            "ann_out_px": ann_out,
            "best_comp_star": best["name"],
            "best_comp_ra_deg": best["ra_deg"],
            "best_comp_dec_deg": best["dec_deg"],
            "best_comp_r_opt_px": best["r_opt"],
            "best_comp_max_snr3": best["max_snr"],
            "best_comp_sens": best["sens"],
            "best_comp_coreN": best["coreN"],
            "comp_net_flux_adu": comp_flux_out,
            "comp_flux_err_adu": comp_err_out,
            "gx_x_px": gx_x,
            "gx_y_px": gx_y,
            "gx_r_choice_px": gx_r_choice,
            "gx_r_ap_used_px": gx_r_ap_out,
            "gx_choice_method": gx_choice_method,
            "gx_net_flux_adu": gx_flux_out,
            "gx_flux_err_adu": gx_err_out,
            "gx_snr3": gx_snr_out,
            "gx_max_snr3": gx_max_snr,
            "gx_r_at_max_snr_px": gx_r_at_max_snr,
            "gx_rel_brightness": gx_rel_brightness,
            "gx_rel_brightness_err": gx_rel_err,
            "gx_B_adu": gx_meta["B"],
            "gx_sigma_sky_adu": gx_meta["sigma_sky"],
            "gx_coreN": gx_meta["n_core"],
        }

        ensure_dir(os.path.dirname(SUMMARY_ALL_FRAMES))
        file_exists = os.path.exists(SUMMARY_ALL_FRAMES)
        with open(SUMMARY_ALL_FRAMES, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                w.writeheader()
            w.writerow(row)

    return date_obs


# ---------------- MAIN BATCH LOOP ----------------

def main():
    ensure_dir(OUT_ROOT)

    fits_files = glob.glob(os.path.join(FITS_FOLDER, FITS_GLOB))
    if not fits_files:
        print("No FITS files found in:", FITS_FOLDER)
        return

    # sort by DATE-OBS if possible
    dated = []
    for fp in fits_files:
        d = get_date_obs_quick(fp)
        dated.append((d, fp))
    dated.sort(key=lambda t: t[0])
    fits_files = [fp for _, fp in dated]

    print(f"Found {len(fits_files)} FITS files.")
    print("Output root:", OUT_ROOT)
    print("Master summary:", SUMMARY_ALL_FRAMES)
    print("------------------------------------------------------------")

    for i, fp in enumerate(fits_files, start=1):
        fits_base = os.path.basename(fp)
        date_obs = get_date_obs_quick(fp)
        folder_name = safe_name(date_obs if date_obs and date_obs != "UNKNOWN_DATE" else os.path.splitext(fits_base)[0])
        outdir = os.path.join(OUT_ROOT, folder_name)
        ensure_dir(outdir)

        log_path = os.path.join(outdir, "console_log.txt")

        print(f"\n[{i}/{len(fits_files)}] {fits_base}")
        print(" ->", outdir)

        try:
            with open(log_path, "w", encoding="utf-8") as logf:
                tee = Tee(logf, os.sys.stdout)
                with contextlib.redirect_stdout(tee):
                    process_one_frame(fp, outdir)
        except Exception as e:
            # write failure to log and continue
            with open(log_path, "a", encoding="utf-8") as logf:
                logf.write("\n\n[ERROR]\n")
                logf.write(repr(e) + "\n")
            print("[ERROR] Failed on:", fits_base, "|", repr(e))
            continue

    print("\nDONE.")
    print("Master summary CSV:", SUMMARY_ALL_FRAMES)

if __name__ == "__main__":
    main()
