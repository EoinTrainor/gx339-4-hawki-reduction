# title: GX 339-4 Robust Aperture Fitting (Locked Annulus + Core Npix SNR + Max SNR Comp Star + Stability GX + Background Structure + Mask Overlays + Screening + Per Frame Summary)

"""
Single frame robust aperture selection for GX 339-4, anchored to comparison star PSF scale.

Changes applied from critiques:
A) SNR and geometry consistency
  1) Annulus is locked everywhere using global median FWHM from comp stars.
  2) SNR3 uses N_core (effective core pixels) in the penalty term (prevents overstated SNR).
  3) Any remaining 95 percent rule is removed. If stability fails, GX fallback is pure max SNR.

B) Visual diagnostics and background sanity
  4) Adds a background structure view with aggressive scaling and a strong colormap.
  5) Overlays the annulus core mask (pixels actually used after truncated core selection).
  6) Adds a radial median profile (and keeps existing radial profile and growth diagnostics).

C) Comparison star selection logic
  7) Exports stability metric (sens) with max SNR and other per star metrics.
  8) Adds saturation and nonlinearity screening (peak ADU thresholds) and fit quality screening.

D) Output and reproducibility
  9) Writes a per frame summary row (chosen comp star, FWHM median, annulus, GX radius choice logic, GX flux, etc).
  10) Keeps curve of growth as diagnostic only.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy.modeling import models, fitting
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry


# ---------------- USER INPUTS ----------------
FITS_FILE = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

# GX 339-4 sky coordinates (ICRS, degrees)
GX_RA_DEG = 255.7057818297
GX_DEC_DEG = -48.7897466540

# 5 bright comparison star search boxes (pixel coordinates)
STAR_BOXES = {
    "Star 1": (1500, 1580, 1520, 1580),
    "Star 2": (1020, 1100, 1370, 1450),
    "Star 3": (2080, 2150, 1410, 1460),
    "Star 4": (1210, 1270, 1310, 1370),
    "Star 5": (1360, 1430,  880,  940),
}

# Display / cutouts
CUTOUT_HALF_SIZE = 80
PERCENTILE_STRETCH = (5, 99.7)

# Background structure view
BG_STRUCT_NSIG = 3.0
BG_STRUCT_CMAP = "inferno"  # strong colormap as requested

# Centroiding + Gaussian fit window
CENTROID_HALF_SIZE = 12
GAUSS_FIT_HALF_SIZE = 12

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

# Stability criterion for GX aperture choice
STAB_CONSECUTIVE = 2
STAB_SIGMA = 2.0

# Star ranking weights
W_SNR = 1.0
W_SENS = 0.7
W_CORE = 0.2

# Screening thresholds (set these if you know saturation / nonlinearity)
SAT_ADU_ABS = None          # example: 60000 (16 bit). If None use percentile rule only.
SAT_FRAME_PCTL = 99.999     # if peak pixel in star cutout exceeds this percentile of frame, flag

MAX_AXIS_RATIO = 1.35       # fwhm_x / fwhm_y elongation flag
MAX_RESID_RMS = 500.0       # gaussian fit residual RMS flag

# Output
WRITE_SUMMARY_CSV = True
SUMMARY_CSV_OUT = "gx3394_single_frame_summary.csv "
# ---------------- END INPUTS ----------------


# ---------------- HELPERS ----------------
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

def centroid_from_box(data, box):
    x1, x2, y1, y2 = box
    sub = data[y1:y2, x1:x2]
    if sub.size == 0 or not np.isfinite(sub).any():
        return np.nan, np.nan
    yy, xx = np.mgrid[y1:y2, x1:x2]
    mask = np.isfinite(sub)
    flux = sub[mask]
    if flux.size == 0 or np.nansum(flux) == 0:
        return np.nan, np.nan
    xc = np.sum(xx[mask] * flux) / np.sum(flux)
    yc = np.sum(yy[mask] * flux) / np.sum(flux)
    return float(xc), float(yc)

def fit_fwhm_2d_gaussian(data, x, y, half):
    """
    Returns:
      fwhm_mean, fwhm_x, fwhm_y, x_fit, y_fit, resid_rms, axis_ratio, fit_ok
    """
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

    # requested diagnostic print only when fitter indicates issue
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
    """
    SNR3 uses n_ann_core (effective N) matching the background estimator pixel set (core pixels).
    """
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

def choose_gx_radius_by_stability(radii, flux, ferr, consecutive=4, sigma=1.0):
    radii = np.asarray(radii, float)
    flux = np.asarray(flux, float)
    ferr = np.asarray(ferr, float)

    ok = np.isfinite(radii) & np.isfinite(flux) & np.isfinite(ferr) & (ferr > 0)
    radii, flux, ferr = radii[ok], flux[ok], ferr[ok]
    if radii.size < consecutive + 2:
        return np.nan, None

    dF = np.abs(np.diff(flux))
    dE = np.sqrt(ferr[:-1] ** 2 + ferr[1:] ** 2)
    stable = dF <= (sigma * dE)

    for start in range(0, stable.size - consecutive + 1):
        if np.all(stable[start:start + consecutive]):
            idx = start + 1
            return float(radii[idx]), int(idx)
    return np.nan, None

def radial_profile_and_growth(sub, xc, yc, r_max, B=None):
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)

    dr = 0.5
    bins = np.arange(0, r_max + dr, dr)
    prof = np.zeros(len(bins) - 1, dtype=float)
    r_centers = 0.5 * (bins[:-1] + bins[1:])

    for k in range(len(bins) - 1):
        m = (rr >= bins[k]) & (rr < bins[k + 1]) & np.isfinite(sub)
        prof[k] = np.nanmean(sub[m]) if np.any(m) else np.nan

    r_ap = np.arange(0.5, r_max + 0.01, 0.5)
    growth = np.zeros_like(r_ap, dtype=float)
    for j, r in enumerate(r_ap):
        m = (rr <= r) & np.isfinite(sub)
        raw_sum = np.nansum(sub[m])
        area = np.sum(m)
        growth[j] = raw_sum - B * area if (B is not None and np.isfinite(B)) else raw_sum

    return r_centers, prof, r_ap, growth

def radial_median_profile(sub, xc, yc, r_max, dr=1.0):
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - xc) ** 2 + (yy - yc) ** 2)

    bins = np.arange(0, r_max + dr, dr)
    r_cent = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(r_cent.shape, np.nan, dtype=float)

    for i in range(r_cent.size):
        m = (rr >= bins[i]) & (rr < bins[i + 1]) & np.isfinite(sub)
        if np.any(m):
            med[i] = float(np.nanmedian(sub[m]))
    return r_cent, med

def maybe_add_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(frameon=False)


# ---------------- LOAD IMAGE ----------------
data, wcs, header = load_first_2d_hdu(FITS_FILE)

print("\n==============================")
print("Processing:", FITS_FILE)
print("==============================")

frame_finite = data[np.isfinite(data)]
frame_sat_level = float(np.nanpercentile(frame_finite, SAT_FRAME_PCTL)) if frame_finite.size else np.nan


# ---------------- STAGE 1: PSF SCALE FROM COMPARISON STARS ----------------
star_results = []
fwhm_list = []

for name, box in STAR_BOXES.items():
    x0, y0 = centroid_from_box(data, box)
    x_c, y_c = centroid_local_fluxweighted(data, x0, y0, CENTROID_HALF_SIZE)

    fwhm_mean, fx, fy, x_fit, y_fit, resid_rms, axis_ratio, fit_ok = fit_fwhm_2d_gaussian(
        data, x_c, y_c, GAUSS_FIT_HALF_SIZE
    )

    if np.isfinite(fwhm_mean):
        fwhm_list.append(fwhm_mean)

    # peak pixel in the star cutout for saturation screening
    x1, x2, y1, y2 = box
    box_cut = data[y1:y2, x1:x2]
    peak_box = float(np.nanmax(box_cut)) if np.isfinite(box_cut).any() else np.nan

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
        "x": x_fit if np.isfinite(x_fit) else x_c,
        "y": y_fit if np.isfinite(y_fit) else y_c,
        "fwhm": fwhm_mean,
        "fwhm_x": fx,
        "fwhm_y": fy,
        "axis_ratio": axis_ratio,
        "resid_rms": resid_rms,
        "fit_ok": bool(fit_ok),
        "peak_box": peak_box,
        "sat_flag": sat_flag,
        "fit_flag": fit_flag,
    })

fwhm_med = float(np.nanmedian(fwhm_list)) if len(fwhm_list) else np.nan
if not np.isfinite(fwhm_med):
    raise RuntimeError("FWHM estimation failed for comparison stars. Check boxes or GAUSS_FIT_HALF_SIZE.")

# LOCKED annulus for everything
ann_in = ANN_IN_FWHM * fwhm_med
ann_out = ANN_OUT_FWHM * fwhm_med

print("\n================= PSF / ANNULUS GEOMETRY =================")
print(f"Median FWHM from comp stars = {fwhm_med:.3f} px")
print("Annulus locked from median FWHM scaling:")
print(f"  r_in  = {ANN_IN_FWHM:.1f} x FWHM = {ann_in:.1f} px")
print(f"  r_out = {ANN_OUT_FWHM:.1f} x FWHM = {ann_out:.1f} px")
print("Reason: consistent geometry for all SNR and background estimates in this frame.")
print("==========================================================\n")


# ---------------- STAGE 2: RANK COMPARISON STARS (max SNR, stability metric as diagnostic) ----------------
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

    # penalty for screening flags
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
        "fwhm_x": s["fwhm_x"],
        "fwhm_y": s["fwhm_y"],
        "axis_ratio": s["axis_ratio"],
        "resid_rms": s["resid_rms"],
        "peak_box": s["peak_box"],
        "sat_flag": s["sat_flag"],
        "fit_flag": s["fit_flag"],
        "max_snr": max_snr,
        "r_opt": r_opt,     # pure max SNR optimum
        "sens": sens,       # diagnostic stability metric
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
    print(f"{r['name']:<7} {r['fwhm']:>9.2f} {r['max_snr']:>10.3f} {r['r_opt']:>10.2f} {r['sens']:>8.3f} {r['coreN']:>8d} "
          f"{int(r['sat_flag']):>5d} {int(r['fit_flag']):>5d} {ann_str:>12} {r['score']:>12.2f}")
print("\nBest comparison star (by score):", best["name"])
print(f"Suggested optimum aperture radius (px): {best['r_opt']:.2f} (pure max SNR)")
print(f"Annulus (px): {ann_in:.1f}–{ann_out:.1f} (locked)")
print("==========================================================================\n")


# ---------------- STAGE 3: APPLY TO GX AND CHOOSE GX R BY STABILITY ----------------
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

gx_r_choice, gx_idx_choice = choose_gx_radius_by_stability(
    gx_r, gx_net, gx_err, consecutive=STAB_CONSECUTIVE, sigma=STAB_SIGMA
)

# Fallback changed: pure max SNR (no 95 percent rule)
gx_choice_method = "stability"
if (gx_idx_choice is None) or (not np.isfinite(gx_r_choice)):
    ok = np.isfinite(gx_r) & np.isfinite(gx_snr)
    rr = gx_r[ok]
    ss = gx_snr[ok]
    if rr.size > 3 and np.isfinite(ss).any():
        i_max = int(np.nanargmax(ss))
        gx_r_choice = float(rr[i_max])
        gx_idx_choice = int(np.argmin(np.abs(gx_r - gx_r_choice)))
        gx_choice_method = "max_snr_fallback"

        print("================= GX APERTURE FALLBACK (MAX SNR) =================")
        print("Stability criterion did not find a plateau.")
        print("Supervisor preference: choose peak SNR for now.")
        print(f"max SNR3 = {float(ss[i_max]):.3f} at r = {gx_r_choice:.2f} px")
        print("===============================================================\n")
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
    print("No radius chosen (stability and fallback failed).")
print("==========================================================================\n")

gx_flux_out = np.nan
gx_err_out = np.nan
gx_snr_out = np.nan
if gx_idx_choice is not None:
    gx_flux_out = float(gx_net[gx_idx_choice])
    gx_err_out = float(gx_err[gx_idx_choice])
    gx_snr_out = float(gx_snr[gx_idx_choice])

    print("================= GX MEASUREMENT AT CHOSEN APERTURE =================")
    print(f"r_ap     = {gx_r[gx_idx_choice]:.2f} px")
    print(f"net flux = {gx_flux_out:.3f} ADU")
    print(f"err      = {gx_err_out:.3f} ADU")
    print(f"SNR3     = {gx_snr_out:.3f}")
    print("====================================================================\n")


# ---------------- STAGE 4: PLOTS (sanity + background structure + mask overlays) ----------------

# (1) chosen comparison star: SNR vs radius
best_sweep = best["sweep"]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(best_sweep[:, 0], best_sweep[:, 5], marker="o", ms=3)
ax.axvline(best["r_opt"], linestyle="--", lw=2, label=f"opt r (max SNR) = {best['r_opt']:.2f}px")
ax.set_title(f"{best['name']} SNR3 vs aperture radius (locked annulus {ann_in:.0f}–{ann_out:.0f}px)")
ax.set_xlabel("Aperture radius (px)")
ax.set_ylabel("SNR3")
maybe_add_legend(ax)
plt.tight_layout()
plt.show()

# (2) GX net flux vs radius with chosen radius marked
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.errorbar(gx_r, gx_net, yerr=gx_err, fmt="o", ms=3, capsize=2)
if np.isfinite(gx_r_choice):
    ax.axvline(gx_r_choice, linestyle="--", lw=2, label=f"chosen r = {gx_r_choice:.2f}px")
ax.set_title("GX 339-4 net flux vs aperture radius (diagnostic)")
ax.set_xlabel("Aperture radius (px)")
ax.set_ylabel("Net flux (ADU)")
maybe_add_legend(ax)
plt.tight_layout()
plt.show()

# (3) Stability diagnostic: |ΔF| vs allowed
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
    ax.set_title("GX stability test: flux change vs allowed change")
    ax.set_xlabel("Radius (px) (upper of each adjacent pair)")
    ax.set_ylabel("ADU")
    maybe_add_legend(ax)
    plt.tight_layout()
    plt.show()

# (4) GX annulus histogram + core window
ann_vals = gx_meta["ann_vals"]
B = gx_meta["B"]
lo, hi = gx_meta["core_window"]

fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.hist(ann_vals, bins=120, alpha=0.60, density=True, label="Annulus values")
ax.axvline(B, lw=2, label="B (core mean)")
ax.axvline(lo, lw=1.5, linestyle=":", label="core window")
ax.axvline(hi, lw=1.5, linestyle=":")
ax.set_title("GX 339-4 annulus histogram with truncated core window")
ax.set_xlabel("Annulus pixel value (ADU)")
ax.set_ylabel("Density")
maybe_add_legend(ax)
plt.tight_layout()
plt.show()

# (5) Background structure view (aggressive scaling around annulus median)
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
CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.8)
ax.plot([xc], [yc], "w+", ms=12, mew=2)
ax.set_title("GX background structure view (annulus shown)")
ax.axis("off")
plt.tight_layout()
plt.show()

# (6) Overlay: aperture + annulus + core used mask
gx_r_plot = gx_r_choice if np.isfinite(gx_r_choice) else best["r_opt"]

# build annulus core mask in cutout coords
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
ax.set_title("GX overlay with annulus core used mask")
ax.axis("off")
plt.tight_layout()
plt.show()

# (7) 2 by 2 effectiveness panel (kept)
r_max = min(CUTOUT_HALF_SIZE - 2, ann_out + 5)
r_cent, prof_mean, r_ap, growth = radial_profile_and_growth(sub_gx, xc, yc, r_max=r_max, B=B)

# extra radial median profile (requested)
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
ax.set_title("Curve of growth (background subtracted) (diagnostic)")
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

fig.suptitle(os.path.basename(FITS_FILE), y=0.99)
plt.tight_layout()
plt.show()

# (8) Overlay panel for best comparison star (kept)
bx, by = best["x"], best["y"]
subb, xoffb, yoffb = extract_cutout(data, bx, by, CUTOUT_HALF_SIZE)
xcb, ycb = bx - xoffb, by - yoffb
vminb, vmaxb = safe_percentile_limits(subb, *PERCENTILE_STRETCH)

fig, ax = plt.subplots(figsize=(5.6, 5.6))
ax.imshow(subb, origin="lower", cmap="gray", vmin=vminb, vmax=vmaxb)
CircularAperture([(xcb, ycb)], r=best["r_opt"]).plot(ax=ax, color="yellow", lw=2.0)
CircularAnnulus([(xcb, ycb)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.6)
ax.plot([xcb], [ycb], "r+", ms=12, mew=2)
ax.set_title(f"{best['name']} overlay | r_ap={best['r_opt']:.2f}px | ann={ann_in:.0f}-{ann_out:.0f}px")
ax.axis("off")
plt.tight_layout()
plt.show()


# ---------------- STAGE 5: WRITE PER FRAME SUMMARY ROW ----------------
if WRITE_SUMMARY_CSV:
    file_exists = os.path.exists(SUMMARY_CSV_OUT)

    row = {
        "fits_file": os.path.basename(FITS_FILE),
        "fwhm_median_px": fwhm_med,
        "ann_in_px": ann_in,
        "ann_out_px": ann_out,
        "best_comp_star": best["name"],
        "best_comp_r_opt_px": best["r_opt"],
        "best_comp_max_snr3": best["max_snr"],
        "best_comp_sens": best["sens"],
        "best_comp_coreN": best["coreN"],
        "gx_x_px": gx_x,
        "gx_y_px": gx_y,
        "gx_r_choice_px": gx_r_choice,
        "gx_choice_method": gx_choice_method,
        "gx_net_flux_adu": gx_flux_out,
        "gx_flux_err_adu": gx_err_out,
        "gx_snr3": gx_snr_out,
        "gx_B_adu": gx_meta["B"],
        "gx_sigma_sky_adu": gx_meta["sigma_sky"],
        "gx_coreN": gx_meta["n_core"],
    }

    fieldnames = list(row.keys())
    try:
        with open(SUMMARY_CSV_OUT, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow(row)
        print(f"[Saved] per frame summary row -> {SUMMARY_CSV_OUT}")
    except Exception as e:
        print("CSV write failed:", repr(e))
