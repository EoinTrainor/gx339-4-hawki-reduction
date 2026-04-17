# title: Stage A (Single Frame) – 5-Star Review Browsers + Full-Frame Map (Locked Annulus + Core-Npix SNR + Max-SNR Opt + Background Structure Diagnostics)

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.modeling import models, fitting

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry


# ---------------- USER INPUTS ----------------
FITS_FILE = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

STAR_BOXES = {
    "Star 1": (1500, 1580, 1520, 1580),
    "Star 2": (1020, 1100, 1370, 1450),
    "Star 3": (2080, 2150, 1410, 1460),
    "Star 4": (1210, 1270, 1310, 1370),
    "Star 5": (1360, 1430,  880,  940),
}

# GX 339-4 target position (pixel coordinates)
# Set this to your known target centroid in the full image.
GX_XY = (1292.5, 1278.5)

# Background core definition (truncated core)
CORE_NSIG = 2.0  # core window = median ± CORE_NSIG * robust_sigma

# Aperture sweep (in multiples of each star's own FWHM) for plotting SNR curves
AP_R_MIN_FWHM = 0.5
AP_R_MAX_FWHM = 3.0
AP_R_STEP_FWHM = 0.1

# Locked annulus geometry from GLOBAL median FWHM (frame-consistent)
ANN_IN_FWHM = 3.0
ANN_OUT_FWHM = 5.0

# Cutouts
CUTOUT_HALF_SIZE = 60
PERCENTILE_STRETCH = (5, 99.7)

# Background structure stretch around annulus median (aggressive)
# vmin/vmax = ann_median ± BG_STRUCT_NSIG * ann_robust_sigma
BG_STRUCT_NSIG = 3.0
BG_STRUCT_CMAP = "inferno"  # aggressive colormap requested

# Full-frame display stretch
FULLFRAME_STRETCH = (1, 99.7)

# Optional: label offsets in pixels on full-frame map
LABEL_DX_DY = (14, 14)

# Screening thresholds (simple, defendable)
# If you know saturation threshold for the detector, set it here.
# Otherwise, keep conservative: exclude stars with very high peaks relative to frame.
SAT_ADU_ABS = None        # e.g., 60000 for 16-bit saturation. If None, use percentile test only.
SAT_FRAME_PCTL = 99.999   # exclude if peak pixel is above this percentile of full-frame finite pixels

# Gaussian fit quality / elongation screening
MAX_RESID_RMS = 500.0     # reject if fit residual is too large (blends/trails)
MAX_AXIS_RATIO = 1.35     # reject if FWHM_x/FWHM_y too elongated

# Export option
EXPORT_CSV = True
CSV_OUT = "stageA_star_summary_single_frame.csv"  # saved in current working directory
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

def extract_cutout(data, x, y, half):
    ny, nx = data.shape
    x0, y0 = int(round(float(x))), int(round(float(y)))
    x1, x2 = max(0, x0 - half), min(nx, x0 + half + 1)
    y1, y2 = max(0, y0 - half), min(ny, y0 + half + 1)
    sub = data[y1:y2, x1:x2]
    return sub, x1, y1

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

def centroid_bgsub_in_box(data, x1, x2, y1, y2):
    cut = data[y1:y2, x1:x2]
    if cut.size == 0 or not np.isfinite(cut).any():
        return np.nan, np.nan

    finite = np.isfinite(cut)
    if finite.sum() == 0:
        return np.nan, np.nan

    idx = np.nanargmax(np.where(finite, cut, -np.inf))
    iy, ix = np.unravel_index(idx, cut.shape)

    x_seed = x1 + ix
    y_seed = y1 + iy

    half = 10
    ny, nx = data.shape
    xa, xb = max(0, x_seed - half), min(nx, x_seed + half + 1)
    ya, yb = max(0, y_seed - half), min(ny, y_seed + half + 1)

    win = data[ya:yb, xa:xb]
    if win.size == 0 or not np.isfinite(win).any():
        return float(x_seed), float(y_seed)

    yy, xx = np.mgrid[ya:yb, xa:xb]
    m = np.isfinite(win)

    bkg = float(np.nanmedian(win[m]))
    w = win[m] - bkg
    w[w < 0] = 0

    if w.size == 0 or np.nansum(w) == 0:
        return float(x_seed), float(y_seed)

    xc = float(np.sum(xx[m] * w) / np.sum(w))
    yc = float(np.sum(yy[m] * w) / np.sum(w))
    return xc, yc

def fit_2d_gaussian_fwhm(data, x_cen, y_cen, half=10):
    """
    Returns:
      fwhm_mean, fwhm_x, fwhm_y, resid_rms, axis_ratio, fit_ok
    """
    sub, xoff, yoff = extract_cutout(data, x_cen, y_cen, half)
    if sub.size == 0 or not np.isfinite(sub).any():
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    yy, xx = np.mgrid[0:sub.shape[0], 0:sub.shape[1]]

    amp0 = np.nanmax(sub) - np.nanmedian(sub)
    bkg0 = np.nanmedian(sub)
    x0 = (x_cen - xoff)
    y0 = (y_cen - yoff)
    sig0 = max(1.0, half / 3)

    g = (
        models.Gaussian2D(amplitude=amp0, x_mean=x0, y_mean=y0, x_stddev=sig0, y_stddev=sig0)
        + models.Const2D(amplitude=bkg0)
    )

    fitter = fitting.LevMarLSQFitter()

    mask = np.isfinite(sub)
    if mask.sum() < 20:
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    fit = fitter(g, xx[mask], yy[mask], sub[mask])

    # Print fit issues (requested)
    if hasattr(fitter, "fit_info"):
        msg = fitter.fit_info.get("message", "")
        if msg and "unsuccessful" in str(msg).lower():
            print("WARNING: Gaussian fit issue at x,y =", x_cen, y_cen, "| message:", msg)

    sx = float(np.abs(fit[0].x_stddev.value))
    sy = float(np.abs(fit[0].y_stddev.value))
    if not (np.isfinite(sx) and np.isfinite(sy) and sx > 0 and sy > 0):
        return np.nan, np.nan, np.nan, np.nan, np.nan, False

    fwhm_x = 2.35482 * sx
    fwhm_y = 2.35482 * sy
    fwhm_mean = 0.5 * (fwhm_x + fwhm_y)

    axis_ratio = float(max(fwhm_x, fwhm_y) / max(1e-9, min(fwhm_x, fwhm_y)))

    model_full = np.full_like(sub, np.nan, dtype=float)
    model_full[mask] = fit(xx[mask], yy[mask])
    resid = sub - model_full
    resid_rms = float(np.sqrt(np.nanmean((resid[mask])**2)))

    fit_ok = True
    return float(fwhm_mean), float(fwhm_x), float(fwhm_y), resid_rms, axis_ratio, fit_ok

def annulus_mask_and_values(data, x, y, r_in, r_out):
    ann = CircularAnnulus([(x, y)], r_in=r_in, r_out=r_out)
    m = ann.to_mask(method="exact")[0]
    arr = m.multiply(data)
    vals = arr[m.data > 0]
    vals = vals[np.isfinite(vals)]
    return vals, m

def core_mask_from_values(mask_obj, data, x, y, r_in, r_out, core_lo, core_hi):
    """
    Build a boolean mask image (same size as data cutout later) is done elsewhere.
    Here: return a boolean mask for the full image around (x,y) for annulus pixels that are within core window.
    We do this by reconstructing annulus values and applying threshold in-place on full frame.
    """
    # This helper exists for clarity; for overlays we build the mask in cutout coordinates.
    return None

def truncated_core_background(vals, core_nsig=2.0):
    """
    Returns:
      B, sigma_sky, med, sig_rob, core_lo, core_hi, n_raw, n_core
    """
    v = vals[np.isfinite(vals)]
    n_raw = int(v.size)
    if n_raw < 80:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n_raw, 0

    med = float(np.nanmedian(v))
    sig_rob = robust_sigma_mad(v)
    if not np.isfinite(sig_rob) or sig_rob <= 0:
        sig_rob = float(np.nanstd(v))

    core_lo = med - core_nsig * sig_rob
    core_hi = med + core_nsig * sig_rob

    core = v[(v >= core_lo) & (v <= core_hi)]
    n_core = int(core.size)

    if n_core < 50:
        B = med
        sigma_sky = sig_rob
        return float(B), float(sigma_sky), med, sig_rob, float(core_lo), float(core_hi), n_raw, n_core

    B = float(np.nanmean(core))
    sigma_sky = robust_sigma_mad(core)  # use robust sky sigma (consistent)
    if not np.isfinite(sigma_sky) or sigma_sky <= 0:
        sigma_sky = float(np.nanstd(core, ddof=1)) if core.size > 1 else float(np.nanstd(core))
        if not np.isfinite(sigma_sky) or sigma_sky <= 0:
            sigma_sky = sig_rob

    return float(B), float(sigma_sky), med, sig_rob, float(core_lo), float(core_hi), n_raw, n_core

def compute_snr3(raw_sum, ap_area, B, sigma_sky, n_ann_effective):
    """
    SNR3:
      F_net = raw_sum - B*Npix
      var = sigma_sky^2 * (Npix + Npix^2 / Nann_effective)
    Note: Nann_effective MUST correspond to the pixel set used to estimate background (core pixels).
    """
    if not (np.isfinite(raw_sum) and np.isfinite(ap_area) and np.isfinite(B) and np.isfinite(sigma_sky)):
        return np.nan, np.nan, np.nan
    if ap_area <= 0 or sigma_sky <= 0 or n_ann_effective <= 0:
        return np.nan, np.nan, np.nan

    F_net = float(raw_sum - B * ap_area)
    var = (sigma_sky ** 2) * (ap_area + (ap_area ** 2) / float(n_ann_effective))
    if not np.isfinite(var) or var <= 0:
        return F_net, np.nan, np.nan

    F_err = float(np.sqrt(var))
    snr = float(F_net / F_err) if F_err > 0 else np.nan
    return F_net, F_err, snr

def radial_median_profile(data, x, y, r_max, dr=1.0):
    """
    Radial median profile of ADU around (x,y) in the full frame (local cutout).
    Returns centers and median values.
    """
    sub, xoff, yoff = extract_cutout(data, x, y, int(np.ceil(r_max)) + 2)
    if sub.size == 0 or not np.isfinite(sub).any():
        return np.array([]), np.array([])

    xc, yc = x - xoff, y - yoff
    yy, xx = np.indices(sub.shape)
    rr = np.sqrt((xx - xc)**2 + (yy - yc)**2)

    bins = np.arange(0, r_max + dr, dr)
    r_cent = 0.5 * (bins[:-1] + bins[1:])
    med = np.full(r_cent.shape, np.nan, dtype=float)
    for i in range(len(r_cent)):
        m = (rr >= bins[i]) & (rr < bins[i+1]) & np.isfinite(sub)
        if np.any(m):
            med[i] = float(np.nanmedian(sub[m]))
    return r_cent, med


# ---------------- BROWSER UI ----------------
def _key_is_left(k: str) -> bool:
    return k in ("left", ",", "<", "comma")

def _key_is_right(k: str) -> bool:
    return k in ("right", ".", ">", "period")

def _key_is_exit(k: str) -> bool:
    return k in ("enter", "return", "escape", "q")

def browse_plots(items, draw_fn, window_title="Browser"):
    if len(items) == 0:
        print(f"[{window_title}] No items to browse.")
        return

    idx = {"i": 0}

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    try:
        fig.canvas.manager.set_window_title(window_title)
    except Exception:
        pass

    def redraw():
        ax.clear()
        draw_fn(ax, items[idx["i"]])
        fig.canvas.draw_idle()

    def on_key(event):
        k = (event.key or "").lower()
        if _key_is_left(k):
            idx["i"] = (idx["i"] - 1) % len(items)
            redraw()
        elif _key_is_right(k):
            idx["i"] = (idx["i"] + 1) % len(items)
            redraw()
        elif _key_is_exit(k):
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    redraw()
    plt.show()


# ---------------- LOAD IMAGE ----------------
data, wcs, header = load_first_2d_hdu(FITS_FILE)

print("\n==============================")
print("Processing:", FITS_FILE)
print("==============================")

# Precompute frame percentile-based saturation proxy
frame_finite = data[np.isfinite(data)]
frame_sat_level = float(np.nanpercentile(frame_finite, SAT_FRAME_PCTL)) if frame_finite.size else np.nan

results = []
snr_items = []
hist_items = []
overlay_items = []
bgstruct_items = []
radprof_items = []

# ---------------- PASS 1: CENTROIDS + FWHM for ALL STARS ----------------
star_base = []
fwhm_list = []

for name, (x1, x2, y1, y2) in STAR_BOXES.items():
    x_cen, y_cen = centroid_bgsub_in_box(data, x1, x2, y1, y2)

    fwhm_mean, fwhm_x, fwhm_y, resid_rms, axis_ratio, fit_ok = fit_2d_gaussian_fwhm(data, x_cen, y_cen, half=10)
    if np.isfinite(fwhm_mean):
        fwhm_list.append(fwhm_mean)

    # simple peak in the local box for saturation screening
    box = data[y1:y2, x1:x2]
    peak_box = float(np.nanmax(box)) if np.isfinite(box).any() else np.nan

    star_base.append({
        "star": name,
        "x": x_cen,
        "y": y_cen,
        "fwhm_mean": fwhm_mean,
        "fwhm_x": fwhm_x,
        "fwhm_y": fwhm_y,
        "resid_rms": resid_rms,
        "axis_ratio": axis_ratio,
        "fit_ok": bool(fit_ok),
        "peak_box": peak_box,
        "box": (x1, x2, y1, y2),
    })

fwhm_global = float(np.nanmedian(fwhm_list)) if len(fwhm_list) else np.nan
if not np.isfinite(fwhm_global):
    raise RuntimeError("Failed to estimate global median FWHM from comparison stars. Check STAR_BOXES or fit window.")

ANN_IN_PX_LOCKED = max(5.0, ANN_IN_FWHM * fwhm_global)
ANN_OUT_PX_LOCKED = max(ANN_IN_PX_LOCKED + 5.0, ANN_OUT_FWHM * fwhm_global)

print("\n================= LOCKED ANNULUS (FRAME CONSISTENT) =================")
print(f"Global median FWHM = {fwhm_global:.3f} px")
print(f"Locked annulus: r_in = {ANN_IN_PX_LOCKED:.1f} px  | r_out = {ANN_OUT_PX_LOCKED:.1f} px")
print("Reason: ensures comp-star ranking is comparable (same background geometry for all stars).")
print("=====================================================================\n")


# ---------------- PASS 2: PER-STAR BACKGROUND + SNR CURVES + DIAGNOSTICS ----------------
for s in star_base:
    name = s["star"]
    x_cen, y_cen = s["x"], s["y"]

    print(f"\n--- {name} ---")
    print(f"Centroid (box)  x={x_cen:.2f} px, y={y_cen:.2f} px")
    print(f"FWHM            mean={s['fwhm_mean']:.2f} px  (x={s['fwhm_x']:.2f}, y={s['fwhm_y']:.2f}) | "
          f"axis_ratio={s['axis_ratio']:.2f} | resid_rms={s['resid_rms']:.2f}")

    # Quality screening flags
    sat_flag = False
    if SAT_ADU_ABS is not None and np.isfinite(s["peak_box"]) and s["peak_box"] >= SAT_ADU_ABS:
        sat_flag = True
    if SAT_ADU_ABS is None and np.isfinite(s["peak_box"]) and np.isfinite(frame_sat_level) and s["peak_box"] >= frame_sat_level:
        sat_flag = True

    fit_flag = False
    if (not s["fit_ok"]) or (not np.isfinite(s["resid_rms"])) or (s["resid_rms"] > MAX_RESID_RMS):
        fit_flag = True
    if (not np.isfinite(s["axis_ratio"])) or (s["axis_ratio"] > MAX_AXIS_RATIO):
        fit_flag = True

    if sat_flag:
        print(f"SCREEN: saturation-risk flag ON | peak_box={s['peak_box']:.1f} ADU | frame pctl({SAT_FRAME_PCTL})={frame_sat_level:.1f} ADU")
    if fit_flag:
        print("SCREEN: fit/shape flag ON | (resid_rms too high and/or elongation too high and/or fit failed)")

    # Locked annulus used for ALL stars
    ann_vals, ann_mask_obj = annulus_mask_and_values(data, x_cen, y_cen, ANN_IN_PX_LOCKED, ANN_OUT_PX_LOCKED)
    B, sigma_sky, med, sig_rob, core_lo, core_hi, n_raw, n_core = truncated_core_background(
        ann_vals, core_nsig=CORE_NSIG
    )

    print(f"Annulus (locked)   {ANN_IN_PX_LOCKED:.1f}–{ANN_OUT_PX_LOCKED:.1f} px | N_raw={n_raw} px | N_core={n_core} px")
    print(f"Background (core)  B={B:.3f} ADU | sigma_sky={sigma_sky:.3f} ADU (MAD on core)")
    print(f"Core window        [{core_lo:.3f}, {core_hi:.3f}] ADU  (median={med:.3f}, robust sigma={sig_rob:.3f})")
    print("SNR3 note: uses N_core (effective pixels) in penalty term to avoid overstating SNR.")

    # Aperture radii sweep (scaled by star FWHM for diagnostic curves)
    if not np.isfinite(s["fwhm_mean"]) or s["fwhm_mean"] <= 0:
        r_list = np.array([], dtype=float)
        snr_arr = np.array([], dtype=float)
    else:
        r_list = np.arange(AP_R_MIN_FWHM, AP_R_MAX_FWHM + 1e-9, AP_R_STEP_FWHM) * s["fwhm_mean"]

        snr_list = []
        net_list = []
        err_list = []
        for r in r_list:
            ap = CircularAperture([(x_cen, y_cen)], r=float(r))
            ap_tbl = aperture_photometry(data, ap)
            raw_sum = float(ap_tbl["aperture_sum"][0])
            ap_area = float(ap.area)
            F_net, F_err, snr = compute_snr3(raw_sum, ap_area, B, sigma_sky, max(n_core, 1))
            snr_list.append(snr)
            net_list.append(F_net)
            err_list.append(F_err)

        snr_arr = np.array(snr_list, dtype=float)
        net_arr = np.array(net_list, dtype=float)
        err_arr = np.array(err_list, dtype=float)

    # Choose optimum radius by PURE MAX SNR (supervisor preference)
    opt_r = np.nan
    max_snr = np.nan
    idx_best = None
    if snr_arr.size > 0 and np.isfinite(snr_arr).any():
        idx_best = int(np.nanargmax(snr_arr))
        opt_r = float(r_list[idx_best])
        max_snr = float(snr_arr[idx_best])

    # Stability metric (kept as diagnostic / explainability)
    # Use local sensitivity around optimum: fractional drop within ±10% of r_opt
    sens = np.nan
    if idx_best is not None and np.isfinite(opt_r) and opt_r > 0:
        r_lo = 0.9 * opt_r
        r_hi = 1.1 * opt_r
        i_lo = int(np.argmin(np.abs(r_list - r_lo)))
        i_hi = int(np.argmin(np.abs(r_list - r_hi)))
        if np.isfinite(snr_arr[i_lo]) and np.isfinite(snr_arr[i_hi]) and np.isfinite(max_snr) and max_snr > 0:
            sens = float(np.abs(snr_arr[i_hi] - snr_arr[i_lo]) / max_snr)

    # Save summary row
    results.append({
        "star": name,
        "x_px": x_cen,
        "y_px": y_cen,
        "fwhm_mean_px": s["fwhm_mean"],
        "fwhm_x_px": s["fwhm_x"],
        "fwhm_y_px": s["fwhm_y"],
        "axis_ratio": s["axis_ratio"],
        "resid_rms": s["resid_rms"],
        "peak_box_ADU": s["peak_box"],
        "sat_flag": int(sat_flag),
        "fit_flag": int(fit_flag),
        "ann_in_px": ANN_IN_PX_LOCKED,
        "ann_out_px": ANN_OUT_PX_LOCKED,
        "B_ADU": B,
        "sigma_sky_ADU": sigma_sky,
        "ann_Npix_raw": n_raw,
        "ann_Npix_core": n_core,
        "max_snr3": max_snr,
        "opt_r_px": opt_r,
        "snr_sensitivity_frac": sens,
    })

    # For browsers
    snr_items.append({
        "star": name,
        "r_list": r_list.copy(),
        "snr_arr": snr_arr.copy(),
        "max_snr": max_snr,
        "opt_r": opt_r,
        "ann_in": ANN_IN_PX_LOCKED,
        "ann_out": ANN_OUT_PX_LOCKED,
        "sens": sens,
        "sat_flag": sat_flag,
        "fit_flag": fit_flag,
    })

    hist_items.append({
        "star": name,
        "ann_vals": ann_vals.copy(),
        "B": B,
        "sigma_sky": sigma_sky,
        "core_lo": core_lo,
        "core_hi": core_hi,
        "ann_in": ANN_IN_PX_LOCKED,
        "ann_out": ANN_OUT_PX_LOCKED,
        "n_raw": n_raw,
        "n_core": n_core,
    })

    # Normal overlay (star visibility stretch)
    sub, xoff, yoff = extract_cutout(data, x_cen, y_cen, CUTOUT_HALF_SIZE)
    xc, yc = x_cen - xoff, y_cen - yoff
    vmin, vmax = safe_percentile_limits(sub, *PERCENTILE_STRETCH)

    overlay_items.append({
        "star": name,
        "sub": sub.copy(),
        "xc": float(xc),
        "yc": float(yc),
        "vmin": vmin,
        "vmax": vmax,
        "r_opt": float(opt_r) if np.isfinite(opt_r) else float(1.5 * fwhm_global),
        "ann_in": float(ANN_IN_PX_LOCKED),
        "ann_out": float(ANN_OUT_PX_LOCKED),
        "core_lo": float(core_lo),
        "core_hi": float(core_hi),
        "B": float(B),
        "sat_flag": sat_flag,
        "fit_flag": fit_flag,
    })

    # Background structure overlay (aggressive stretch around annulus median)
    if np.isfinite(med) and np.isfinite(sig_rob) and sig_rob > 0:
        vmin_bg = med - BG_STRUCT_NSIG * sig_rob
        vmax_bg = med + BG_STRUCT_NSIG * sig_rob
    else:
        vmin_bg, vmax_bg = safe_percentile_limits(sub, *PERCENTILE_STRETCH)

    bgstruct_items.append({
        "star": name,
        "sub": sub.copy(),
        "xc": float(xc),
        "yc": float(yc),
        "vmin_bg": float(vmin_bg),
        "vmax_bg": float(vmax_bg),
        "ann_in": float(ANN_IN_PX_LOCKED),
        "ann_out": float(ANN_OUT_PX_LOCKED),
        "core_lo": float(core_lo),
        "core_hi": float(core_hi),
        "B": float(B),
        "med": float(med),
        "sig_rob": float(sig_rob),
    })

    # Radial median profile (diagnostic for gradients / wings)
    r_cent, prof_med = radial_median_profile(data, x_cen, y_cen, r_max=min(80.0, ANN_OUT_PX_LOCKED + 10.0), dr=1.0)
    radprof_items.append({
        "star": name,
        "r_cent": r_cent,
        "prof_med": prof_med,
        "ann_in": float(ANN_IN_PX_LOCKED),
        "ann_out": float(ANN_OUT_PX_LOCKED),
        "B": float(B),
        "core_lo": float(core_lo),
        "core_hi": float(core_hi),
    })


# ---------------- RANKING SUMMARY (downweight flagged stars) ----------------
for r in results:
    max_snr = r["max_snr3"]
    sens = r["snr_sensitivity_frac"]
    coreN = r["ann_Npix_core"]

    if not np.isfinite(max_snr):
        r["score"] = -np.inf
        continue

    # screening penalty
    penalty = 1.0
    if r["sat_flag"] == 1:
        penalty *= 0.1
    if r["fit_flag"] == 1:
        penalty *= 0.2

    sens_term = 1.0 / (1.0 + (sens if np.isfinite(sens) else 1.0))
    core_term = np.sqrt(coreN) if (coreN is not None and coreN > 0) else 1.0

    r["score"] = float(max_snr * sens_term * core_term * penalty)

results_sorted = sorted(results, key=lambda d: d["score"], reverse=True)

print("\n================= COMPARISON STAR RANKING (Single Frame) =================")
print("Locked annulus used for ALL stars.")
print("SNR3 uses N_core (effective background pixels) to avoid overstating SNR.")
print("Metric notes:")
print("  max_snr3             higher is better")
print("  snr_sensitivity_frac lower is better (diagnostic stability around optimum)")
print("  core_Npix            higher is better (more robust background core)")
print("  score                includes screening penalties (sat/fit)\n")

print(f"{'Star':<8} {'FWHM(px)':>9} {'maxSNR3':>9} {'opt r(px)':>9} {'sens':>8} {'coreN':>8} {'sat':>5} {'fit':>5} {'ann(px)':>12} {'score':>10}")
for r in results_sorted:
    ann_str = f"{r['ann_in_px']:.0f}-{r['ann_out_px']:.0f}"
    print(f"{r['star']:<8} {r['fwhm_mean_px']:>9.2f} {r['max_snr3']:>9.3f} {r['opt_r_px']:>9.2f} "
          f"{r['snr_sensitivity_frac']:>8.3f} {int(r['ann_Npix_core']):>8d} {int(r['sat_flag']):>5d} {int(r['fit_flag']):>5d} "
          f"{ann_str:>12} {r['score']:>10.2f}")

best = results_sorted[0]
print("\nBest comparison star (by score):", best["star"])
print("Suggested optimum aperture radius (px):", f"{best['opt_r_px']:.2f}", "(pure max SNR)")
print("Annulus (px):", f"{best['ann_in_px']:.1f}–{best['ann_out_px']:.1f}", "(locked)")
print("==========================================================================\n")


# ---------------- EXPORT SUMMARY CSV (one row per star) ----------------
if EXPORT_CSV:
    try:
        import pandas as pd
        df = pd.DataFrame(results_sorted)
        df.to_csv(CSV_OUT, index=False)
        print(f"[Saved] {CSV_OUT}")
    except Exception as e:
        print("CSV export failed:", repr(e))


# ---------------- BROWSE: SNR PLOTS ----------------
def draw_snr(ax, it):
    r_list = it["r_list"]
    snr_arr = it["snr_arr"]
    opt_r = it["opt_r"]
    max_snr = it["max_snr"]

    if r_list.size == 0:
        ax.text(0.5, 0.5, "No SNR curve (FWHM failed)", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.plot(r_list, snr_arr, marker="o", linestyle="-", ms=3)
    if np.isfinite(opt_r):
        ax.axvline(opt_r, linestyle="--", lw=2, label=f"opt r (max SNR) = {opt_r:.2f}px")
    if np.isfinite(max_snr):
        ax.axhline(max_snr, linestyle=":", lw=1.5, label=f"max SNR = {max_snr:.1f}")

    flags = []
    if it["sat_flag"]:
        flags.append("SAT?")
    if it["fit_flag"]:
        flags.append("FIT?")
    flag_txt = (" | flags: " + ",".join(flags)) if flags else ""

    ax.set_xlabel("Aperture radius (px)")
    ax.set_ylabel("SNR3 (dimensionless)")
    ax.set_title(
        f"{it['star']}: SNR3 vs aperture radius (locked annulus {it['ann_in']:.1f}–{it['ann_out']:.1f}px)\n"
        f"sens={it['sens']:.3f}{flag_txt} | Controls: < / > (or arrows), Enter exits"
    )
    ax.legend(frameon=False)

browse_plots(snr_items, draw_snr, window_title="SNR Browser (5 stars) – max SNR + locked annulus + N_core")


# ---------------- BROWSE: ANNULUS HISTOGRAMS (core window markers) ----------------
def draw_hist(ax, it):
    v = it["ann_vals"]
    v = v[np.isfinite(v)]
    if v.size == 0:
        ax.text(0.5, 0.5, "No finite annulus values", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    B = it["B"]
    sigma_sky = it["sigma_sky"]
    core_lo, core_hi = it["core_lo"], it["core_hi"]

    ax.hist(v, bins=140, alpha=0.55, density=True, label="Annulus values")

    # Visualise truncated core window explicitly
    ax.axvline(core_lo, linestyle=":", lw=2, label=f"core_lo ({CORE_NSIG}σ_rob)")
    ax.axvline(core_hi, linestyle=":", lw=2, label=f"core_hi ({CORE_NSIG}σ_rob)")
    ax.axvline(B, linestyle="--", lw=2, label="B (core mean)")

    # Plot a Gaussian using sigma_sky just as a *core* reference (not full distribution fit)
    if np.isfinite(B) and np.isfinite(sigma_sky) and sigma_sky > 0:
        xgrid = np.linspace(np.nanpercentile(v, 0.5), np.nanpercentile(v, 99.5), 500)
        gauss = (1.0 / (sigma_sky * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((xgrid - B) / sigma_sky) ** 2)
        ax.plot(xgrid, gauss, lw=2.0, label="Gaussian(core proxy)")

    ax.set_xlabel("Annulus pixel value (ADU)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"{it['star']}: annulus histogram + truncated core window\n"
        f"locked ann {it['ann_in']:.1f}–{it['ann_out']:.1f}px | N_raw={it['n_raw']} | N_core={it['n_core']} | "
        f"Controls: < / > (or arrows), Enter exits"
    )
    ax.legend(frameon=False)

browse_plots(hist_items, draw_hist, window_title="Histogram Browser (5 stars) – core window + N_core")


# ---------------- BROWSE: APERTURE / ANNULUS OVERLAYS (star visibility) + CORE MASK OVERLAY ----------------
def draw_overlay(ax, it):
    sub = it["sub"]
    ax.imshow(sub, origin="lower", cmap="gray", vmin=it["vmin"], vmax=it["vmax"])

    xc, yc = it["xc"], it["yc"]
    r_opt = it["r_opt"]
    ann_in = it["ann_in"]
    ann_out = it["ann_out"]

    # Draw aperture + annulus
    CircularAperture([(xc, yc)], r=r_opt).plot(ax=ax, color="yellow", lw=2.0)
    CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.6)
    ax.plot([xc], [yc], "r+", ms=12, mew=2)

    # Overlay "core-used" annulus pixels mask (in cutout coords)
    ann_cut = CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out)
    m = ann_cut.to_mask(method="exact")[0]
    ann_mask_img = m.to_image(sub.shape)
    if ann_mask_img is not None:
        ann_pixels = (ann_mask_img > 0) & np.isfinite(sub)
        core_pixels = ann_pixels & (sub >= it["core_lo"]) & (sub <= it["core_hi"])
        # show core pixels as a semi-transparent overlay
        overlay = np.zeros_like(sub, dtype=float)
        overlay[core_pixels] = 1.0
        ax.imshow(overlay, origin="lower", alpha=0.28, cmap="winter")

    flags = []
    if it["sat_flag"]:
        flags.append("SAT?")
    if it["fit_flag"]:
        flags.append("FIT?")
    flag_txt = (" | flags: " + ",".join(flags)) if flags else ""

    ax.set_title(
        f"{it['star']} overlay | r_ap={r_opt:.2f}px | ann={ann_in:.1f}-{ann_out:.1f}px"
        f"{flag_txt}\n(core-used annulus pixels overlaid)"
    )
    ax.set_axis_off()

browse_plots(overlay_items, draw_overlay, window_title="Overlay Browser (5 stars) – annulus core mask overlay")


# ---------------- BROWSE: BACKGROUND STRUCTURE VIEW (aggressive stretch) ----------------
def draw_bgstruct(ax, it):
    sub = it["sub"]
    ax.imshow(sub, origin="lower", cmap=BG_STRUCT_CMAP, vmin=it["vmin_bg"], vmax=it["vmax_bg"])

    xc, yc = it["xc"], it["yc"]
    CircularAnnulus([(xc, yc)], r_in=it["ann_in"], r_out=it["ann_out"]).plot(ax=ax, color="cyan", lw=1.8)
    ax.plot([xc], [yc], "w+", ms=12, mew=2)

    ax.set_title(
        f"{it['star']} background structure view ({BG_STRUCT_CMAP})\n"
        f"stretch = median ± {BG_STRUCT_NSIG:.1f}×robust_sigma | annulus shown"
    )
    ax.set_axis_off()

browse_plots(bgstruct_items, draw_bgstruct, window_title="Background Structure Browser (5 stars)")


# ---------------- BROWSE: RADIAL MEDIAN PROFILE ----------------
def draw_radprof(ax, it):
    r_cent = it["r_cent"]
    prof = it["prof_med"]
    if r_cent.size == 0:
        ax.text(0.5, 0.5, "No radial profile", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return

    ax.plot(r_cent, prof, marker="o", ms=3)
    ax.axvline(it["ann_in"], linestyle=":", lw=2, label="annulus in")
    ax.axvline(it["ann_out"], linestyle=":", lw=2, label="annulus out")
    if np.isfinite(it["B"]):
        ax.axhline(it["B"], linestyle="--", lw=2, label="B (core mean)")

    ax.set_xlabel("Radius (px)")
    ax.set_ylabel("Median ADU")
    ax.set_title(f"{it['star']}: radial median profile (gradient / wings check)")
    ax.legend(frameon=False)

browse_plots(radprof_items, draw_radprof, window_title="Radial Median Profile Browser (5 stars)")


# ---------------- FULL-FRAME MAP (ALL COMPARISON STARS + GX 339-4) ----------------
print("Opening full-frame map...")

vmin_full, vmax_full = safe_percentile_limits(data, *FULLFRAME_STRETCH)

fig, ax = plt.subplots(figsize=(11.8, 11.0))
try:
    fig.canvas.manager.set_window_title("Full FITS Frame – Comparison Stars + GX 339-4 (Locked Annulus)")
except Exception:
    pass

ax.imshow(data, origin="lower", cmap="gray", vmin=vmin_full, vmax=vmax_full)

dx, dy = LABEL_DX_DY

# Plot each comparison star aperture+annulus and label
for r in results:
    x, y = r["x_px"], r["y_px"]
    if not (np.isfinite(x) and np.isfinite(y)):
        continue

    CircularAperture([(x, y)], r=r["opt_r_px"]).plot(ax=ax, color="yellow", lw=1.8)
    CircularAnnulus([(x, y)], r_in=r["ann_in_px"], r_out=r["ann_out_px"]).plot(ax=ax, color="cyan", lw=1.3)

    label = r["star"]
    if r["sat_flag"] == 1:
        label += " [SAT?]"
    if r["fit_flag"] == 1:
        label += " [FIT?]"

    ax.text(x + dx, y + dy, label, color="yellow", fontsize=10,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2.5))

# GX 339-4 marker
gx, gy = GX_XY
if np.isfinite(gx) and np.isfinite(gy):
    ax.plot([gx], [gy], marker="+", color="red", ms=18, mew=2.5)
    ax.text(gx + dx, gy + dy, "GX 339-4", color="red", fontsize=12,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=2.5))
else:
    ax.text(0.02, 0.02,
            "GX_XY not set (set GX_XY=(x,y) in USER INPUTS to draw red cross).",
            transform=ax.transAxes, color="red", fontsize=11,
            bbox=dict(facecolor="black", alpha=0.35, edgecolor="none", pad=3))

ax.set_title("Full FITS frame: comparison star apertures (yellow) + annuli (cyan, locked) + GX 339-4 (red +)")
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
plt.tight_layout()
plt.show()
