# Compariosn Stars - Residuals

"""
Produces every plot you need for the write-up, using FITS -> fixed aperture photometry.

Assumptions / choices (kept consistent with your latest runs):
- Fixed aperture radius: R_FIXED = 4.5 px
- Fixed background annulus: ANN_IN=12 px, ANN_OUT=20 px
- Truncated-core background estimate (CORE_K sigma)
- Centroid refinement (flux-weighted) around WCS-predicted position
- No saving: SHOW ONLY

Figures generated (in order):
FIG 1: Raw instrumental magnitudes for Stars 1–5 (median-subtracted) vs time
FIG 2: Ensemble common-mode magnitude m_ens vs time (shows frame-wide systematic drift)
FIG 3: Leave-one-out residuals r_i = m_i - m_ens,-i with uncertainties vs time (tests star stability)
FIG 4: Pairwise differential magnitudes Δm(i,j) for all 10 star pairs vs time (small multiples)
FIG 5: GX relative to Star 3: Δm(GX,3) with uncertainties vs time
FIG 6: GX relative to ensemble reference: Δm(GX,ens) with uncertainties vs time

Console outputs:
- Pairwise constancy metrics table (WRMS, chi2_red, slope_sig)
- Leave-one-out residual summary (std(res), median sigma, frac(|res|>sigma))
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.time import Time

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

# Optional SciPy for chi2 p-values
try:
    from scipy.stats import chi2 as _chi2_dist
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# ---------------- USER INPUT ----------------
FITS_FOLDER = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"
FITS_GLOB = "*.fits"  # or "*.fit"

STAR_WCS_DEG = {
    "Star 1": (255.69493499, -48.78181777),
    "Star 2": (255.71652635, -48.78600190),
    "Star 3": (255.66889394, -48.78490688),
    "Star 4": (255.70811477, -48.78784538),
    "Star 5": (255.70074950, -48.80085623),
}

GX_RA_DEG = 255.70575434196945
GX_DEC_DEG = -48.78971028788833

R_FIXED = 4.5
CENTROID_HALF = 12

# Fixed annulus for this diagnostic set
ANN_IN = 12.0
ANN_OUT = 20.0

CORE_K = 2.5
MIN_ANN_PIXELS = 80
MIN_CORE_PIXELS = 40

# Uncertainty model:
# Set to 0.0 for photon-only; set e.g. 0.010 for 10 mmag systematic floor for plots.
SIGMA_SYS_MAG = 0.010

# Pairwise constancy thresholds (for your interpretation in write-up)
P_VALUE_THRESHOLD = 0.01
SLOPE_SIG_THRESHOLD = 3.0
# -------------------------------------------------


def robust_sigma(v):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med))
    sig = 1.4826 * mad
    if not np.isfinite(sig) or sig <= 0:
        sig = np.nanstd(v)
    return sig


def load_first_2d_hdu(fp):
    with fits.open(fp) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data is not None and np.ndim(h.data) == 2:
                data = h.data.astype(float)
                hdr = h.header
                data[~np.isfinite(data)] = np.nan
                return data, hdr
    raise ValueError("No 2D image HDU found")


def wcs_to_pixel(wcs, ra, dec):
    c = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(c)
    return float(np.squeeze(x)), float(np.squeeze(y))


def centroid_fluxweighted(data, x0, y0, half):
    ny, nx = data.shape
    x0i, y0i = int(round(x0)), int(round(y0))
    x1 = max(0, x0i - half)
    x2 = min(nx, x0i + half + 1)
    y1 = max(0, y0i - half)
    y2 = min(ny, y0i + half + 1)
    cut = data[y1:y2, x1:x2]
    if cut.size == 0 or not np.isfinite(cut).any():
        return x0, y0
    yy, xx = np.mgrid[y1:y2, x1:x2]
    m = np.isfinite(cut)
    flux = cut[m]
    if flux.size == 0 or np.nansum(flux) == 0:
        return x0, y0
    xc = np.sum(xx[m] * flux) / np.sum(flux)
    yc = np.sum(yy[m] * flux) / np.sum(flux)
    return float(xc), float(yc)


def background_truncated(data, x, y):
    ann = CircularAnnulus([(x, y)], r_in=ANN_IN, r_out=ANN_OUT)
    mask = ann.to_mask(method="exact")[0]
    vals = mask.multiply(data)[mask.data > 0]
    vals = vals[np.isfinite(vals)]
    if vals.size < MIN_ANN_PIXELS:
        return np.nan, np.nan, 0

    med = np.nanmedian(vals)
    sig = robust_sigma(vals)
    if not np.isfinite(sig) or sig <= 0:
        return np.nan, np.nan, 0

    core = vals[(vals >= med - CORE_K*sig) & (vals <= med + CORE_K*sig)]
    if core.size < MIN_CORE_PIXELS:
        core = vals

    B = float(np.nanmean(core))
    sigma_sky = float(robust_sigma(core))
    return B, sigma_sky, int(core.size)


def net_flux_and_err(data, x, y):
    B, sigma_sky, n_core = background_truncated(data, x, y)
    if not (np.isfinite(B) and np.isfinite(sigma_sky) and n_core > 0 and sigma_sky > 0):
        return np.nan, np.nan

    ap = CircularAperture([(x, y)], r=R_FIXED)
    raw = float(aperture_photometry(data, ap)["aperture_sum"][0])
    A = float(ap.area)
    net = raw - B * A
    var = (sigma_sky**2) * (A + (A**2)/float(n_core))
    err = float(np.sqrt(var)) if np.isfinite(var) and var > 0 else np.nan
    return net, err


def flux_to_mag_and_err(F, sF):
    ln10 = np.log(10.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = -2.5 * np.log10(F)
        sm = (2.5/ln10) * (sF/F)
    return m, sm


def weighted_mean_and_err(y, s):
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0)
    if ok.sum() == 0:
        return np.nan, np.nan
    w = 1.0/(s[ok]**2)
    mu = float(np.sum(w*y[ok]) / np.sum(w))
    se = float(np.sqrt(1.0/np.sum(w)))
    return mu, se


def wmean(y, s):
    mu, _ = weighted_mean_and_err(y, s)
    return mu


def wrms(y, s, mu):
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0) & np.isfinite(mu)
    if ok.sum() == 0:
        return np.nan
    w = 1.0/(s[ok]**2)
    return float(np.sqrt(np.sum(w*(y[ok]-mu)**2)/np.sum(w)))


def chi2_red_p(y, s, mu):
    ok = np.isfinite(y) & np.isfinite(s) & (s > 0) & np.isfinite(mu)
    n = int(ok.sum())
    if n < 2:
        return np.nan, np.nan
    chi2 = float(np.sum(((y[ok]-mu)/s[ok])**2))
    dof = n - 1
    chi2r = chi2/dof
    if _HAVE_SCIPY:
        p = float(_chi2_dist.sf(chi2, dof))
    else:
        p = np.nan
    return chi2r, p


def weighted_slope(x, y, s):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(s) & (s > 0)
    if ok.sum() < 2:
        return np.nan, np.nan
    xx = x[ok].astype(float)
    yy = y[ok].astype(float)
    ww = 1.0/(s[ok].astype(float)**2)

    Sw = np.sum(ww)
    Sx = np.sum(ww*xx)
    Sy = np.sum(ww*yy)
    Sxx = np.sum(ww*xx*xx)
    Sxy = np.sum(ww*xx*yy)
    denom = Sw*Sxx - Sx*Sx
    if denom <= 0 or not np.isfinite(denom):
        return np.nan, np.nan
    b = (Sw*Sxy - Sx*Sy)/denom
    sb = np.sqrt(Sw/denom)
    return float(b), float(sb)


def main():
    files = sorted(glob.glob(os.path.join(FITS_FOLDER, FITS_GLOB)))
    if not files:
        print("No FITS files found:", FITS_FOLDER)
        return

    recs = []
    for fp in files:
        base = os.path.basename(fp)
        data, hdr = load_first_2d_hdu(fp)
        wcs = WCS(hdr)

        # Time axis (MJD)
        t = None
        if "DATE-OBS" in hdr:
            try:
                t = Time(str(hdr["DATE-OBS"]), format="isot", scale="utc")
            except Exception:
                t = None
        if t is None and "MJD-OBS" in hdr:
            try:
                t = Time(float(hdr["MJD-OBS"]), format="mjd", scale="utc")
            except Exception:
                t = None

        mjd = float(t.mjd) if t is not None else np.nan
        row = {"fits_file": base, "mjd": mjd}

        # Stars 1..5
        for sname, (ra, dec) in STAR_WCS_DEG.items():
            x0, y0 = wcs_to_pixel(wcs, ra, dec)
            xc, yc = centroid_fluxweighted(data, x0, y0, CENTROID_HALF)
            F, sF = net_flux_and_err(data, xc, yc)
            i = int(sname.split()[-1])
            row[f"F{i}"] = F
            row[f"s{i}"] = sF

        # GX
        x0, y0 = wcs_to_pixel(wcs, GX_RA_DEG, GX_DEC_DEG)
        xc, yc = centroid_fluxweighted(data, x0, y0, CENTROID_HALF)
        FGX, sGX = net_flux_and_err(data, xc, yc)
        row["FGX"] = FGX
        row["sGX"] = sGX

        recs.append(row)

    df = pd.DataFrame(recs)
    df = df.sort_values("mjd") if np.isfinite(df["mjd"]).any() else df

    # X-axis
    if np.isfinite(df["mjd"]).any():
        x = df["mjd"].to_numpy(dtype=float)
        xlab = "MJD"
        x_days = x - np.nanmin(x)
    else:
        x = np.arange(len(df), dtype=float)
        xlab = "frame index"
        x_days = x.copy()

    # mags/errors for stars
    m = {}
    sm = {}
    for i in [1, 2, 3, 4, 5]:
        F = df[f"F{i}"].to_numpy(dtype=float)
        sF = df[f"s{i}"].to_numpy(dtype=float)
        ok = np.isfinite(F) & np.isfinite(sF) & (F > 0) & (sF > 0)
        mi, smi = flux_to_mag_and_err(F, sF)
        m[i] = np.where(ok, mi, np.nan)
        # add systematic floor (in mag) for all plots
        smi_tot = np.sqrt(smi**2 + SIGMA_SYS_MAG**2)
        sm[i] = np.where(ok, smi_tot, np.nan)

    # GX mags/errors
    FGX = df["FGX"].to_numpy(dtype=float)
    sGX = df["sGX"].to_numpy(dtype=float)
    ok = np.isfinite(FGX) & np.isfinite(sGX) & (FGX > 0) & (sGX > 0)
    mGX, smGX = flux_to_mag_and_err(FGX, sGX)
    mGX = np.where(ok, mGX, np.nan)
    smGX = np.where(ok, np.sqrt(smGX**2 + SIGMA_SYS_MAG**2), np.nan)

    # ========== FIG 1: Raw instrumental magnitudes (median-subtracted) ==========
    plt.figure(figsize=(10.5, 5.0))
    for i in [1, 2, 3, 4, 5]:
        yy = m[i]
        yy0 = np.nanmedian(yy)
        plt.plot(x, yy - yy0, "o-", label=f"Star {i}")
    plt.xlabel(xlab)
    plt.ylabel("m_inst - median(m_inst)  (mag)")
    plt.title("FIG 1 — Raw instrumental magnitudes (median-subtracted): shared frame-wide drift")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()

    # ========== Ensemble m_ens (all stars) ==========
    m_ens = np.full(len(df), np.nan, dtype=float)
    sm_ens = np.full(len(df), np.nan, dtype=float)
    for k in range(len(df)):
        yy = np.array([m[i][k] for i in [1,2,3,4,5]], dtype=float)
        ss = np.array([sm[i][k] for i in [1,2,3,4,5]], dtype=float)
        mu, se = weighted_mean_and_err(yy, ss)
        m_ens[k] = mu
        sm_ens[k] = se

    # ========== FIG 2: Ensemble common mode ==========
    plt.figure(figsize=(10.5, 4.6))
    plt.errorbar(x, m_ens - np.nanmedian(m_ens), yerr=sm_ens, fmt="o-", capsize=3)
    plt.xlabel(xlab)
    plt.ylabel("m_ens - median(m_ens)  (mag)")
    plt.title("FIG 2 — Ensemble reference magnitude (common-mode systematic component)")
    plt.grid(alpha=0.25)
    plt.show()

    # ========== FIG 3: Leave-one-out residuals with uncertainties ==========
    res = {}
    sres = {}
    for i in [1,2,3,4,5]:
        ri = np.full(len(df), np.nan)
        sri = np.full(len(df), np.nan)
        for k in range(len(df)):
            yi = m[i][k]
            si = sm[i][k]
            if not (np.isfinite(yi) and np.isfinite(si) and si > 0):
                continue
            others = [j for j in [1,2,3,4,5] if j != i]
            y_oth = np.array([m[j][k] for j in others], dtype=float)
            s_oth = np.array([sm[j][k] for j in others], dtype=float)
            mu, se = weighted_mean_and_err(y_oth, s_oth)
            if not (np.isfinite(mu) and np.isfinite(se) and se > 0):
                continue
            ri[k] = yi - mu
            sri[k] = np.sqrt(si**2 + se**2)
        res[i] = ri
        sres[i] = sri

    plt.figure(figsize=(10.8, 5.2))
    for i in [1,2,3,4,5]:
        plt.errorbar(x, res[i], yerr=sres[i], fmt="o-", capsize=3, label=f"Star {i}")
    plt.xlabel(xlab)
    plt.ylabel("r_i = m_i - m_ens,-i  (mag)")
    plt.title(f"FIG 3 — Leave-one-out residuals with uncertainties (σ_sys={SIGMA_SYS_MAG:.3f} mag)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.show()

    # ========== FIG 4: Pairwise Δm for all star pairs ==========
    pairs = [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)]
    fig, axes = plt.subplots(5, 2, figsize=(11.0, 14.0), sharex=True)
    axes = axes.flatten()
    for ax, (i,j) in zip(axes, pairs):
        dm = m[i] - m[j]
        sdm = np.sqrt(sm[i]**2 + sm[j]**2)
        ax.errorbar(x, dm, yerr=sdm, fmt="o", capsize=3)
        ax.set_title(f"Δm = m({i}) - m({j})")
        ax.grid(alpha=0.25)
        ax.set_ylabel("mag")
    for ax in axes[-2:]:
        ax.set_xlabel(xlab)
    fig.suptitle("FIG 4 — Pairwise differential magnitudes between comparison stars", y=0.995)
    plt.tight_layout()
    plt.show()

    # ========== FIG 5: GX vs Star 3 ==========
    dm_gx3 = mGX - m[3]
    sdm_gx3 = np.sqrt(smGX**2 + sm[3]**2)
    plt.figure(figsize=(10.5, 4.8))
    plt.errorbar(x, dm_gx3, yerr=sdm_gx3, fmt="o", capsize=3)
    plt.xlabel(xlab)
    plt.ylabel("Δm = m(GX) - m(Star 3)  (mag)")
    plt.title("FIG 5 — GX 339-4 relative to Star 3 (fixed aperture)")
    plt.grid(alpha=0.25)
    plt.show()

    # ========== FIG 6: GX vs Ensemble ==========
    dm_gx_ens = mGX - m_ens
    sdm_gx_ens = np.sqrt(smGX**2 + sm_ens**2)
    plt.figure(figsize=(10.5, 4.8))
    plt.errorbar(x, dm_gx_ens, yerr=sdm_gx_ens, fmt="o", capsize=3)
    plt.xlabel(xlab)
    plt.ylabel("Δm = m(GX) - m_ens  (mag)")
    plt.title("FIG 6 — GX 339-4 relative to ensemble reference (common-mode reduced)")
    plt.grid(alpha=0.25)
    plt.show()

    # ---------------- CONSOLE TABLES (for write-up text) ----------------
    # Pairwise constancy metrics
    rows = []
    for (i,j) in pairs:
        dm = m[i] - m[j]
        sdm = np.sqrt(sm[i]**2 + sm[j]**2)
        mu = wmean(dm, sdm)
        w = wrms(dm, sdm, mu)
        chi2r, p = chi2_red_p(dm, sdm, mu)
        b, sb = weighted_slope(x_days, dm, sdm)
        sig = (abs(b)/sb) if np.isfinite(b) and np.isfinite(sb) and sb > 0 else np.nan
        rows.append({
            "pair": f"{i}-{j}",
            "n": int(np.isfinite(dm).sum()),
            "wrms_mag": w,
            "chi2_red": chi2r,
            "p_value": p,
            "slope_mag_per_day": b,
            "slope_sig": sig
        })

    out = pd.DataFrame(rows).sort_values("chi2_red", ascending=True)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda v: f"{v: .4g}" if np.isfinite(v) else " NaN")
    print("\nPairwise constancy metrics (smaller chi2_red is better):")
    print(out.to_string(index=False))

    print("\nLeave-one-out residual summary (with plotted total uncertainties):")
    for i in [1,2,3,4,5]:
        ok = np.isfinite(res[i]) & np.isfinite(sres[i]) & (sres[i] > 0)
        std_r = float(np.nanstd(res[i][ok])) if ok.sum() > 1 else np.nan
        med_s = float(np.nanmedian(sres[i][ok])) if ok.sum() > 0 else np.nan
        frac_out = float(np.mean(np.abs(res[i][ok]) > sres[i][ok])) if ok.sum() > 0 else np.nan
        print(f"Star {i}: std(res)={std_r:.4g} mag | median σ={med_s:.4g} mag | frac(|res|>σ)={frac_out:.3f}")

    print("\nInterpretation:")
    print("- FIG 1–2 demonstrate strong common-mode (frame-wide) systematics across nights.")
    print("- FIG 3 tests star stability after removing common-mode using leave-one-out ensemble.")
    print("- FIG 4 shows all pairwise differential magnitudes (consistency check).")
    print("- FIG 5 is the requested GX vs Star 3 light curve.")
    print("- FIG 6 is a more robust GX light curve using the ensemble reference.")
    print(f"- Uncertainties shown include a systematic floor of {SIGMA_SYS_MAG:.3f} mag (set to 0 for photon-only).")


if __name__ == "__main__":
    main()