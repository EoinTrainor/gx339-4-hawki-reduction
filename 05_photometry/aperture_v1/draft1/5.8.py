# title: Stage 4 (Single Frame) – Overlays for 3 Comparison Stars + GX, and GX Annulus High Clip + SNR3

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry


# ---------------- USER INPUTS ----------------
FITS_FILE = (
    r"C:/Users/40328449/OneDrive - University College Cork/"
    r"GX 339-4/SI_Chronologic_DATE_OBS/"
    r"2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"
)

# GX 339-4 (decimal degrees)
GX_RA_DEG  = 255.7057818297
GX_DEC_DEG = -48.7897466540

# Comparison stars (your WCS values)
STARS = {
    "Star 1": (255.707140, -48.788802),
    "Star 2": (255.711256, -48.791347),
    "Star 3": (255.711298, -48.788757),
}

# Geometry (based on your validated SNR test)
AP_R_PX = 8.0
ANN_IN_PX = 15.0
ANN_OUT_PX = 30.0

# Centroiding
CENTROID_HALF_SIZE = 10

# Cutout display
CUTOUT_HALF_SIZE = 60
PERCENTILE_STRETCH = (5, 99.7)

# GX annulus clipping (high-side only)
CLIP_K = 2.0  # clip threshold = median + k*std (raw annulus)


# ---------------- HELPERS ----------------
def load_first_2d_hdu(fp: str):
    with fits.open(fp) as hdul:
        for h in hdul:
            if getattr(h, "data", None) is not None and h.data is not None and np.ndim(h.data) == 2:
                data = h.data.astype(float)
                header = h.header
                break
    data[~np.isfinite(data)] = np.nan
    return data, WCS(header), header

def wcs_to_pixel(wcs: WCS, ra_deg: float, dec_deg: float):
    c = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    x, y = wcs.world_to_pixel(c)
    return float(np.squeeze(x)), float(np.squeeze(y))

def centroid_local_fluxweighted(data, x0, y0, half):
    """Flux-weighted centroid in a local square window around (x0,y0)."""
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

def annulus_values(data, x, y, r_in, r_out):
    ann = CircularAnnulus([(x, y)], r_in=r_in, r_out=r_out)
    m = ann.to_mask(method="exact")[0]
    arr = m.multiply(data)
    vals = arr[m.data > 0]
    return vals[np.isfinite(vals)], m

def compute_snr3(raw_sum, ap_area, B, sky_sigma, n_ann_kept):
    """
    SNR3:
      F_net = raw_sum - B*Npix
      var = sky_sigma^2 * (Npix + Npix^2 / Nann)
      SNR = F_net / sqrt(var)
    """
    if not (np.isfinite(raw_sum) and np.isfinite(ap_area) and np.isfinite(B) and np.isfinite(sky_sigma)):
        return np.nan, np.nan, np.nan

    if ap_area <= 0 or sky_sigma <= 0 or n_ann_kept <= 0:
        return np.nan, np.nan, np.nan

    F_net = float(raw_sum - B * ap_area)
    var = (sky_sigma ** 2) * (ap_area + (ap_area ** 2) / float(n_ann_kept))
    if not np.isfinite(var) or var <= 0:
        return F_net, np.nan, np.nan

    F_err = float(np.sqrt(var))
    snr = float(F_net / F_err) if F_err > 0 else np.nan
    return F_net, F_err, snr

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
    """
    Trimmed standard deviation for diagnostics.
    Trims trim_frac from BOTH tails (default 5% low + 5% high) then std().
    """
    v = v[np.isfinite(v)]
    if v.size < 20:
        return float(np.nanstd(v))
    lo, hi = np.nanquantile(v, [trim_frac, 1.0 - trim_frac])
    vt = v[(v >= lo) & (v <= hi)]
    if vt.size < 20:
        return float(np.nanstd(v))
    return float(np.nanstd(vt))

def clip_high_side(vals, k):
    """
    STAGE 4 DEFINITIONS (LOCKED)

    Geometry:
      - Aperture: r = AP_R_PX
      - Annulus: r_in = ANN_IN_PX, r_out = ANN_OUT_PX

    High-side clip rule:
      - Compute robust sigma on RAW annulus pixels:
          sigma_raw = 1.4826 * MAD(raw)
      - Threshold:
          thresh = median(raw) + k * sigma_raw
      - Keep:
          kept = raw[raw <= thresh]

    Background level:
      - B = median(kept)

    Noise term for SNR3:
      - sky_sigma = 1.4826 * MAD(kept)

    Diagnostics (NOT used in SNR):
      - raw_std_diag = std(raw)
      - raw_std_trim = trimmed std(raw)
      - kept_std_diag = std(kept)
      - kept_std_trim = trimmed std(kept)
      - min/max for raw and kept (sanity)
    """
    v = vals[np.isfinite(vals)]
    if v.size == 0:
        ...
    raw_med0 = float(np.nanmedian(v))
    raw_sig0 = robust_sigma_mad(v)
    # Remove nonphysical low pixels (mask artefacts / padding / bad pixels)
    low_cut = raw_med0 - 10.0 * raw_sig0
    v = v[v >= low_cut]
    if v.size == 0:
        v = vals[np.isfinite(vals)]  # fallback to original finite if filter wipes it

    if v.size == 0:
        return (np.nan, np.nan, np.nan, np.array([]),
                np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan, np.nan, np.nan)

    raw_med = float(np.nanmedian(v))
    raw_sig = robust_sigma_mad(v)
    thresh = raw_med + k * raw_sig

    kept = v[v <= thresh]
    if kept.size < 50:
        # fail-safe: disable clipping if it becomes unstable
        kept = v.copy()
        thresh = np.nan

    B = float(np.nanmedian(kept))
    sky_sigma = robust_sigma_mad(kept)

    # Diagnostics
    raw_std_diag = float(np.nanstd(v))
    raw_std_trim = trimmed_std(v, trim_frac=0.05)
    kept_std_diag = float(np.nanstd(kept))
    kept_std_trim = trimmed_std(kept, trim_frac=0.05)

    raw_min, raw_max = float(np.nanmin(v)), float(np.nanmax(v))
    kept_min, kept_max = float(np.nanmin(kept)), float(np.nanmax(kept))

    return (B, sky_sigma, float(thresh), kept,
            raw_med, raw_sig,
            raw_std_diag, raw_std_trim, kept_std_diag, kept_std_trim,
            raw_min, raw_max, kept_min, kept_max)



# ---------------- LOAD IMAGE ----------------
data, wcs, header = load_first_2d_hdu(FITS_FILE)

print("\n==============================")
print("Processing:", FITS_FILE)
print("==============================")

# ---------------- STAGE 3h: OVERLAYS FOR COMPARISON STARS ----------------
def show_overlay(name, ra, dec, ap_r=AP_R_PX, ann_in=ANN_IN_PX, ann_out=ANN_OUT_PX):
    x_guess, y_guess = wcs_to_pixel(wcs, ra, dec)
    x_cen, y_cen = centroid_local_fluxweighted(data, x_guess, y_guess, CENTROID_HALF_SIZE)

    sub, xoff, yoff = extract_cutout(data, x_cen, y_cen, CUTOUT_HALF_SIZE)
    xc, yc = x_cen - xoff, y_cen - yoff

    vmin, vmax = safe_percentile_limits(sub, *PERCENTILE_STRETCH)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(sub, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)

    CircularAperture([(xc, yc)], r=ap_r).plot(ax=ax, color="yellow", lw=2.0)
    CircularAnnulus([(xc, yc)], r_in=ann_in, r_out=ann_out).plot(ax=ax, color="cyan", lw=1.6)
    ax.plot([xc], [yc], "r+", ms=12, mew=2)

    ax.set_title(f"{name} overlay (ap={ap_r:.0f}px, ann={ann_in:.0f}–{ann_out:.0f}px)")
    ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Return centroid for possible later use
    return x_cen, y_cen

# show the three comparison stars
for nm, (ra, dec) in STARS.items():
    show_overlay(nm, ra, dec)

# ---------------- STAGE 4: GX OVERLAY + GX ANNULUS HIST + CLIP + SIDE-BY-SIDE RAW VS CLEANED DISPLAY ----------------
gx_x_guess, gx_y_guess = wcs_to_pixel(wcs, GX_RA_DEG, GX_DEC_DEG)
gx_x, gx_y = centroid_local_fluxweighted(data, gx_x_guess, gx_y_guess, CENTROID_HALF_SIZE)

print("\nGX 339-4:")
print(f"  WCS guess  x={gx_x_guess:.2f}, y={gx_y_guess:.2f}")
print(f"  Centroid   x={gx_x:.2f}, y={gx_y:.2f}")
print(f"  Offset     dx={gx_x-gx_x_guess:+.2f}, dy={gx_y-gx_y_guess:+.2f}")

# Photometry on original data
ap = CircularAperture([(gx_x, gx_y)], r=AP_R_PX)
ann = CircularAnnulus([(gx_x, gx_y)], r_in=ANN_IN_PX, r_out=ANN_OUT_PX)

ap_tbl = aperture_photometry(data, ap)
raw_sum = float(ap_tbl["aperture_sum"][0])
ap_area = float(ap.area)

ann_vals, ann_mask_obj = annulus_values(data, gx_x, gx_y, ANN_IN_PX, ANN_OUT_PX)

# Clip high side and compute background stats from clipped values
(B, sky_sigma, thresh, kept_vals,
 raw_med, raw_sig,
 raw_std_diag, raw_std_trim, kept_std_diag, kept_std_trim,
 raw_min, raw_max, kept_min, kept_max) = clip_high_side(ann_vals, CLIP_K)

n_ann_raw = int(ann_vals.size)
n_ann_kept = int(kept_vals.size)
clip_frac = 1.0 - (n_ann_kept / n_ann_raw) if n_ann_raw > 0 else np.nan

net_flux, net_err, snr = compute_snr3(raw_sum, ap_area, B, sky_sigma, n_ann_kept)

print("\nGX background + SNR3 (Stage 4 locked definitions):")
print(f"  Annulus pixels raw  = {n_ann_raw}")
print(f"  Annulus pixels kept = {n_ann_kept} (clip frac={clip_frac:.3f})")

print(f"  RAW  median         = {raw_med:.3f}")
print(f"  RAW  robust sigma   = {raw_sig:.3f}  (MAD-based)")
print(f"  RAW  std diag       = {raw_std_diag:.3f}   | trimmed std (5%) = {raw_std_trim:.3f}")
print(f"  RAW  min/max        = {raw_min:.3f} / {raw_max:.3f}")

if np.isfinite(thresh):
    print(f"  Clip threshold      = median + {CLIP_K}*robust_sigma = {thresh:.3f}")
else:
    print("  Clip threshold      = (no clipping applied)")

print(f"  KEPT B (median)     = {B:.3f}")
print(f"  KEPT robust sigma   = {sky_sigma:.3f}  (MAD-based, used in SNR3)")
print(f"  KEPT std diag       = {kept_std_diag:.3f} | trimmed std (5%) = {kept_std_trim:.3f}")
print(f"  KEPT min/max        = {kept_min:.3f} / {kept_max:.3f}")

print(f"  Raw aperture sum    = {raw_sum:.3f}")
print(f"  Net flux            = {net_flux:.3f}  ± {net_err:.3f}")
print(f"  SNR3                = {snr:.3f}")

# ---- Histogram (GX annulus) with sigma lines ----
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.hist(ann_vals, bins=100, alpha=0.45, label="Raw annulus", density=False)
ax.hist(kept_vals, bins=100, alpha=0.75, label="Kept after high-clip", density=False)

# Plot all +n sigma lines from RAW median/std for readability
for n in range(1, 6):
    ax.axvline(raw_med + n * raw_sig, linestyle=":", lw=1.2,
               label=f"raw med + {n}σ (robust)" if n == 1 else None)
    
for n in range(1, 4):
    ax.axvline(raw_med + n * raw_std_trim, linestyle="--", lw=1.0, alpha=0.6,
               label=f"raw med + {n}σ (trimmed std)" if n == 1 else None)

# Plot threshold and B
if np.isfinite(thresh):
    ax.axvline(thresh, linestyle="--", lw=2.0, label=f"clip thresh (k={CLIP_K})")
ax.axvline(B, linestyle="-", lw=2.0, label="B (median kept)")

ax.set_title("GX 339-4 annulus histogram + high-side clip")
ax.set_xlabel("Annulus pixel value")
ax.set_ylabel("Counts")
ax.legend(frameon=False)
plt.tight_layout()
plt.show()

# ---- Side-by-side cutouts: RAW vs DISPLAY-ONLY CLEANED ANNULUS ----
sub_raw, xoff, yoff = extract_cutout(data, gx_x, gx_y, CUTOUT_HALF_SIZE)
xc, yc = gx_x - xoff, gx_y - yoff

# Build a display-only cleaned copy: replace high pixels in annulus with B
sub_clean = np.array(sub_raw, dtype=float, copy=True)

# Build a full-size annulus mask image in cutout coordinates (same shape as sub_clean)
ann_cut = CircularAnnulus([(xc, yc)], r_in=ANN_IN_PX, r_out=ANN_OUT_PX)
ann_mask_obj = ann_cut.to_mask(method="exact")[0]
ann_mask_img = ann_mask_obj.to_image(sub_clean.shape)  # same shape as sub_clean

# ann_mask_img can be None if the annulus falls outside cutout; guard it
if ann_mask_img is None:
    ann_pixels = np.zeros_like(sub_clean, dtype=bool)
else:
    ann_pixels = (ann_mask_img > 0) & np.isfinite(sub_clean)


if np.isfinite(thresh):
    # Compute high-pixel condition using the SAME threshold defined from full-image annulus values
    high = ann_pixels & (sub_clean > thresh)
    sub_clean[high] = B

# Display stretch shared so frames match visually
vmin, vmax = safe_percentile_limits(sub_raw, *PERCENTILE_STRETCH)

fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2))
for ax, img, title in zip(
    axes,
    [sub_raw, sub_clean],
    ["GX cutout (raw)", "GX cutout (display-only annulus cleaned)"]
):
    ax.imshow(img, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
    CircularAperture([(xc, yc)], r=AP_R_PX).plot(ax=ax, color="yellow", lw=2.0)
    CircularAnnulus([(xc, yc)], r_in=ANN_IN_PX, r_out=ANN_OUT_PX).plot(ax=ax, color="cyan", lw=1.6)
    ax.plot([xc], [yc], "r+", ms=12, mew=2)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()
