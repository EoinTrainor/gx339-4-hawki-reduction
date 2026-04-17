# title: GX 339-4 (Heida ephemeris) — Segments+Gaps timebar + 13 labelled FITS locator images

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# ===================== USER INPUTS =====================

FITS_FOLDER = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Visualising fits files/Science Images/SI_Chronologic_DATE_OBS/"

SEG_GAP_TABLE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Observed Orbital Phase/Analysis Outputs/8) gx3394_orbit_segments_and_gaps.csv"

OUT_FOLDER = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Phase Correlated Locator Images"
os.makedirs(OUT_FOLDER, exist_ok=True)

# Only these 13 FITS files (your list)
FITS_LIST = [
    "2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits",
    "2025-06-02_06-15-32.062000__ADP.2025-07-08T07-31-50.422.fits",
    "2025-06-04_05-00-10.244100__ADP.2025-07-08T07-33-00.522.fits",
    "2025-06-19_01-33-43.189300__ADP.2025-07-08T07-36-59.578.fits",
    "2025-07-03_02-45-26.226800__ADP.2025-08-05T08-42-03.245.fits",
    "2025-07-14_04-28-20.784700__ADP.2025-08-05T08-45-21.059.fits",
    "2025-07-18_03-13-50.850700__ADP.2025-08-05T08-46-56.121.fits",
    "2025-07-20_01-41-02.141300__ADP.2025-08-05T08-47-32.925.fits",
    "2025-07-23_00-16-39.830100__ADP.2025-08-05T08-49-02.085.fits",
    "2025-07-23_00-38-06.741600__ADP.2025-08-05T08-49-02.089.fits",
    "2025-08-14_23-50-53.625400__ADP.2025-09-04T08-49-42.619.fits",
    "2025-08-29_23-41-47.883200__ADP.2025-09-04T08-52-18.504.fits",
    "2025-09-19_23-37-04.694700__ADP.2025-10-07T08-32-26.881.fits",
]

# GX 339-4 coordinates (Gandhi+ 2010)
GX339_RA_DEG  = 255.7057818297
GX339_DEC_DEG = -48.7897466540

# Locator settings
ZOOM_HALF_SIZE_PX = 250
FIND_HALF_SIZE_PX = 60
PLO, PHI = 1, 99

# ---------------- Heida et al. ephemeris ----------------
# Use the same ones you’re referencing:
P_DAYS   = 1.7587
T0_MJD   = 57529.397
# --------------------------------------------------------

# =======================================================


# ===================== FITS time + phase =====================

def find_date_obs(hdul):
    if "DATE-OBS" in hdul[0].header:
        return str(hdul[0].header["DATE-OBS"]).strip()
    for hdu in hdul[1:]:
        if hdu.header is not None and "DATE-OBS" in hdu.header:
            return str(hdu.header["DATE-OBS"]).strip()
    return None

def get_mjd_and_dateobs(fits_path):
    with fits.open(fits_path, memmap=False) as hdul:
        d = find_date_obs(hdul)
    if d is None:
        raise RuntimeError(f"DATE-OBS not found in: {fits_path}")
    t = Time(d, format="isot", scale="utc")
    return float(t.mjd), d

def mjd_to_phase(mjd, T0_mjd, P_days):
    return float(((mjd - T0_mjd) / P_days) % 1.0)


# ===================== Locator code (integrated) =====================

def robust_limits(img, p_lo=1, p_hi=99):
    m = np.isfinite(img)
    if not np.any(m):
        return None, None
    vmin = np.nanpercentile(img[m], p_lo)
    vmax = np.nanpercentile(img[m], p_hi)
    if vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax

def reduce_to_2d(data):
    a = np.array(data)
    while a.ndim > 2:
        a = a[0]
    return a

def first_2d_image_hdu(hdul):
    for hdu in hdul:
        if hdu.data is None:
            continue
        img = reduce_to_2d(hdu.data)
        if img.ndim == 2:
            return img, hdu.header
    return None, None

def make_cutout(img, x, y, half):
    ny, nx = img.shape
    cx = int(round(x))
    cy = int(round(y))
    x0 = max(cx - half, 0)
    x1 = min(cx + half + 1, nx)
    y0 = max(cy - half, 0)
    y1 = min(cy + half + 1, ny)
    return img[y0:y1, x0:x1], (x0, x1, y0, y1)

def pixel_scale_arcsec_per_pix(hdr):
    if all(k in hdr for k in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]):
        cd11 = float(hdr["CD1_1"]); cd12 = float(hdr["CD1_2"])
        cd21 = float(hdr["CD2_1"]); cd22 = float(hdr["CD2_2"])
        sx = np.sqrt(cd11**2 + cd21**2)
        sy = np.sqrt(cd12**2 + cd22**2)
        return float(((sx + sy) / 2.0) * 3600.0)
    if all(k in hdr for k in ["CDELT1", "CDELT2"]):
        return float((abs(hdr["CDELT1"]) + abs(hdr["CDELT2"])) / 2.0 * 3600.0)
    return None

def add_scale_bar(ax, arcsec_per_pix, length_arcsec=5.0, pad_px=10):
    if arcsec_per_pix is None or arcsec_per_pix <= 0:
        return
    length_px = length_arcsec / arcsec_per_pix
    x0, y0 = pad_px, pad_px
    x1 = x0 + length_px
    ax.plot([x0, x1], [y0, y0], lw=3)
    ax.text((x0 + x1)/2.0, y0 + 6, f'{length_arcsec:g}"', ha="center", va="bottom", fontsize=10)

def save_locator_image(fits_path, out_png_path, label_text, date_obs_text, phase):
    gx = SkyCoord(GX339_RA_DEG * u.deg, GX339_DEC_DEG * u.deg, frame="icrs")

    with fits.open(fits_path, memmap=False) as hdul:
        img, hdr = first_2d_image_hdu(hdul)
        if img is None:
            raise RuntimeError(f"No 2D image HDU found in: {fits_path}")
        if "CTYPE1" not in hdr or "CTYPE2" not in hdr:
            raise RuntimeError(f"No usable WCS in: {fits_path}")

        w = WCS(hdr)
        x, y = w.world_to_pixel(gx)
        x = float(x); y = float(y)

        ny, nx = img.shape
        if not (0 <= x < nx and 0 <= y < ny):
            raise RuntimeError(f"GX pixel position out of bounds for: {fits_path}")

        cut_zoom, (zx0, zx1, zy0, zy1) = make_cutout(img, x, y, ZOOM_HALF_SIZE_PX)
        cut_find, (fx0, fx1, fy0, fy1) = make_cutout(img, x, y, FIND_HALF_SIZE_PX)

        vmin_full, vmax_full = robust_limits(img, PLO, PHI)
        vmin_zoom, vmax_zoom = robust_limits(cut_zoom, PLO, PHI)
        vmin_find, vmax_find = robust_limits(cut_find, PLO, PHI)

        arcsec_per_pix = pixel_scale_arcsec_per_pix(hdr)

        fig, axes = plt.subplots(1, 3, figsize=(21, 7))
        fig.suptitle(
            f"{label_text} | DATE-OBS: {date_obs_text} | phase={phase:.6f}",
            fontsize=16, y=0.98
        )
        fig.text(0.5, 0.955, os.path.basename(fits_path), ha="center", va="top", fontsize=11)

        ax = axes[0]
        ax.imshow(img, origin="lower", cmap="gray", vmin=vmin_full, vmax=vmax_full)
        ax.plot(x, y, marker="x", markersize=12, mew=2)
        ax.set_title("Mosaic (full frame)")
        ax.set_xlabel("X (pix)")
        ax.set_ylabel("Y (pix)")
        ax.plot([zx0, zx1, zx1, zx0, zx0], [zy0, zy0, zy1, zy1, zy0], linewidth=1.5)

        ax2 = axes[1]
        ax2.imshow(cut_zoom, origin="lower", cmap="gray", vmin=vmin_zoom, vmax=vmax_zoom)
        ax2.plot(x - zx0, y - zy0, marker="x", markersize=12, mew=2)
        ax2.set_title(f"Zoom (±{ZOOM_HALF_SIZE_PX} px)")
        ax2.set_xlabel("X (cutout pix)")
        ax2.set_ylabel("Y (cutout pix)")

        ax3 = axes[2]
        ax3.imshow(cut_find, origin="lower", cmap="gray", vmin=vmin_find, vmax=vmax_find)
        ax3.set_title(f"Finding chart (±{FIND_HALF_SIZE_PX} px)")
        ax3.set_xlabel("X (cutout pix)")
        ax3.set_ylabel("Y (cutout pix)")
        cx, cy = x - fx0, y - fy0
        ax3.plot(cx, cy, marker="+", markersize=18, mew=2)
        ax3.axhline(cy, linewidth=1)
        ax3.axvline(cx, linewidth=1)
        add_scale_bar(ax3, arcsec_per_pix, length_arcsec=5.0, pad_px=10)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig.savefig(out_png_path, dpi=300)
        plt.close(fig)


# ===================== Timebar from segments+gaps table =====================

def draw_timebar_segments_gaps(seg_gap_csv, phases, labels, out_png):
    df = pd.read_csv(seg_gap_csv)

    for c in ["Status", "Phase_start", "Phase_end"]:
        if c not in df.columns:
            raise KeyError(f"Missing column {c} in {seg_gap_csv}")

    fig, ax = plt.subplots(figsize=(14, 2.4))

    y = 0.5
    h = 0.55

    # Draw each row (wrap-aware)
    for _, r in df.iterrows():
        status = str(r["Status"]).strip().lower()
        p0 = float(r["Phase_start"])
        p1 = float(r["Phase_end"])
        color = "green" if status == "observed" else "red"

        if p1 >= p0:
            ax.broken_barh([(p0, p1 - p0)], (y - h/2, h),
                           facecolors=color, edgecolors="black", linewidth=1)
        else:
            # wrap case
            ax.broken_barh([(p0, 1.0 - p0)], (y - h/2, h),
                           facecolors=color, edgecolors="black", linewidth=1)
            ax.broken_barh([(0.0, p1 - 0.0)], (y - h/2, h),
                           facecolors=color, edgecolors="black", linewidth=1)

    # Overlay numbered markers for your 13 FITS (01–13)
    for ph, lab in zip(phases, labels):
        ax.scatter([ph], [y + 0.48], s=85, edgecolors="black", linewidths=1.0, zorder=5)
        ax.text(ph, y + 0.62, lab, ha="center", va="bottom", fontsize=9, weight="bold", zorder=6)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, 1, 11))
    ax.grid(axis="x", alpha=0.25)
    ax.set_xlabel("Orbital Phase (0 → 1)")

    ax.set_title(f"GX 339-4 — Observed (green) vs Unobserved (red) Orbit Coverage (P = {P_DAYS} d)")

    ax.legend(handles=[
        Patch(facecolor="green", edgecolor="black", label="Observed (Segment)"),
        Patch(facecolor="red", edgecolor="black", label="Unobserved (Gap)"),
    ], loc="upper right")

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    plt.close(fig)


# ===================== Main =====================

def main():
    fits_paths = [os.path.join(FITS_FOLDER, f) for f in FITS_LIST]

    # Read MJD + DATE-OBS from headers
    mjds = []
    dateobs = []
    for fp in fits_paths:
        m, d = get_mjd_and_dateobs(fp)
        mjds.append(m)
        dateobs.append(d)

    # Sort timewise by MJD
    order = np.argsort(mjds)
    fits_paths = [fits_paths[i] for i in order]
    mjds       = [mjds[i] for i in order]
    dateobs    = [dateobs[i] for i in order]

    # Compute phases using Heida ephemeris
    phases = [mjd_to_phase(m, T0_MJD, P_DAYS) for m in mjds]

    # Labels 01..13 in timewise order
    labels = [f"{i+1:02d}" for i in range(len(fits_paths))]

    # Output 1: timebar like your reference image
    timebar_path = os.path.join(OUT_FOLDER, "00_GX339-4_Timebar_Heida_SegmentsGaps_Labelled_01-13.png")
    draw_timebar_segments_gaps(SEG_GAP_TABLE, phases, labels, timebar_path)
    print("Saved:", timebar_path)

    # Outputs 2–14: locator images
    for i, fp in enumerate(fits_paths):
        out_img = os.path.join(OUT_FOLDER, f"{labels[i]}_GX339-4_Locator.png")
        save_locator_image(fp, out_img, f"Image {labels[i]} of 13", dateobs[i], phases[i])
        print("Saved:", out_img)

    print("\nDone. Total outputs:", 1 + len(fits_paths))
    print("Folder:", OUT_FOLDER)


if __name__ == "__main__":
    main()
