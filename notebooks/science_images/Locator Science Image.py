# title: GX 339-4 Locator in HAWK-I Science Mosaic (full + zoom + finding chart + WCS check)

import os
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u


# ---------------- USER INPUTS ----------------
FITS_FOLDER = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS"
FITS_FILE = FITS_FOLDER + r"/2025-05-17_05-45-20.232300__ADP.2025-06-04T07-48-45.944.fits"

GX339_RA_DEG  = 255.7057818297
GX339_DEC_DEG = -48.7897466540

ZOOM_HALF_SIZE_PX = 250          # main zoom (±px)
FIND_HALF_SIZE_PX = 60           # finding chart cutout (±px)

PLO, PHI = 1, 99
# --------------------------------------------


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


def find_date_obs_any_hdu(hdul):
    if "DATE-OBS" in hdul[0].header:
        return str(hdul[0].header["DATE-OBS"]).strip()
    for hdu in hdul[1:]:
        if hdu.header is not None and "DATE-OBS" in hdu.header:
            return str(hdu.header["DATE-OBS"]).strip()
    return "DATE-OBS not found"


def first_2d_image_hdu(hdul):
    for i, hdu in enumerate(hdul):
        if hdu.data is None:
            continue
        img = reduce_to_2d(hdu.data)
        if img.ndim == 2:
            return i, img, hdu.header
    return None


def wcs_summary(header):
    keys = ["CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2",
            "CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2", "EXPTIME"]
    out = {}
    for k in keys:
        if k in header:
            out[k] = header.get(k)
    return out


def make_cutout(img, x, y, half):
    ny, nx = img.shape
    cx = int(round(x))
    cy = int(round(y))

    x0 = max(cx - half, 0)
    x1 = min(cx + half + 1, nx)
    y0 = max(cy - half, 0)
    y1 = min(cy + half + 1, ny)

    cut = img[y0:y1, x0:x1]
    return cut, (x0, x1, y0, y1)


def pixel_scale_arcsec_per_pix(hdr):
    """
    Estimate pixel scale from CD matrix (robust for rotated images).
    Returns mean arcsec/pix if CD present, else None.
    """
    if all(k in hdr for k in ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]):
        cd11 = float(hdr["CD1_1"]); cd12 = float(hdr["CD1_2"])
        cd21 = float(hdr["CD2_1"]); cd22 = float(hdr["CD2_2"])
        # scale in deg/pix along each axis
        sx = np.sqrt(cd11**2 + cd21**2)
        sy = np.sqrt(cd12**2 + cd22**2)
        return float(((sx + sy) / 2.0) * 3600.0)  # arcsec/pix
    if all(k in hdr for k in ["CDELT1", "CDELT2"]):
        return float((abs(hdr["CDELT1"]) + abs(hdr["CDELT2"])) / 2.0 * 3600.0)
    return None


def add_compass_NE(ax, wcs_obj, x0, y0, cut_shape, length_px=40):
    """
    Add N/E arrows to a cutout axis.
    x0,y0 are the cutout origin in full-image pixels.
    """
    h, w = cut_shape
    cx = x0 + w / 2.0
    cy = y0 + h / 2.0

    # centre sky coordinate
    c_sky = wcs_obj.pixel_to_world(cx, cy)

    # move a small amount in Dec for North, RA for East
    # Use 10 arcsec step; direction is what matters
    step = 10.0 * u.arcsec
    north_sky = SkyCoord(c_sky.ra, c_sky.dec + step, frame="icrs")
    east_sky  = SkyCoord(c_sky.ra + (step / np.cos(c_sky.dec.to(u.rad).value)), c_sky.dec, frame="icrs")

    nx, ny = wcs_obj.world_to_pixel(north_sky)
    ex, ey = wcs_obj.world_to_pixel(east_sky)

    # vectors in pixel space
    vN = np.array([nx - cx, ny - cy], dtype=float)
    vE = np.array([ex - cx, ey - cy], dtype=float)

    # normalise to desired length
    def norm_to(v, L):
        n = np.hypot(v[0], v[1])
        if n == 0:
            return v
        return v / n * L

    vN = norm_to(vN, length_px)
    vE = norm_to(vE, length_px)

    # place arrows near bottom-left of cutout
    base_x = 10
    base_y = 10

    ax.annotate("", xy=(base_x + vN[0], base_y + vN[1]), xytext=(base_x, base_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(base_x + vN[0] + 4, base_y + vN[1] + 4, "N", fontsize=11, weight="bold")

    ax.annotate("", xy=(base_x + vE[0], base_y + vE[1]), xytext=(base_x, base_y),
                arrowprops=dict(arrowstyle="->", lw=2))
    ax.text(base_x + vE[0] + 4, base_y + vE[1] + 4, "E", fontsize=11, weight="bold")


def add_scale_bar(ax, arcsec_per_pix, length_arcsec=5.0, pad_px=10):
    """
    Draw a simple scale bar of given length (arcsec) on the cutout.
    """
    if arcsec_per_pix is None or arcsec_per_pix <= 0:
        return

    length_px = length_arcsec / arcsec_per_pix

    x0 = pad_px
    y0 = pad_px
    x1 = x0 + length_px

    ax.plot([x0, x1], [y0, y0], lw=3)
    ax.text((x0 + x1) / 2.0, y0 + 6, f"{length_arcsec:g}\"", ha="center", va="bottom", fontsize=10)


def main():
    if not os.path.exists(FITS_FILE):
        raise FileNotFoundError(f"File not found:\n{FITS_FILE}")

    gx = SkyCoord(GX339_RA_DEG * u.deg, GX339_DEC_DEG * u.deg, frame="icrs")

    with fits.open(FITS_FILE, memmap=False) as hdul:
        date_obs = find_date_obs_any_hdu(hdul)

        result = first_2d_image_hdu(hdul)
        if result is None:
            raise RuntimeError("No 2D image HDU found in this FITS file.")

        idx, img, hdr = result

        if "CTYPE1" not in hdr or "CTYPE2" not in hdr:
            raise RuntimeError("No usable WCS found (missing CTYPE1/CTYPE2).")

        w = WCS(hdr)

        x, y = w.world_to_pixel(gx)
        x = float(x); y = float(y)

        ny, nx = img.shape
        in_bounds = (0 <= x < nx) and (0 <= y < ny)

        print("\n================ GX 339-4 MOSAIC WCS REPORT ================\n")
        print(f"File: {os.path.basename(FITS_FILE)}")
        print(f"DATE-OBS: {date_obs}")
        print(f"Mosaic image HDU: HDU[{idx}]")
        print(f"Image shape: (ny, nx)=({ny}, {nx})\n")

        print(f"Target (ICRS): RA={GX339_RA_DEG:.10f} deg, Dec={GX339_DEC_DEG:.10f} deg")
        print(f"WCS world_to_pixel -> x={x:.3f}, y={y:.3f}   in_bounds={in_bounds}")

        recon = w.pixel_to_world(x, y)
        recon_sc = SkyCoord(recon.ra, recon.dec, frame="icrs")

        dra_arcsec = (recon_sc.ra - gx.ra).to(u.arcsec).value * np.cos(gx.dec.to(u.rad).value)
        ddec_arcsec = (recon_sc.dec - gx.dec).to(u.arcsec).value
        sep_arcsec = float(gx.separation(recon_sc).to(u.arcsec).value)

        print("\nWCS self-consistency check (pixel -> sky -> compare):")
        print(f"  Recovered sky: RA={recon_sc.ra.deg:.10f} deg, Dec={recon_sc.dec.deg:.10f} deg")
        print(f"  ΔRA*cos(Dec) = {dra_arcsec:+.3f} arcsec")
        print(f"  ΔDec         = {ddec_arcsec:+.3f} arcsec")
        print(f"  Separation   = {sep_arcsec:.3f} arcsec")

        print("\nKey WCS keywords (if present):")
        for k, v in wcs_summary(hdr).items():
            print(f"  {k:6s} = {v}")

        if not in_bounds:
            print("\n❌ The computed pixel position is outside the mosaic bounds.")
            return

        # Cutouts
        cut_zoom, (zx0, zx1, zy0, zy1) = make_cutout(img, x, y, ZOOM_HALF_SIZE_PX)
        cut_find, (fx0, fx1, fy0, fy1) = make_cutout(img, x, y, FIND_HALF_SIZE_PX)

        vmin_full, vmax_full = robust_limits(img, PLO, PHI)
        vmin_zoom, vmax_zoom = robust_limits(cut_zoom, PLO, PHI)
        vmin_find, vmax_find = robust_limits(cut_find, PLO, PHI)

        arcsec_per_pix = pixel_scale_arcsec_per_pix(hdr)

        # Plot full + zoom + finding chart
        fig, axes = plt.subplots(1, 3, figsize=(21, 7))

        fig.suptitle(f"DATE-OBS: {date_obs}", fontsize=16, y=0.98)
        fig.text(0.5, 0.955, os.path.basename(FITS_FILE), ha="center", va="top", fontsize=11)

        # Full mosaic
        ax = axes[0]
        ax.imshow(img, origin="lower", cmap="gray", vmin=vmin_full, vmax=vmax_full)
        ax.plot(x, y, marker="x", markersize=12, mew=2)
        ax.set_title("Mosaic (full frame)")
        ax.set_xlabel("X (pix)")
        ax.set_ylabel("Y (pix)")

        # Draw zoom box
        rect_x = [zx0, zx1, zx1, zx0, zx0]
        rect_y = [zy0, zy0, zy1, zy1, zy0]
        ax.plot(rect_x, rect_y, linewidth=1.5)

        # Zoom panel
        ax2 = axes[1]
        ax2.imshow(cut_zoom, origin="lower", cmap="gray", vmin=vmin_zoom, vmax=vmax_zoom)
        ax2.plot(x - zx0, y - zy0, marker="x", markersize=12, mew=2)
        ax2.set_title(f"Zoom (±{ZOOM_HALF_SIZE_PX} px)")
        ax2.set_xlabel("X (cutout pix)")
        ax2.set_ylabel("Y (cutout pix)")

        # Finding chart panel
        ax3 = axes[2]
        ax3.imshow(cut_find, origin="lower", cmap="gray", vmin=vmin_find, vmax=vmax_find)
        ax3.set_title(f"Finding chart (±{FIND_HALF_SIZE_PX} px)")
        ax3.set_xlabel("X (cutout pix)")
        ax3.set_ylabel("Y (cutout pix)")

        # Crosshair on finding chart
        cx = x - fx0
        cy = y - fy0
        ax3.plot(cx, cy, marker="+", markersize=18, mew=2)
        ax3.axhline(cy, linewidth=1)
        ax3.axvline(cx, linewidth=1)

        # Add N/E and scale bar to finding chart
        add_compass_NE(ax3, w, fx0, fy0, cut_find.shape, length_px=40)
        add_scale_bar(ax3, arcsec_per_pix, length_arcsec=5.0, pad_px=10)

        # Add a small text box with pixel coords and scale
        scale_txt = f"x={x:.2f}, y={y:.2f}"
        if arcsec_per_pix is not None:
            scale_txt += f"\nscale ≈ {arcsec_per_pix:.3f}\"/pix"
        ax3.text(0.98, 0.98, scale_txt, ha="right", va="top", transform=ax3.transAxes,
                 fontsize=10, bbox=dict(facecolor="black", alpha=0.5, edgecolor="none"), color="white")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()


if __name__ == "__main__":
    main()
