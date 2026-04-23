"""
09b_visualise_zogy.py
---------------------
Report-quality figures for the ZOGY difference imaging stage.

Produces four figures saved to DIFF_DIR/report_figures/:
  fig1_reference_image.pdf   — Coadded reference R, full field + GX 339-4 zoom
  fig2_before_after.pdf      — Science frame vs D image at GX 339-4 position
  fig3_sample_diff.pdf       — Best D frame: full field + zoom
  fig4_dstd_timeline.pdf     — Clipped D_std per frame across all OBs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization import ZScaleInterval, ImageNormalize, LinearStretch, SqrtStretch
from astropy.stats import sigma_clipped_stats

from config import DIFF_DIR, ALIGNED_DIR, LOGS_DIR, TARGET_RA, TARGET_DEC

# ── Output directory ──────────────────────────────────────────────────────────
FIG_DIR = DIFF_DIR / "report_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Target pixel position (from reference WCS) ────────────────────────────────
REF_FITS   = DIFF_DIR / "reference_R.fits"
QUALITY_DIR = LOGS_DIR / "zogy" / "quality"

TARGET_X = 750.0   # pixels (0-indexed) — visually confirmed
TARGET_Y = 1010.0

ZOOM_HALF = 80     # half-width of zoom box in pixels

REPRESENTATIVE_OB   = "GX339_Ks_Imaging_1"
REPRESENTATIVE_STEM = "HAWKI.2025-05-17T05_45_20.232_1"   # D_std ≈ 0.92

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
    "font.size":   11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})


def zscale_norm(data, contrast=0.25):
    interval = ZScaleInterval(contrast=contrast)
    vmin, vmax = interval.get_limits(data[np.isfinite(data)])
    return ImageNormalize(vmin=vmin, vmax=vmax, stretch=LinearStretch())


def zoom_slice(cy, cx, half, shape):
    y0 = max(0, int(cy) - half)
    y1 = min(shape[0], int(cy) + half)
    x0 = max(0, int(cx) - half)
    x1 = min(shape[1], int(cx) + half)
    return slice(y0, y1), slice(x0, x1)


def add_target_marker(ax, x, y, color="red", label="GX 339-4"):
    ax.plot(x, y, "+", color=color, ms=14, mew=1.5, zorder=10)
    ax.plot(x, y, "o", color=color, ms=20, mew=1.5, fillstyle="none", zorder=10)
    ax.text(x + 22, y, label, color=color, fontsize=9, va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))


# =============================================================================
# Figure 1 — Reference image
# =============================================================================
def fig1_reference():
    print("Figure 1: reference image …")
    data = fits.getdata(REF_FITS).astype(float)

    fig = plt.figure(figsize=(12, 5.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.05)

    # Full field
    ax1 = fig.add_subplot(gs[0])
    norm = zscale_norm(data, contrast=0.15)
    im = ax1.imshow(data, origin="lower", cmap="inferno", norm=norm,
                    interpolation="nearest")
    add_target_marker(ax1, TARGET_X, TARGET_Y)
    ax1.set_title("Coadded Reference (R)  —  236 frames, Ks band", pad=8)
    ax1.set_xlabel("x (pixels)")
    ax1.set_ylabel("y (pixels)")
    cb = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cb.set_label("ADU")

    # Zoom on GX 339-4
    ys, xs = zoom_slice(TARGET_Y, TARGET_X, ZOOM_HALF, data.shape)
    ax2 = fig.add_subplot(gs[1])
    cut = data[ys, xs]
    norm_z = zscale_norm(cut, contrast=0.3)
    ax2.imshow(cut, origin="lower", cmap="inferno", norm=norm_z,
               interpolation="nearest",
               extent=[xs.start, xs.stop, ys.start, ys.stop])
    add_target_marker(ax2, TARGET_X, TARGET_Y)
    ax2.set_title(f"GX 339-4  (±{ZOOM_HALF} px zoom)", pad=8)
    ax2.set_xlabel("x (pixels)")
    ax2.tick_params(labelleft=False)

    # Draw zoom box on full image
    rect = mpatches.Rectangle((xs.start, ys.start),
                               xs.stop - xs.start, ys.stop - ys.start,
                               linewidth=1.2, edgecolor="cyan", facecolor="none")
    ax1.add_patch(rect)

    fig.suptitle("HAWK-I Ks  |  GX 339-4 Field  |  ZOGY Reference Image",
                 fontsize=13, y=1.01)
    out = FIG_DIR / "fig1_reference_image.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 2 — Before / After comparison
# =============================================================================
def fig2_before_after():
    print("Figure 2: before/after panel …")
    ob   = REPRESENTATIVE_OB
    stem = REPRESENTATIVE_STEM

    sci_path  = ALIGNED_DIR / ob / f"{stem}_cal_aligned.fits"
    diff_path = DIFF_DIR    / ob / f"{stem}_diff.fits"

    if not sci_path.exists():
        print(f"  Science frame not found: {sci_path}"); return
    if not diff_path.exists():
        print(f"  Diff frame not found: {diff_path}"); return

    sci  = fits.getdata(sci_path ).astype(float)
    diff = fits.getdata(diff_path).astype(float)

    ys, xs = zoom_slice(TARGET_Y, TARGET_X, ZOOM_HALF, sci.shape)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    # Panel A — science frame zoom
    cut_sci = sci[ys, xs]
    norm_s  = zscale_norm(cut_sci, contrast=0.25)
    axes[0].imshow(cut_sci, origin="lower", cmap="inferno", norm=norm_s,
                   interpolation="nearest",
                   extent=[xs.start, xs.stop, ys.start, ys.stop])
    add_target_marker(axes[0], TARGET_X, TARGET_Y)
    axes[0].set_title("(a)  Science frame (N)")
    axes[0].set_xlabel("x (pixels)"); axes[0].set_ylabel("y (pixels)")

    # Panel B — difference frame zoom
    cut_d = diff[ys, xs]
    _, _, d_std = sigma_clipped_stats(diff[np.isfinite(diff)], sigma=3.0)
    vmin_d, vmax_d = -5 * d_std, 5 * d_std
    axes[1].imshow(cut_d, origin="lower", cmap="RdBu_r",
                   vmin=vmin_d, vmax=vmax_d,
                   interpolation="nearest",
                   extent=[xs.start, xs.stop, ys.start, ys.stop])
    add_target_marker(axes[1], TARGET_X, TARGET_Y, color="black")
    axes[1].set_title("(b)  Difference image (D)")
    axes[1].set_xlabel("x (pixels)")
    axes[1].tick_params(labelleft=False)

    # Panel C — reference zoom (for context)
    ref = fits.getdata(REF_FITS).astype(float)
    cut_r = ref[ys, xs]
    norm_r = zscale_norm(cut_r, contrast=0.3)
    axes[2].imshow(cut_r, origin="lower", cmap="inferno", norm=norm_r,
                   interpolation="nearest",
                   extent=[xs.start, xs.stop, ys.start, ys.stop])
    add_target_marker(axes[2], TARGET_X, TARGET_Y)
    axes[2].set_title("(c)  Reference (R, 236 frames)")
    axes[2].set_xlabel("x (pixels)")
    axes[2].tick_params(labelleft=False)

    fig.suptitle(
        f"ZOGY Before/After  |  {ob}  |  {stem}\n"
        f"±{ZOOM_HALF} px zoom around GX 339-4",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    out = FIG_DIR / "fig2_before_after.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 3 — Full best difference frame + zoom
# =============================================================================
def fig3_sample_diff():
    print("Figure 3: sample difference image …")
    ob   = REPRESENTATIVE_OB
    stem = REPRESENTATIVE_STEM
    diff_path = DIFF_DIR / ob / f"{stem}_diff.fits"

    if not diff_path.exists():
        print(f"  Diff frame not found: {diff_path}"); return

    diff = fits.getdata(diff_path).astype(float)
    _, _, d_std = sigma_clipped_stats(diff[np.isfinite(diff)], sigma=3.0)

    fig = plt.figure(figsize=(12, 5.5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], wspace=0.05)

    # Full frame
    ax1 = fig.add_subplot(gs[0])
    vspan = 5 * d_std
    im = ax1.imshow(diff, origin="lower", cmap="RdBu_r",
                    vmin=-vspan, vmax=vspan, interpolation="nearest")
    add_target_marker(ax1, TARGET_X, TARGET_Y, color="lime")
    ax1.set_title("Difference image D  (full frame, ±5σ stretch)")
    ax1.set_xlabel("x (pixels)"); ax1.set_ylabel("y (pixels)")
    cb = fig.colorbar(im, ax=ax1, fraction=0.03, pad=0.02)
    cb.set_label("D [σ units approx]")

    # Zoom
    ys, xs = zoom_slice(TARGET_Y, TARGET_X, ZOOM_HALF, diff.shape)
    ax2 = fig.add_subplot(gs[1])
    cut = diff[ys, xs]
    ax2.imshow(cut, origin="lower", cmap="RdBu_r",
               vmin=-vspan, vmax=vspan, interpolation="nearest",
               extent=[xs.start, xs.stop, ys.start, ys.stop])
    add_target_marker(ax2, TARGET_X, TARGET_Y, color="lime", label="GX 339-4")
    ax2.set_title(f"GX 339-4 zoom (±{ZOOM_HALF} px)")
    ax2.set_xlabel("x (pixels)")
    ax2.tick_params(labelleft=False)

    rect = mpatches.Rectangle((xs.start, ys.start),
                               xs.stop - xs.start, ys.stop - ys.start,
                               linewidth=1.2, edgecolor="cyan", facecolor="none")
    ax1.add_patch(rect)

    fig.suptitle(
        f"ZOGY Sample Difference Frame  |  {ob}  |  "
        f"D_std(3σ-clip) = {d_std:.3f}",
        fontsize=13, y=1.01,
    )
    out = FIG_DIR / "fig3_sample_diff.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 4 — D_std timeline across all frames
# =============================================================================
def fig4_dstd_timeline():
    print("Figure 4: D_std timeline …")
    csv_path = QUALITY_DIR / "diff_stats.csv"
    rows = list(csv.DictReader(open(csv_path)))

    ob_order = sorted(set(r["ob"] for r in rows))
    ob_index = {ob: i for i, ob in enumerate(ob_order)}

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Top: D_std_clipped per frame, coloured by OB
    frame_idx = 0
    xticks, xlabels = [], []
    ob_handles = []
    prev_ob = None
    for r in rows:
        ob  = r["ob"]
        col = OB_COLOURS.get(ob, "grey")
        val = float(r["D_std"])
        axes[0].scatter(frame_idx, val, color=col, s=18, zorder=3, alpha=0.85)
        if ob != prev_ob:
            if frame_idx > 0:
                axes[0].axvline(frame_idx - 0.5, color="grey", lw=0.6, ls="--", alpha=0.5)
            xticks.append(frame_idx)
            xlabels.append(ob.replace("GX339_Ks_Imaging_", "OB "))
            ob_handles.append(mpatches.Patch(color=col,
                                             label=ob.replace("GX339_Ks_Imaging_", "OB ")))
        frame_idx += 1
        prev_ob = ob

    axes[0].axhline(1.0, color="black", lw=1.0, ls="-",  label="Ideal (1.0)")
    axes[0].axhline(0.7, color="grey",  lw=0.8, ls=":",  label="Lower guideline (0.7)")
    axes[0].axhline(1.3, color="grey",  lw=0.8, ls=":",  label="Upper guideline (1.3)")
    axes[0].set_ylabel("D_std  (3σ-clipped)")
    axes[0].set_title("ZOGY Difference Image Quality: per-frame noise diagnostic")
    axes[0].set_ylim(0.3, 1.5)
    axes[0].set_xlim(-1, frame_idx)
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    axes[0].legend(handles=ob_handles + [
        mpatches.Patch(color="black", label="Ideal (1.0)"),
        mpatches.Patch(color="grey",  label="Guideline (0.7–1.3)"),
    ], loc="upper right", fontsize=9, ncol=4)
    axes[0].grid(axis="y", alpha=0.3)

    # Bottom: FWHM per frame
    frame_idx = 0
    prev_ob = None
    for r in rows:
        ob  = r["ob"]
        col = OB_COLOURS.get(ob, "grey")
        fwhm = float(r["fwhm_n"])
        axes[1].scatter(frame_idx, fwhm, color=col, s=18, zorder=3, alpha=0.85)
        if ob != prev_ob and frame_idx > 0:
            axes[1].axvline(frame_idx - 0.5, color="grey", lw=0.6, ls="--", alpha=0.5)
        frame_idx += 1
        prev_ob = ob

    axes[1].set_ylabel("Science frame FWHM (pixels)")
    axes[1].set_title("Per-frame seeing (FWHM) across all OBs")
    axes[1].set_xlim(-1, frame_idx)
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].set_ylim(3, 14)

    plt.tight_layout()
    out = FIG_DIR / "fig4_dstd_timeline.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
# Figure 5 — Pixel histogram of D (one representative frame)
# =============================================================================
def fig5_d_histogram():
    print("Figure 5: D pixel histogram …")
    ob   = REPRESENTATIVE_OB
    stem = REPRESENTATIVE_STEM
    diff_path = DIFF_DIR / ob / f"{stem}_diff.fits"

    if not diff_path.exists():
        print(f"  Diff frame not found: {diff_path}"); return

    diff = fits.getdata(diff_path).astype(float)
    D_fin = diff[np.isfinite(diff)].ravel()
    d_mean_c, d_med_c, d_std_c = sigma_clipped_stats(D_fin, sigma=3.0, maxiters=5)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Centre plot window on the data, ±6σ
    lim = 6 * d_std_c
    xlo, xhi = d_mean_c - lim, d_mean_c + lim
    bins = np.linspace(xlo, xhi, 150)
    ax.hist(D_fin, bins=bins, color="#4C72B0", alpha=0.75, density=True,
            label="D  (background pixels)", zorder=2)

    # Both curves centred on d_mean_c so width comparison is meaningful.
    # Note: ZOGY guarantees background pixels follow N(0,1) in D; a non-zero
    # mean here reflects a residual sky offset between N and R — corrected
    # locally at the target position in Stage 10.
    x = np.linspace(xlo, xhi, 500)
    gauss_fit = (np.exp(-0.5 * ((x - d_mean_c) / d_std_c) ** 2)
                 / (d_std_c * np.sqrt(2 * np.pi)))
    ax.plot(x, gauss_fit, "r-", lw=1.8,
            label=f"Gaussian fit: mean={d_mean_c:.2f}, σ={d_std_c:.3f}", zorder=3)

    gauss_ideal = (np.exp(-0.5 * ((x - d_mean_c) / 1.0) ** 2)
                   / (1.0 * np.sqrt(2 * np.pi)))
    ax.plot(x, gauss_ideal, "k--", lw=1.2,
            label=f"Ideal noise model: same mean, σ=1.0", zorder=3)

    ax.axvline(d_mean_c, color="orange", lw=1.2, ls="--",
               label=f"Clipped mean = {d_mean_c:.3f}  (sky offset; see Stage 10)")
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("D  [ZOGY background statistic]")
    ax.set_ylabel("Probability density")
    ax.set_title(
        f"D background noise diagnostic  |  {ob}\n"
        f"Width: σ(3σ-clip) = {d_std_c:.3f}  (ZOGY theory: 1.0;  crowded field OK > 0.7)"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    out = FIG_DIR / "fig5_d_histogram.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  -> {out}")


# =============================================================================
if __name__ == "__main__":
    print(f"Saving report figures to: {FIG_DIR}\n")
    fig1_reference()
    fig2_before_after()
    fig3_sample_diff()
    fig4_dstd_timeline()
    fig5_d_histogram()
    print("\nDone.")
