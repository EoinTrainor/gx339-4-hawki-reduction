"""
08_align.py
-----------
Align all 317 calibrated science frames to a common reference frame using
astroalign (star-triangle pattern matching + affine transform). Corrects
for dither offsets (up to 60 arcsec) and any residual rotation between OBs.

Reference frame: first frame of GX339_Ks_Imaging_1 (configurable below).

Outputs (full run):
  aligned/GX339_Ks_Imaging_N/HAWKI.*_1_cal_aligned.fits
  logs/alignment/GX339_Ks_Imaging_N_alignment_report.txt

MODES
-----
  TEST_MODE = True  -> 5 frames from OB1 only; diagnostic plots to _test/
  TEST_MODE = False -> all 317 frames; saves to aligned/ and logs/alignment/
"""

# ---- SETTINGS ----------------------------------------------------------------
TEST_MODE    = False
TEST_N_OB    = "GX339_Ks_Imaging_1"   # OB to test on
TEST_N_FRAMES = 5                      # how many frames from that OB

# Reference frame (relative to CALIBRATED_DIR)
REF_OB    = "GX339_Ks_Imaging_1"
REF_INDEX = 0                          # 0 = first frame (chronological)

# astroalign detection threshold (lower = more stars, but more false detections)
DETECTION_SIGMA = 3.0
# ------------------------------------------------------------------------------

import warnings
import shutil
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import astroalign

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ============================================================
# 1. PATHS
# ============================================================
CALIBRATED_DIR  = config.CALIBRATED_DIR
ALIGNED_DIR     = config.ALIGNED_DIR
LOGS_ALIGN_DIR  = config.LOGS_ALIGNMENT_DIR
TEST_DIR        = config.OUTPUT_ROOT / "_test"

if TEST_MODE:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"TEST MODE — processing {TEST_N_FRAMES} frames from {TEST_N_OB}")
    print(f"Diagnostic output: {TEST_DIR}\n")
else:
    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_ALIGN_DIR.mkdir(parents=True, exist_ok=True)
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"Removed test folder: {TEST_DIR}")

# ============================================================
# 2. OB SORT
# ============================================================
def ob_sort_key(name):
    parts = name.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return name

# ============================================================
# 3. LOAD REFERENCE FRAME
# ============================================================
ref_files = sorted(
    (CALIBRATED_DIR / REF_OB).glob("HAWKI.*_1_cal.fits")
)
if not ref_files:
    print(f"ERROR: No calibrated frames found in {CALIBRATED_DIR / REF_OB}")
    raise SystemExit

ref_path = ref_files[REF_INDEX]
print(f"Reference frame: {ref_path.name}  (OB={REF_OB}, index={REF_INDEX})")

with fits.open(ref_path) as hdul:
    ref_data   = hdul[0].data.astype(np.float64)
    ref_header = hdul[0].header.copy()

# Fill NaNs with median for astroalign source detection
ref_median  = float(np.nanmedian(ref_data))
ref_filled  = np.where(np.isnan(ref_data), ref_median, ref_data)
print(f"  Shape: {ref_data.shape},  NaNs: {np.sum(np.isnan(ref_data)):,},  "
      f"median: {ref_median:,.0f} ADU\n")

# ============================================================
# 4. ALIGNMENT FUNCTION
# ============================================================
def align_frame(data):
    """
    Align data array to ref_filled via astroalign.
    Returns (registered float64 array, transform, n_matched_stars)
    or raises on failure.
    """
    median = float(np.nanmedian(data))
    filled = np.where(np.isnan(data), median, data)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        transform, (src_pts, dst_pts) = astroalign.find_transform(
            filled, ref_filled,
            detection_sigma=DETECTION_SIGMA
        )
        registered, footprint = astroalign.register(
            filled, ref_filled,
            detection_sigma=DETECTION_SIGMA
        )

    # Pixels outside the affine warp have no valid data
    # astroalign footprint convention: True = INVALID (outside source)
    registered[footprint] = np.nan

    n_matched = len(src_pts)
    return registered, transform, n_matched

# ============================================================
# 5. COLLECT OBs TO PROCESS
# ============================================================
all_ob_dirs = sorted(
    [d for d in CALIBRATED_DIR.iterdir() if d.is_dir()],
    key=lambda d: ob_sort_key(d.name)
)

if TEST_MODE:
    all_ob_dirs = [d for d in all_ob_dirs if d.name == TEST_N_OB]

# ============================================================
# 6. MAIN ALIGNMENT LOOP
# ============================================================
grand_total   = 0
grand_failed  = 0
test_records  = []   # (fname, transform, n_matched) for test diagnostics

for ob_dir in all_ob_dirs:
    ob_name  = ob_dir.name
    cal_files = sorted(ob_dir.glob("HAWKI.*_1_cal.fits"))

    if TEST_MODE:
        cal_files = cal_files[:TEST_N_FRAMES]

    if not cal_files:
        print(f"  {ob_name}: no calibrated frames found — skipping")
        continue

    # Per-OB output dir and log
    if not TEST_MODE:
        out_dir = ALIGNED_DIR / ob_name
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_ALIGN_DIR / f"{ob_name}_alignment_report.txt"
        log_lines = [
            f"ALIGNMENT REPORT — {ob_name}",
            f"Reference: {ref_path.name}",
            "-" * 80,
            f"{'Filename':<50} {'Stars':>6} {'dX(px)':>8} {'dY(px)':>8} "
            f"{'Rot(deg)':>9} {'Scale':>7} {'Status':>8}",
            "-" * 80,
        ]

    print(f"\n{'='*60}")
    print(f"OB: {ob_name}  ({len(cal_files)} frames)")
    print(f"{'='*60}")

    ob_failed = 0
    for i, fpath in enumerate(cal_files):
        with fits.open(fpath) as hdul:
            data   = hdul[0].data.astype(np.float64)
            header = hdul[0].header.copy()

        try:
            registered, tf, n_matched = align_frame(data)
            status = "OK"

            dx    = tf.translation[0]
            dy    = tf.translation[1]
            rot   = np.degrees(tf.rotation)
            scale = tf.scale

            print(f"  [{i+1:>3}/{len(cal_files)}] {fpath.name}  "
                  f"stars={n_matched:>3}  dx={dx:+7.2f}  dy={dy:+7.2f}  "
                  f"rot={rot:+6.3f}°  scale={scale:.5f}")

            if TEST_MODE:
                test_records.append((fpath.name, data, registered, tf, n_matched))
            else:
                # Save aligned FITS — same header, updated filename
                out_name = fpath.name.replace("_cal.fits", "_cal_aligned.fits")
                out_path = out_dir / out_name
                hdu = fits.PrimaryHDU(data=registered.astype(np.float32),
                                      header=header)
                hdu.header["ALIGNED"]  = (True, "astroalign affine registration")
                hdu.header["ALIGNREF"] = (ref_path.name, "reference frame")
                hdu.header["ALINSTAR"] = (n_matched, "matched control stars")
                hdu.header["ALINDX"]   = (float(f"{dx:.4f}"), "X translation (px)")
                hdu.header["ALINDY"]   = (float(f"{dy:.4f}"), "Y translation (px)")
                hdu.header["ALINROT"]  = (float(f"{rot:.5f}"), "rotation (deg)")
                hdu.writeto(out_path, overwrite=True)

                log_lines.append(
                    f"{fpath.name:<50} {n_matched:>6} {dx:>8.2f} {dy:>8.2f} "
                    f"{rot:>9.3f} {scale:>7.5f} {status:>8}"
                )

        except Exception as e:
            ob_failed += 1
            grand_failed += 1
            status = "FAILED"
            print(f"  [{i+1:>3}/{len(cal_files)}] {fpath.name}  FAILED: {e}")
            if not TEST_MODE:
                log_lines.append(
                    f"{fpath.name:<50} {'—':>6} {'—':>8} {'—':>8} "
                    f"{'—':>9} {'—':>7} {status:>8}"
                )

        grand_total += 1

    print(f"  OB done — {len(cal_files) - ob_failed} aligned, "
          f"{ob_failed} failed")

    if not TEST_MODE:
        log_lines += [
            "-" * 80,
            f"Frames processed: {len(cal_files)}  "
            f"Aligned: {len(cal_files) - ob_failed}  "
            f"Failed: {ob_failed}",
        ]
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"  Log: {log_path.name}")

# ============================================================
# 7. TEST-MODE DIAGNOSTIC PLOTS
# ============================================================
if TEST_MODE and test_records:
    print(f"\n{'='*60}")
    print("Generating diagnostic plots...")

    # --- Plot 1: reference frame with detected stars marked ---
    ref_median_val = float(np.nanmedian(ref_filled))
    ref_std_val    = float(np.nanstd(ref_filled[~np.isnan(ref_data)]))
    vmin = ref_median_val - 2 * ref_std_val
    vmax = ref_median_val + 5 * ref_std_val

    fig, axes = plt.subplots(1, min(len(test_records), 3) + 1,
                             figsize=(5 * (min(len(test_records), 3) + 1), 5))
    axes[0].imshow(ref_filled, origin="lower", cmap="gray",
                   vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title(f"Reference\n{ref_path.name[:30]}", fontsize=7)
    axes[0].axis("off")

    for j, (fname, raw, registered, tf, n_matched) in enumerate(test_records[:3]):
        ax = axes[j + 1]
        ax.imshow(registered, origin="lower", cmap="gray",
                  vmin=vmin, vmax=vmax, interpolation="nearest")
        dx = tf.translation[0]
        dy = tf.translation[1]
        rot = np.degrees(tf.rotation)
        ax.set_title(f"Aligned: {fname[:25]}\n"
                     f"stars={n_matched}  dx={dx:+.1f}  dy={dy:+.1f}  "
                     f"rot={rot:+.3f}°", fontsize=6)
        ax.axis("off")

    fig.suptitle(f"Alignment test — {TEST_N_OB}  (reference + first 3 aligned)",
                 fontsize=9)
    fig.tight_layout()
    p = TEST_DIR / "align_01_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {p.name}")

    # --- Plot 2: residual images (aligned - reference) ---
    n_panels = len(test_records)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    rlim = ref_std_val * 2
    for j, (fname, raw, registered, tf, n_matched) in enumerate(test_records):
        residual = registered - ref_filled
        im = axes[j].imshow(residual, origin="lower", cmap="RdBu_r",
                             vmin=-rlim, vmax=rlim, interpolation="nearest")
        axes[j].set_title(f"{fname[:25]}\nresidual (aligned−ref)", fontsize=6)
        axes[j].axis("off")
        plt.colorbar(im, ax=axes[j], fraction=0.046, label="ADU")
    fig.suptitle("Residual images — should show no large-scale structure",
                 fontsize=9)
    fig.tight_layout()
    p = TEST_DIR / "align_02_residuals.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {p.name}")

    # --- Plot 3: transform parameters across test frames ---
    fnames  = [r[0][:20] for r in test_records]
    dxs     = [r[3].translation[0] for r in test_records]
    dys     = [r[3].translation[1] for r in test_records]
    rots    = [np.degrees(r[3].rotation) for r in test_records]
    nstars  = [r[4] for r in test_records]
    xs      = np.arange(len(test_records))

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes[0, 0].bar(xs, dxs, color="steelblue")
    axes[0, 0].set_title("X translation (px)")
    axes[0, 0].set_xticks(xs); axes[0, 0].set_xticklabels(fnames, rotation=30, ha="right", fontsize=6)

    axes[0, 1].bar(xs, dys, color="steelblue")
    axes[0, 1].set_title("Y translation (px)")
    axes[0, 1].set_xticks(xs); axes[0, 1].set_xticklabels(fnames, rotation=30, ha="right", fontsize=6)

    axes[1, 0].bar(xs, rots, color="coral")
    axes[1, 0].set_title("Rotation (degrees)")
    axes[1, 0].set_xticks(xs); axes[1, 0].set_xticklabels(fnames, rotation=30, ha="right", fontsize=6)

    axes[1, 1].bar(xs, nstars, color="seagreen")
    axes[1, 1].set_title("Matched control stars")
    axes[1, 1].set_xticks(xs); axes[1, 1].set_xticklabels(fnames, rotation=30, ha="right", fontsize=6)

    fig.suptitle(f"Alignment transform parameters — {TEST_N_OB} test frames", fontsize=10)
    fig.tight_layout()
    p = TEST_DIR / "align_03_transform_params.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {p.name}")

    # --- Plot 4: science image check — registered frame in inferno with GX 339-4 ---
    from astropy.wcs import WCS
    from matplotlib.patches import Rectangle

    # Use reference CALIBRATED frame WCS (most accurate — pre-distortion)
    ref_wcs  = WCS(ref_header)
    px, py   = ref_wcs.all_world2pix([[config.TARGET_RA, config.TARGET_DEC]], 0)[0]
    px, py   = int(np.round(px)), int(np.round(py))

    # Pick the second test frame (frame 1 = non-trivial offset, good science check)
    _, _, reg_frame, tf_frame, n_frame = test_records[1]
    valid    = reg_frame[np.isfinite(reg_frame)]
    vmin_s   = np.percentile(valid, 1)
    vmax_s   = np.percentile(valid, 99.5)

    # Zoom half-width — wide enough to find GX 339-4 even if WCS is ~5" off
    ZOOM = 150   # px (~16 arcsec each side)
    x0 = max(0, px - ZOOM);  x1 = min(reg_frame.shape[1], px + ZOOM)
    y0 = max(0, py - ZOOM);  y1 = min(reg_frame.shape[0], py + ZOOM)
    cutout = reg_frame[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("black")

    # Full frame
    ax = axes[0]
    ax.imshow(reg_frame, origin="lower", cmap="inferno",
              vmin=vmin_s, vmax=vmax_s, interpolation="nearest")
    ax.plot(px, py, "+", color="cyan", ms=14, mew=1.5)
    rect = Rectangle((x0, y0), x1-x0, y1-y0,
                     linewidth=1.2, edgecolor="cyan", facecolor="none")
    ax.add_patch(rect)
    dx_f  = tf_frame.translation[0]
    dy_f  = tf_frame.translation[1]
    rot_f = np.degrees(tf_frame.rotation)
    ax.set_title(f"Aligned frame 2  (inferno)\n"
                 f"dx={dx_f:+.1f}px  dy={dy_f:+.1f}px  rot={rot_f:+.3f}°  stars={n_frame}",
                 color="white", fontsize=8)
    ax.tick_params(colors="white"); ax.set_xlabel("X (px)", color="white")
    ax.set_ylabel("Y (px)", color="white")
    for sp in ax.spines.values(): sp.set_edgecolor("white")

    # Zoom on GX 339-4
    ax2 = axes[1]
    cutout_valid = cutout[np.isfinite(cutout)]
    if cutout_valid.size > 0:
        vmin_z = np.percentile(cutout_valid, 2)
        vmax_z = np.percentile(cutout_valid, 99)
    else:
        vmin_z, vmax_z = vmin_s, vmax_s
    ax2.imshow(cutout, origin="lower", cmap="inferno",
               vmin=vmin_z, vmax=vmax_z, interpolation="nearest",
               extent=[x0, x1, y0, y1])
    ax2.plot(px, py, "+", color="cyan", ms=20, mew=2.0)
    scale_px = 50
    ax2.plot([x0+6, x0+6+scale_px], [y0+8, y0+8], color="white", lw=2)
    ax2.text(x0+6+scale_px/2, y0+16, f'{scale_px*0.106:.1f}"',
             color="white", fontsize=8, ha="center")
    ax2.set_title(f"GX 339-4 region  ({2*ZOOM*0.106:.0f}\" × {2*ZOOM*0.106:.0f}\")\n"
                  f"RA={config.TARGET_RA:.4f}  Dec={config.TARGET_DEC:.4f}  "
                  f"[cyan + = WCS position]",
                  color="white", fontsize=8)
    ax2.tick_params(colors="white"); ax2.set_xlabel("X (px)", color="white")
    ax2.set_ylabel("Y (px)", color="white")
    for sp in ax2.spines.values(): sp.set_edgecolor("white")

    fig.suptitle("Science image check — fully processed aligned frame",
                 color="white", fontsize=11)
    fig.tight_layout()
    p = TEST_DIR / "align_04_science_check.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {p.name}")
    print(f"       GX 339-4 WCS pixel position: ({px}, {py})")

    # --- Text summary ---
    lines = [
        f"ALIGNMENT TEST SUMMARY — {TEST_N_OB}",
        f"Reference: {ref_path.name}",
        "-" * 70,
        f"{'Frame':<45} {'Stars':>6} {'dX':>7} {'dY':>7} {'Rot(deg)':>9}",
        "-" * 70,
    ]
    for fname, _, _, tf, n_matched in test_records:
        dx  = tf.translation[0]
        dy  = tf.translation[1]
        rot = np.degrees(tf.rotation)
        lines.append(f"{fname:<45} {n_matched:>6} {dx:>7.2f} {dy:>7.2f} {rot:>9.4f}")
    lines += [
        "-" * 70,
        f"Frames aligned: {len(test_records)} / {TEST_N_FRAMES}",
        f"Mean |dX|: {np.mean(np.abs(dxs)):.2f} px",
        f"Mean |dY|: {np.mean(np.abs(dys)):.2f} px",
        f"Max offset: {max(np.sqrt(np.array(dxs)**2 + np.array(dys)**2)):.2f} px",
    ]
    log_path = TEST_DIR / "align_test_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[log]  {log_path.name}")

# ============================================================
# 8. GRAND SUMMARY
# ============================================================
print()
print("=" * 60)
if TEST_MODE:
    print(f"TEST COMPLETE — {len(test_records)} frames aligned")
    print(f"Plots and log saved to: {TEST_DIR}")
    print("Set TEST_MODE = False to run on all 317 frames.")
else:
    print(f"ALIGNMENT COMPLETE")
    print(f"  Total frames : {grand_total}")
    print(f"  Aligned OK   : {grand_total - grand_failed}")
    print(f"  Failed       : {grand_failed}")
    print(f"  Aligned FITS : {ALIGNED_DIR}")
    print(f"  Logs         : {LOGS_ALIGN_DIR}")
print("=" * 60)

# ============================================================
# 9. EMAIL NOTIFICATION (full run only)
# ============================================================
if not TEST_MODE:
    import os, smtplib
    from email.message import EmailMessage
    from datetime import datetime

    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pw   = os.environ.get("GMAIL_APP_PW")

    if gmail_user and gmail_pw:
        try:
            msg = EmailMessage()
            msg["Subject"] = "GX 339-4 Pipeline — Alignment complete"
            msg["From"]    = gmail_user
            msg["To"]      = gmail_user
            msg.set_content(
                f"Alignment run finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"  Total frames : {grand_total}\n"
                f"  Aligned OK   : {grand_total - grand_failed}\n"
                f"  Failed       : {grand_failed}\n\n"
                f"Aligned FITS saved to:\n  {ALIGNED_DIR}\n"
            )
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(gmail_user, gmail_pw)
                smtp.send_message(msg)
            print(f"Email notification sent to {gmail_user}")
        except Exception as e:
            print(f"Email notification failed: {e}")
    else:
        print("(No email sent — set GMAIL_USER and GMAIL_APP_PW env vars to enable)")
