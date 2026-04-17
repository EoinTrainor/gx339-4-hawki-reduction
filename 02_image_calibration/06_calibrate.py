"""
06_calibrate.py
---------------
Apply dark subtraction, flat fielding, and running-median sky subtraction
to all Ks science frames (Detector 1 only).

Calibration sequence per frame:
    calibrated[i] = (raw[i] - master_dark) / master_flat - sky[i]

Sky model for frame i:
    sky[i] = nanmedian of +-SKY_WINDOW nearest corrected frames within the same OB
             (excluding frame i itself; window clipped to OB boundaries)

MODES
-----
  TEST_MODE = True  -> process first OB only; save plots + stats log to
                       OUTPUT_ROOT/_test/  Nothing else written to disk.
  TEST_MODE = False -> process all frames; save calibrated FITS + log reports;
                       delete OUTPUT_ROOT/_test/ if it exists.
"""

# ---- SETTINGS ----------------------------------------------------------------
TEST_MODE  = False     # True = diagnostics to _test folder. False = full run.
SKY_WINDOW = 5         # use +-SKY_WINDOW frames for running-median sky
# ------------------------------------------------------------------------------

import shutil
import warnings
from pathlib import Path
import sys
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")           # always non-interactive; saves to file
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ============================================================
# 1. PATHS
# ============================================================
SCIENCE_DIR    = config.SCIENCE_DIR
CALIBRATED_DIR = config.CALIBRATED_DIR
MASTER_DARK    = config.MASTER_DARK_FILE
MASTER_FLAT    = config.MASTER_FLAT_FILE
BAD_PIXEL_MASK = config.BAD_PIXEL_MASK
LOGS_DIR       = config.LOGS_CALIBRATION_DIR
DETECTOR_EXT   = 1          # CHIP1.INT1 — Detector 1, GX 339-4 location

TEST_DIR = config.OUTPUT_ROOT / "_test"

# ============================================================
# 2. MODE SETUP
# ============================================================
if TEST_MODE:
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    log_lines = []          # accumulate stats for the text log

    def log(msg=""):
        print(msg)
        log_lines.append(msg)
else:
    # Full run: remove any leftover test folder
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"Removed test folder: {TEST_DIR}")

    def log(msg=""):
        print(msg)

# ============================================================
# 3. CHECK INPUTS
# ============================================================
for label, path in [("Science dir",    SCIENCE_DIR),
                    ("Master dark",    MASTER_DARK),
                    ("Master flat",    MASTER_FLAT),
                    ("Bad pixel mask", BAD_PIXEL_MASK)]:
    if not path.exists():
        log(f"ERROR: {label} not found:\n  {path}")
        raise SystemExit

science_files = sorted(SCIENCE_DIR.glob("HAWKI*.fits"))
if not science_files:
    log(f"ERROR: No science FITS found in:\n  {SCIENCE_DIR}")
    raise SystemExit

log("=" * 90)
log("CALIBRATE SCIENCE FRAMES — Dark / Flat / Sky")
log("=" * 90)
log(f"Science files found : {len(science_files)}")
log(f"Sky window          : +-{SKY_WINDOW} frames")
log(f"TEST MODE           : {TEST_MODE}")
log()

# ============================================================
# 4. LOAD CALIBRATION PRODUCTS
# ============================================================
log("Loading calibration products...")

with fits.open(MASTER_DARK) as hdul:
    master_dark = np.asarray(hdul[0].data, dtype=np.float64)

with fits.open(MASTER_FLAT) as hdul:
    master_flat = np.asarray(hdul[0].data, dtype=np.float64)

with fits.open(BAD_PIXEL_MASK) as hdul:
    bad_pixel_mask = np.asarray(hdul[0].data, dtype=bool)   # True = bad pixel

flat_safe = master_flat.copy()
flat_safe[flat_safe == 0] = np.nan      # avoid division by zero at dead pixels

n_bad = int(np.sum(bad_pixel_mask))
log(f"  Master dark  : {master_dark.shape},  median = {np.nanmedian(master_dark):.2f} ADU")
log(f"  Master flat  : {master_flat.shape},  median = {np.nanmedian(master_flat):.4f}")
log(f"  Bad pixels   : {n_bad:,}  ({100 * n_bad / bad_pixel_mask.size:.2f}%)")
log()

# ============================================================
# 5. GROUP SCIENCE FILES BY OB (sorted numerically, then by TPL EXPNO)
# ============================================================
log("Grouping science frames by OB...")

ob_groups = {}

for fpath in science_files:
    try:
        with fits.open(fpath, ignore_missing_simple=True) as hdul:
            h       = hdul[0].header
            ob_name = h.get("HIERARCH ESO OBS NAME", "UNKNOWN_OB")
            exp_no  = h.get("HIERARCH ESO TPL EXPNO", 9999)
    except Exception as e:
        log(f"  WARNING: Could not read {fpath.name}: {e}")
        ob_name, exp_no = "UNKNOWN_OB", 9999

    ob_groups.setdefault(ob_name, []).append((exp_no, fpath))

def ob_sort_key(name):
    """Sort OB names by trailing integer (GX339_Ks_Imaging_10 > _2)."""
    parts = name.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return name

ob_names = sorted(ob_groups.keys(), key=ob_sort_key)
for ob in ob_names:
    ob_groups[ob].sort(key=lambda x: x[0])
    ob_groups[ob] = [p for _, p in ob_groups[ob]]

log(f"  OBs found: {len(ob_names)}")
for ob in ob_names:
    log(f"    {ob}: {len(ob_groups[ob])} frames")
log()

if TEST_MODE:
    first_ob = ob_names[0]
    ob_names = [first_ob]
    log(f"TEST MODE: restricting to '{first_ob}' ({len(ob_groups[first_ob])} frames)")
    log()

# ============================================================
# 6. HELPERS
# ============================================================

def load_and_correct(fpath):
    """Load Detector 1, dark-subtract and flat-field. Returns float64 array."""
    with fits.open(fpath, ignore_missing_simple=True) as hdul:
        raw     = np.asarray(hdul[DETECTOR_EXT].data, dtype=np.float64)
        hdr_pri = hdul[0].header.copy()
        hdr_det = hdul[DETECTOR_EXT].header.copy()
    return (raw - master_dark) / flat_safe, hdr_pri, hdr_det


def build_sky(stack, idx, window):
    """Running-median sky for frame idx using +-window neighbours (same OB)."""
    n   = len(stack)
    lo  = max(0, idx - window)
    hi  = min(n, idx + window + 1)
    neighbours = [j for j in range(lo, hi) if j != idx and stack[j] is not None]
    if len(neighbours) < 3:
        neighbours = [j for j in range(n) if j != idx and stack[j] is not None]
    if len(neighbours) == 0:
        raise RuntimeError(f"No valid neighbours for frame {idx} — cannot build sky.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # suppress all-NaN slice
        return np.nanmedian(np.stack([stack[j] for j in neighbours], axis=0), axis=0)


def robust_clim(arr, lo=1, hi=99):
    finite = arr[np.isfinite(arr)]
    return np.percentile(finite, lo), np.percentile(finite, hi)


def save_fig(fig, name):
    path = TEST_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  [plot] {path.name}")

# ============================================================
# 7. PROCESS OBs
# ============================================================
for ob_name in ob_names:
    frames   = ob_groups[ob_name]
    n_frames = len(frames)

    log("-" * 70)
    log(f"OB: {ob_name}  ({n_frames} frames)")
    log("-" * 70)

    # ── 7a. Load and dark/flat correct entire OB into memory ──────────────
    log(f"  Loading {n_frames} frames...")
    corrected_stack = []
    primary_headers = []
    chip_headers    = []
    filenames       = []

    for i, fpath in enumerate(frames, start=1):
        try:
            corr, hdr_p, hdr_c = load_and_correct(fpath)
            corrected_stack.append(corr)
            primary_headers.append(hdr_p)
            chip_headers.append(hdr_c)
        except Exception as e:
            log(f"    [{i:2d}] ERROR loading {fpath.name}: {e} — skipping")
            corrected_stack.append(None)
            primary_headers.append(None)
            chip_headers.append(None)
        filenames.append(fpath.stem)

    # ── 7b. Build sky, subtract, apply BPM ────────────────────────────────
    log(f"\n  {'Frame':<10} {'Filename':<45} {'Sky (ADU)':>10} {'Post-sky median':>16}")
    log(f"  {'-'*10} {'-'*45} {'-'*10} {'-'*16}")

    sky_levels       = []
    post_medians     = []
    calibrated_stack = []

    for i in range(n_frames):
        if corrected_stack[i] is None:
            sky_levels.append(np.nan)
            post_medians.append(np.nan)
            calibrated_stack.append(None)
            continue

        sky       = build_sky(corrected_stack, i, SKY_WINDOW)
        sky_level = float(np.nanmedian(sky))
        cal       = corrected_stack[i] - sky
        cal[bad_pixel_mask] = np.nan
        post_med  = float(np.nanmedian(cal))

        sky_levels.append(sky_level)
        post_medians.append(post_med)
        calibrated_stack.append(cal)

        log(f"  [{i+1:2d}/{n_frames}]  {filenames[i]:<45} {sky_level:>10.1f} {post_med:>16.3f}")

        if not TEST_MODE:
            out_dir = CALIBRATED_DIR / ob_name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_hdr = chip_headers[i].copy()
            out_hdr["HISTORY"] = f"Dark subtracted: {MASTER_DARK.name}"
            out_hdr["HISTORY"] = f"Flat fielded: {MASTER_FLAT.name}"
            out_hdr["HISTORY"] = f"Sky subtracted: running nanmedian +-{SKY_WINDOW} frames"
            out_hdr["HISTORY"] = f"Bad pixels set to NaN: {BAD_PIXEL_MASK.name}"
            out_hdr["IMAGETYP"] = "CALIBRATED"
            out_hdr["SKYMED"]   = (round(sky_level, 3), "Median sky ADU subtracted")
            out_hdr["BUNIT"]    = "ADU"

            fits.writeto(out_dir / (filenames[i] + "_cal.fits"),
                         cal.astype(np.float32), out_hdr, overwrite=True)

    # ── 7c. Summary stats ─────────────────────────────────────────────────
    valid_sky = [s for s in sky_levels if not np.isnan(s)]
    log()
    log(f"  Sky statistics for {ob_name}:")
    log(f"    Min sky  : {min(valid_sky):.1f} ADU")
    log(f"    Max sky  : {max(valid_sky):.1f} ADU")
    log(f"    Mean sky : {np.mean(valid_sky):.1f} ADU")
    log(f"    Std sky  : {np.std(valid_sky):.1f} ADU  (frame-to-frame variation)")

    # ── 7d. Save log report (full run only) ───────────────────────────────
    if not TEST_MODE:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = LOGS_DIR / f"{ob_name}_calibration_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write(f"CALIBRATION REPORT — {ob_name}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Frames processed : {n_frames}\n")
            f.write(f"Sky window       : +-{SKY_WINDOW} frames\n")
            f.write(f"Master dark      : {MASTER_DARK.name}\n")
            f.write(f"Master flat      : {MASTER_FLAT.name}\n")
            f.write(f"Bad pixel mask   : {BAD_PIXEL_MASK.name}\n\n")
            f.write(f"{'Frame':<45} {'Sky (ADU)':>12} {'Post-sky median':>16}\n")
            f.write("-" * 80 + "\n")
            for fname, sky, post in zip(filenames, sky_levels, post_medians):
                f.write(f"{fname:<45} {sky:>12.1f} {post:>16.3f}\n")
        log(f"  Report saved: {report_path}")

    # ── 7e. Diagnostic plots (TEST_MODE only) ─────────────────────────────
    if TEST_MODE:
        valid_idx = [i for i, s in enumerate(sky_levels) if not np.isnan(s)]
        log()

        # Plot 1: Sky level per frame
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot([i + 1 for i in valid_idx],
                [sky_levels[i] for i in valid_idx],
                "o-", color="steelblue", ms=5, lw=1.5)
        ax.set_xlabel("Frame number within OB")
        ax.set_ylabel("Sky background (ADU)")
        ax.set_title(f"{ob_name} — Sky level per frame")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        save_fig(fig, f"{ob_name}_01_sky_per_frame.png")

        # Plot 2: Frame-to-frame sky variation
        if len(valid_idx) > 1:
            sky_arr  = np.array([sky_levels[i] for i in valid_idx])
            sky_diff = np.diff(sky_arr)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.bar(range(1, len(sky_diff) + 1), sky_diff,
                   color=["tomato" if d > 0 else "steelblue" for d in sky_diff])
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xlabel("Frame interval (i to i+1)")
            ax.set_ylabel("Sky change (ADU)")
            ax.set_title(f"{ob_name} — Frame-to-frame sky variation")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save_fig(fig, f"{ob_name}_02_sky_variation.png")

        # Plot 3: Before / sky model / calibrated for the middle frame
        mid = n_frames // 2
        if corrected_stack[mid] is not None:
            sky_mid = build_sky(corrected_stack, mid, SKY_WINDOW)
            cal_mid = calibrated_stack[mid]

            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            panels = [
                (corrected_stack[mid], "Dark/flat corrected (pre-sky)", "viridis"),
                (sky_mid,              "Sky model",                     "viridis"),
                (cal_mid,              "Calibrated (sky subtracted)",   "gray"),
            ]
            for ax, (data, title, cmap) in zip(axes, panels):
                vmin, vmax = robust_clim(data)
                im = ax.imshow(data, origin="lower", cmap=cmap,
                               vmin=vmin, vmax=vmax, interpolation="none")
                ax.set_title(title, fontsize=10)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="ADU")
            fig.suptitle(f"{ob_name} — frame {mid+1}/{n_frames}: {filenames[mid]}", fontsize=11)
            fig.tight_layout()
            save_fig(fig, f"{ob_name}_03_before_sky_after.png")

        # Plot 4: Pixel histogram of calibrated middle frame
        if calibrated_stack[mid] is not None:
            finite_pix = calibrated_stack[mid][np.isfinite(calibrated_stack[mid])].ravel()
            lo, hi = np.percentile(finite_pix, [0.5, 99.5])
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(finite_pix, bins=300, range=(lo, hi),
                    color="steelblue", alpha=0.8, edgecolor="none")
            ax.axvline(0, color="red", lw=1.5, linestyle="--", label="zero")
            ax.set_xlabel("Pixel value (ADU)")
            ax.set_ylabel("Count")
            ax.set_title(f"{ob_name} — frame {mid+1} calibrated pixel histogram")
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            save_fig(fig, f"{ob_name}_04_histogram.png")

# ============================================================
# 8. FINAL SUMMARY + SAVE TEST LOG
# ============================================================
log()
log("=" * 90)
total = sum(len(ob_groups[ob]) for ob in ob_names)
if TEST_MODE:
    log(f"TEST COMPLETE — {total} frames processed, nothing written to science output.")
    log(f"Diagnostics saved to: {TEST_DIR}")
    log("If diagnostics look good, set TEST_MODE = False and re-run.")
else:
    log(f"DONE — {total} frames calibrated and saved to {CALIBRATED_DIR}")
log("=" * 90)

if TEST_MODE:
    log_path = TEST_DIR / "test_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"\nTest log saved: {log_path}")
