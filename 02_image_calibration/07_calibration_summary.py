"""
07_calibration_summary.py
--------------------------
Read the per-OB calibration report logs produced by 06_calibrate.py and
generate global summary plots across all 12 OBs.

Plots produced:
  1. Mean sky level per OB (bar chart + std error bars)
  2. Sky std per OB — frame-to-frame variation within each OB
  3. Post-sky residuals per OB — scatter of all frame medians
  4. Global sky timeline — all frames in time order, coloured by OB

MODES
-----
  TEST_MODE = True  -> saves plots to OUTPUT_ROOT/_test/  (for review)
  TEST_MODE = False -> saves plots to LOGS_CAL_DIR/calibration_summary_plots/
                       and deletes _test/ if present
"""

# ---- SETTINGS ----------------------------------------------------------------
TEST_MODE = False
# ------------------------------------------------------------------------------

import shutil
import re
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ============================================================
# 1. PATHS
# ============================================================
LOGS_CAL_DIR     = config.LOGS_CALIBRATION_DIR
LOGS_SUMMARY_DIR = config.LOGS_CAL_SUMMARY_DIR
TEST_DIR         = config.OUTPUT_ROOT / "_test"

if TEST_MODE:
    OUT_DIR = TEST_DIR
    TEST_DIR.mkdir(parents=True, exist_ok=True)
else:
    OUT_DIR = LOGS_SUMMARY_DIR
    LOGS_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"Removed test folder: {TEST_DIR}")

# ============================================================
# 2. PARSE LOG FILES
# ============================================================

def ob_sort_key(name):
    parts = name.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return name

log_files = sorted(
    LOGS_CAL_DIR.glob("GX339_Ks_Imaging_*_calibration_report.txt"),
    key=lambda p: ob_sort_key(p.stem.replace("_calibration_report", ""))
)

if not log_files:
    print(f"ERROR: No calibration report logs found in:\n  {LOGS_CAL_DIR}")
    raise SystemExit

print(f"Found {len(log_files)} calibration report(s)")

ob_data = []   # list of dicts, one per OB

for log_path in log_files:
    ob_name   = None
    sky_vals  = []
    post_vals = []
    ob_date   = None

    with open(log_path, encoding="utf-8") as f:
        in_data = False
        for line in f:
            line = line.rstrip()

            # OB name
            if line.startswith("CALIBRATION REPORT"):
                ob_name = line.split("—")[-1].strip()

            # Start of data rows (after the dashes line)
            if set(line.strip()) == {"-"} and len(line.strip()) > 10:
                in_data = True
                continue

            if in_data and line.startswith("HAWKI."):
                parts = line.split()
                if len(parts) >= 3:
                    fname    = parts[0]
                    sky      = float(parts[1])
                    post     = float(parts[2])
                    sky_vals.append(sky)
                    post_vals.append(post)

                    # Extract date from first data frame
                    if ob_date is None:
                        m = re.search(r"(\d{4}-\d{2}-\d{2})", fname)
                        if m:
                            ob_date = m.group(1)

    if ob_name and sky_vals:
        ob_num = int(ob_name.rsplit("_", 1)[-1])
        ob_data.append({
            "ob_name"   : ob_name,
            "ob_num"    : ob_num,
            "ob_date"   : ob_date or "unknown",
            "sky_vals"  : np.array(sky_vals),
            "post_vals" : np.array(post_vals),
            "n_frames"  : len(sky_vals),
            "sky_mean"  : float(np.mean(sky_vals)),
            "sky_std"   : float(np.std(sky_vals)),
            "post_mean" : float(np.mean(post_vals)),
            "post_std"  : float(np.std(post_vals)),
            "post_abs"  : float(np.mean(np.abs(post_vals))),
        })
        print(f"  {ob_name}: {len(sky_vals)} frames, "
              f"sky={np.mean(sky_vals):.0f}±{np.std(sky_vals):.0f} ADU, "
              f"residual mean={np.mean(post_vals):.1f} ADU")

ob_data.sort(key=lambda d: d["ob_num"])

print(f"\nTotal frames parsed: {sum(d['n_frames'] for d in ob_data)}")

# ── Short axis labels ──────────────────────────────────────────────────────────
labels     = [f"OB{d['ob_num']}\n{d['ob_date']}" for d in ob_data]
n_obs      = len(ob_data)
ob_colours = plt.cm.tab10(np.linspace(0, 1, n_obs))

# ============================================================
# 3. PLOT 1 — Mean sky level per OB
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(n_obs)
bars = ax.bar(x, [d["sky_mean"] for d in ob_data],
              yerr=[d["sky_std"] for d in ob_data],
              color=ob_colours, edgecolor="white", linewidth=0.5,
              capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "black"})
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Mean sky background (ADU)")
ax.set_title("Mean Ks-band sky level per OB  (error bars = frame-to-frame std)")
ax.grid(True, axis="y", alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig.tight_layout()
p = OUT_DIR / "summary_01_sky_mean_per_ob.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[plot] {p.name}")

# ============================================================
# 4. PLOT 2 — Sky std (frame-to-frame variation) per OB
# ============================================================
fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(x, [d["sky_std"] for d in ob_data],
       color=ob_colours, edgecolor="white", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Sky std within OB (ADU)")
ax.set_title("Frame-to-frame sky variation per OB  (lower = more stable sky)")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
p = OUT_DIR / "summary_02_sky_std_per_ob.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[plot] {p.name}")

# ============================================================
# 5. PLOT 3 — Post-sky residuals per OB (scatter + mean line)
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
for i, d in enumerate(ob_data):
    jitter  = np.random.default_rng(i).uniform(-0.2, 0.2, d["n_frames"])
    ax.scatter(np.full(d["n_frames"], i) + jitter, d["post_vals"],
               color=ob_colours[i], s=18, alpha=0.7, zorder=2)
    ax.hlines(d["post_mean"], i - 0.35, i + 0.35,
              color="black", linewidth=1.5, zorder=3)

ax.axhline(0, color="red", linewidth=1.2, linestyle="--", label="zero residual")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Post-sky median (ADU)")
ax.set_title("Per-frame sky subtraction residuals per OB\n"
             "(dots = individual frames, horizontal bar = OB mean)")
ax.legend(fontsize=9)
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
p = OUT_DIR / "summary_03_residuals_per_ob.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[plot] {p.name}")

# ============================================================
# 6. PLOT 4 — Global sky timeline (all frames, coloured by OB)
# ============================================================
fig, ax = plt.subplots(figsize=(14, 5))
frame_offset = 0
for i, d in enumerate(ob_data):
    xs = np.arange(frame_offset, frame_offset + d["n_frames"])
    ax.plot(xs, d["sky_vals"], "o-",
            color=ob_colours[i], ms=3, lw=1.2,
            label=f"OB{d['ob_num']} ({d['ob_date']})")
    frame_offset += d["n_frames"]

ax.set_xlabel("Frame index (all OBs in chronological order)")
ax.set_ylabel("Sky background (ADU)")
ax.set_title("Global sky timeline — all 317 frames, coloured by OB")
ax.legend(fontsize=7, ncol=4, loc="upper right")
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
fig.tight_layout()
p = OUT_DIR / "summary_04_global_sky_timeline.png"
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[plot] {p.name}")

# ============================================================
# 7. PRINT GLOBAL SUMMARY TABLE
# ============================================================
print()
print("=" * 80)
print("GLOBAL CALIBRATION SUMMARY")
print("=" * 80)
print(f"{'OB':<25} {'Date':<12} {'N':>4} {'Sky mean':>10} {'Sky std':>8} "
      f"{'Resid mean':>12} {'|Resid| mean':>13}")
print("-" * 80)
for d in ob_data:
    print(f"{d['ob_name']:<25} {d['ob_date']:<12} {d['n_frames']:>4} "
          f"{d['sky_mean']:>10.0f} {d['sky_std']:>8.1f} "
          f"{d['post_mean']:>12.1f} {d['post_abs']:>13.1f}")

all_sky   = np.concatenate([d["sky_vals"]  for d in ob_data])
all_post  = np.concatenate([d["post_vals"] for d in ob_data])
print("-" * 80)
print(f"{'ALL OBs':<25} {'':<12} {len(all_sky):>4} "
      f"{np.mean(all_sky):>10.0f} {np.std(all_sky):>8.1f} "
      f"{np.mean(all_post):>12.1f} {np.mean(np.abs(all_post)):>13.1f}")
print("=" * 80)
print(f"\nPlots saved to: {OUT_DIR}")
if TEST_MODE:
    print("Set TEST_MODE = False to save to the permanent summary folder.")
