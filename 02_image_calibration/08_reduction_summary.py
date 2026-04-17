"""
08_reduction_summary.py
-----------------------
Produce a global CSV summarising the full reduction journey of every
science frame: raw metadata → calibration statistics → alignment results.

One row per frame (317 total). Columns cover:
  - Observation metadata  (OB, date, MJD, airmass, seeing, humidity, wind)
  - Instrument parameters (DIT, NDIT, effective exptime, gain, readnoise)
  - Calibration results   (sky level, post-sky residual)
  - Alignment results     (status, dX, dY, offset, rotation, scale, stars)
  - Pipeline provenance   (raw / calibrated / aligned filenames)
  - Science classification (quiescent / outburst)

Output:
  logs/reduction_summary.csv

HAWK-I Detector 1 nominal instrument values (ESO headers contain
placeholder 1.0/0.0 in pre-processed files):
  Gain     : 1.705  e-/ADU  (Hawaii-2RG chip 66, ESO manual)
  RON      : 4.5    e-      (single read)
  Eff. RON : RON / sqrt(NDIT)  → ~1.5 e-  (after NDIT=9 averaging)
"""

import csv
import re
import sys
import numpy as np
from pathlib import Path
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config

# ── Instrument constants (HAWK-I Detector 1, Hawaii-2RG chip 66) ─────────────
HAWKI_GAIN    = 1.705   # e-/ADU
HAWKI_RON     = 4.5     # e- (single read)
PLATE_SCALE   = 0.106   # arcsec/pixel (HAWK-I Ks)

# ── Outburst classification ───────────────────────────────────────────────────
OUTBURST_OBS  = {11, 12}   # Swift/BAT confirmed outburst OBs

# ── Output ────────────────────────────────────────────────────────────────────
OUT_CSV = config.LOGS_DIR / "reduction_summary.csv"

# ============================================================
# 1. HELPER: OB sort key
# ============================================================
def ob_sort_key(name):
    parts = name.rsplit("_", 1)
    try:
        return int(parts[-1])
    except ValueError:
        return name

# ============================================================
# 2. PARSE CALIBRATION LOGS
# ============================================================
# Returns dict: filename_stem → {"sky": float, "post": float}
def parse_cal_logs():
    data = {}
    for log in config.LOGS_CALIBRATION_DIR.glob(
            "GX339_Ks_Imaging_*_calibration_report.txt"):
        with open(log, encoding="utf-8") as f:
            in_data = False
            for line in f:
                line = line.rstrip()
                if set(line.strip()) == {"-"} and len(line.strip()) > 10:
                    in_data = True
                    continue
                if in_data and line.startswith("HAWKI."):
                    parts = line.split()
                    if len(parts) >= 3:
                        stem = parts[0].replace("_cal.fits", "").replace(".fits", "")
                        data[stem] = {
                            "sky_adu" : float(parts[1]),
                            "post_adu": float(parts[2]),
                        }
    return data

# ============================================================
# 3. PARSE ALIGNMENT LOGS
# ============================================================
# Returns dict: filename_stem → alignment record dict
def parse_align_logs():
    data = {}
    for log in config.LOGS_ALIGNMENT_DIR.glob(
            "GX339_Ks_Imaging_*_alignment_report.txt"):
        with open(log, encoding="utf-8") as f:
            for line in f:
                if not line.startswith("HAWKI."):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                fname = parts[0]
                status = parts[-1].strip()
                stem = fname.replace("_cal.fits", "").replace(".fits", "")
                if status == "OK" and len(parts) >= 7:
                    data[stem] = {
                        "aligned"       : True,
                        "stars_matched" : int(parts[1]),
                        "dx_px"         : float(parts[2]),
                        "dy_px"         : float(parts[3]),
                        "rotation_deg"  : float(parts[4]),
                        "scale"         : float(parts[5]),
                    }
                else:
                    data[stem] = {"aligned": False}
    return data

# ============================================================
# 4. COLLECT RAW FILES BY OB
# ============================================================
# Group raw science files by OB name using header keyword
print("Scanning raw science headers …")
raw_by_ob = {}   # ob_name → sorted list of Path objects

raw_files = sorted(config.SCIENCE_DIR.glob("HAWKI.*_1.fits"))
for fpath in raw_files:
    try:
        ob = fits.getval(str(fpath), "HIERARCH ESO OBS NAME", ext=0)
    except Exception:
        ob = "unknown"
    raw_by_ob.setdefault(ob, []).append(fpath)

# Sort each OB by TPL EXPNO (chronological within OB)
for ob in raw_by_ob:
    raw_by_ob[ob].sort(
        key=lambda p: fits.getval(str(p), "HIERARCH ESO TPL EXPNO", ext=0)
    )

print(f"  Found {sum(len(v) for v in raw_by_ob.values())} raw frames "
      f"across {len(raw_by_ob)} OBs")

# ============================================================
# 5. LOAD PARSED LOG DATA
# ============================================================
print("Parsing calibration logs …")
cal_data   = parse_cal_logs()
print(f"  {len(cal_data)} frames in calibration logs")

print("Parsing alignment logs …")
align_data = parse_align_logs()
print(f"  {len(align_data)} frames in alignment logs")

# ============================================================
# 6. BUILD CSV ROWS
# ============================================================
obs_sorted = sorted(raw_by_ob.keys(), key=ob_sort_key)

rows = []
for ob_name in obs_sorted:
    ob_num = int(ob_name.rsplit("_", 1)[-1])
    state  = "outburst" if ob_num in OUTBURST_OBS else "quiescent"

    for frame_idx, raw_path in enumerate(raw_by_ob[ob_name], start=1):
        raw_stem = raw_path.stem                        # HAWKI.YYYY-...T..._1
        cal_stem = raw_stem + "_cal"                    # HAWKI..._1_cal

        # ── Raw header ──────────────────────────────────────
        try:
            h0 = fits.getheader(str(raw_path), ext=0)
            date_obs     = h0.get("DATE-OBS", "")
            mjd_obs      = float(h0.get("MJD-OBS", np.nan))
            dit          = float(h0.get("HIERARCH ESO DET DIT", np.nan))
            ndit         = int(h0.get("HIERARCH ESO DET NDIT", 0))
            airm_start   = float(h0.get("HIERARCH ESO TEL AIRM START", np.nan))
            airm_end     = float(h0.get("HIERARCH ESO TEL AIRM END",   np.nan))
            fwhm_start   = float(h0.get("HIERARCH ESO TEL AMBI FWHM START", np.nan))
            fwhm_end     = float(h0.get("HIERARCH ESO TEL AMBI FWHM END",   np.nan))
            humidity     = float(h0.get("HIERARCH ESO TEL AMBI RHUM",   np.nan))
            windspeed    = float(h0.get("HIERARCH ESO TEL AMBI WINDSP", np.nan))
            tpl_expno    = int(h0.get("HIERARCH ESO TPL EXPNO", frame_idx))
            tpl_nexp     = int(h0.get("HIERARCH ESO TPL NEXP", 0))
        except Exception as e:
            print(f"  WARNING: could not read header for {raw_path.name}: {e}")
            date_obs = mjd_obs = dit = ndit = ""
            airm_start = airm_end = fwhm_start = fwhm_end = ""
            humidity = windspeed = tpl_expno = tpl_nexp = ""

        # Derived instrument values
        eff_exptime = dit * ndit if (dit and ndit) else np.nan
        eff_ron     = HAWKI_RON / np.sqrt(ndit) if ndit else np.nan
        airm_mean   = (airm_start + airm_end) / 2 if (airm_start and airm_end) else np.nan
        fwhm_mean   = (fwhm_start + fwhm_end) / 2 if (fwhm_start and fwhm_end) else np.nan

        # ── Calibration data ────────────────────────────────
        cal = cal_data.get(raw_stem, {})
        sky_adu  = cal.get("sky_adu",  np.nan)
        post_adu = cal.get("post_adu", np.nan)

        # ── Alignment data ──────────────────────────────────
        al = align_data.get(raw_stem, {})
        aligned     = al.get("aligned", False)
        stars       = al.get("stars_matched", "")
        dx          = al.get("dx_px",       np.nan)
        dy          = al.get("dy_px",       np.nan)
        rotation    = al.get("rotation_deg",np.nan)
        scale       = al.get("scale",       np.nan)
        offset_px   = np.sqrt(dx**2 + dy**2) if (np.isfinite(dx) and np.isfinite(dy)) else np.nan
        offset_asec = offset_px * PLATE_SCALE if np.isfinite(offset_px) else np.nan

        # ── Filenames ───────────────────────────────────────
        raw_fname     = raw_path.name
        cal_fname     = cal_stem + ".fits"
        aligned_fname = (cal_stem + "_aligned.fits") if aligned else ""

        rows.append({
            "ob_name"              : ob_name,
            "ob_num"               : ob_num,
            "science_state"        : state,
            "tpl_expno"            : tpl_expno,
            "tpl_nexp"             : tpl_nexp,
            "date_obs"             : date_obs,
            "mjd_obs"              : f"{mjd_obs:.7f}" if np.isfinite(mjd_obs) else "",
            "raw_filename"         : raw_fname,
            "calibrated_filename"  : cal_fname,
            "aligned_filename"     : aligned_fname,
            # Instrument
            "dit_s"                : dit,
            "ndit"                 : ndit,
            "eff_exptime_s"        : eff_exptime,
            "gain_e_per_adu"       : HAWKI_GAIN,
            "readnoise_e"          : HAWKI_RON,
            "eff_readnoise_e"      : f"{eff_ron:.3f}" if np.isfinite(eff_ron) else "",
            # Environment
            "airmass_start"        : airm_start,
            "airmass_end"          : airm_end,
            "airmass_mean"         : f"{airm_mean:.4f}" if np.isfinite(airm_mean) else "",
            "seeing_fwhm_start_as" : fwhm_start,
            "seeing_fwhm_end_as"   : fwhm_end,
            "seeing_fwhm_mean_as"  : f"{fwhm_mean:.2f}" if np.isfinite(fwhm_mean) else "",
            "humidity_pct"         : humidity,
            "windspeed_ms"         : windspeed,
            # Calibration
            "sky_level_adu"        : f"{sky_adu:.1f}" if np.isfinite(sky_adu) else "",
            "post_sky_residual_adu": f"{post_adu:.2f}" if np.isfinite(post_adu) else "",
            # Alignment
            "aligned"              : "yes" if aligned else "no",
            "align_stars_matched"  : stars,
            "align_dx_px"          : f"{dx:.3f}" if np.isfinite(dx) else "",
            "align_dy_px"          : f"{dy:.3f}" if np.isfinite(dy) else "",
            "align_offset_px"      : f"{offset_px:.3f}" if np.isfinite(offset_px) else "",
            "align_offset_arcsec"  : f"{offset_asec:.2f}" if np.isfinite(offset_asec) else "",
            "align_rotation_deg"   : f"{rotation:.5f}" if np.isfinite(rotation) else "",
            "align_scale"          : f"{scale:.6f}" if np.isfinite(scale) else "",
        })

# ============================================================
# 7. WRITE CSV
# ============================================================
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
fieldnames = list(rows[0].keys())

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV written: {OUT_CSV}")
print(f"  Rows     : {len(rows)}")
print(f"  Columns  : {len(fieldnames)}")

# ── Quick sanity print ────────────────────────────────────────────────────────
print()
print(f"{'OB':<28} {'N':>3} {'Aligned':>7} {'Sky mean':>10} "
      f"{'Seeing mean':>12} {'Airmass mean':>13}")
print("-" * 80)
for ob_name in obs_sorted:
    ob_rows = [r for r in rows if r["ob_name"] == ob_name]
    n_aligned = sum(1 for r in ob_rows if r["aligned"] == "yes")
    sky_vals  = [float(r["sky_level_adu"]) for r in ob_rows if r["sky_level_adu"]]
    see_vals  = [float(r["seeing_fwhm_mean_as"]) for r in ob_rows if r["seeing_fwhm_mean_as"]]
    air_vals  = [float(r["airmass_mean"]) for r in ob_rows if r["airmass_mean"]]
    print(f"{ob_name:<28} {len(ob_rows):>3} {n_aligned:>7} "
          f"{np.mean(sky_vals):>10.0f} "
          f"{np.mean(see_vals):>12.2f}\" "
          f"{np.mean(air_vals):>13.4f}")

print()
print(f"Columns: {', '.join(fieldnames)}")
