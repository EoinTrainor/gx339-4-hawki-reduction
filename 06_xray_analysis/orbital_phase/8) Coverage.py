# title: Compute GX 339-4 orbital phase coverage segments

import pandas as pd

# --- USER SETTINGS ---
INPUT_FILE  = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/2) eso_times_mjd.csv"
OBS_TABLE_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/6) gx3394_observation_phase_table.csv"
COVERAGE_FILE  = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/7) gx3394_orbital_phase_coverage_segments.csv"

P_ORB_DAYS = 1.7587   # orbital period in days (edit if updated)

# --- EPHEMERIS (Heida et al. 2017) ---
T0_MJD = 57529.397  # MJD (phase zero)
T0_ERR = 0.003      # days (1σ)


def main():
    # --- LOAD DATA ---
    df = pd.read_csv(INPUT_FILE)

    # Require these columns (they should already exist from your pipeline)
    required_cols = ["MJD_UTC", "EXPTIME", "FULL_TIMESTAMP", "FILENAME"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns in {INPUT_FILE}: {missing}")

    # Ensure numeric EXPTIME (seconds) and drop nonsense
    df["EXPTIME"] = pd.to_numeric(df["EXPTIME"], errors="coerce")
    df = df.dropna(subset=["EXPTIME", "MJD_UTC"]).copy()
    df = df[df["EXPTIME"] > 0].copy()

    # --- PRINT T0 AND t_min FOR COMPARISON ---
    t_min = df["MJD_UTC"].min()
    print(f"T0 (Heida et al. 2017) = {T0_MJD:.6f} ± {T0_ERR:.3f} MJD")
    print(f"t_min (earliest observed MJD) = {t_min:.6f} MJD")
    print(f"Offset (t_min − T0) = {t_min - T0_MJD:.6f} days")

    # --- PER-EXPOSURE PHASE INTERVALS ---

    # Exposure duration in days
    dt_days = df["EXPTIME"] / 86400.0

    # Start / mid / end times in MJD
    df["MJD_START"] = df["MJD_UTC"]
    df["MJD_MID"]   = df["MJD_START"] + 0.5 * dt_days
    df["MJD_END"]   = df["MJD_START"] + dt_days

    # Helper: phase in [0, 1)
    def phase_from_mjd(mjd):
        return ((mjd - T0_MJD) / P_ORB_DAYS) % 1.0

    # Phases at start / mid / end (per exposure)
    df["PHASE_START"] = phase_from_mjd(df["MJD_START"])
    df["PHASE_MID"]   = phase_from_mjd(df["MJD_MID"])
    df["PHASE_END"]   = phase_from_mjd(df["MJD_END"])

    # Phase span of each exposure (Δφ = Δt / P)
    df["PHASE_SPAN"] = dt_days / P_ORB_DAYS

    # Save a clean per-exposure table
    obs_cols = [
        "FILENAME",
        "FULL_TIMESTAMP",
        "EXPTIME",
        "PHASE_START",
        "PHASE_MID",
        "PHASE_END",
        "PHASE_SPAN",
    ]
    obs_table = df[obs_cols].sort_values("PHASE_START")
    obs_table.to_csv(OBS_TABLE_FILE, index=False)
    print(f"Per-exposure observation table written to:\n  {OBS_TABLE_FILE}")

    # --- BUILD COVERAGE INTERVALS ON 0–1 ORBIT ---

    # For each exposure, construct a [phi_start, phi_end] interval around PHASE_MID
    # using PHASE_SPAN. Handle wrap-around at 0/1 by possibly splitting into two.
    intervals = []

    for _, row in df.iterrows():
        phi_mid = float(row["PHASE_MID"])
        dphi = float(row["PHASE_SPAN"])

        if dphi <= 0:
            continue

        phi_start = phi_mid - 0.5 * dphi
        phi_end   = phi_mid + 0.5 * dphi

        # Wrap-around handling: if interval crosses 0 or 1, split it
        if phi_start < 0:
            intervals.append((phi_start + 1.0, 1.0))   # tail end of orbit
            intervals.append((0.0, phi_end))           # beginning of orbit
        elif phi_end > 1:
            intervals.append((phi_start, 1.0))
            intervals.append((0.0, phi_end - 1.0))
        else:
            intervals.append((phi_start, phi_end))

    # Remove any degenerate or negative intervals
    intervals = [(max(0.0, s), min(1.0, e)) for (s, e) in intervals if e > s]

    if not intervals:
        print("No valid intervals found – check your input file.")
        return

    # --- MERGE OVERLAPPING / TOUCHING INTERVALS (UNION COVERAGE) ---
    intervals.sort(key=lambda x: x[0])

    merged = []
    cur_start, cur_end = intervals[0]

    for s, e in intervals[1:]:
        if s <= cur_end:  # overlap or just touching
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e

    merged.append((cur_start, cur_end))

    # --- CONVERT MERGED PHASE INTERVALS TO TIME DURATIONS ---
    cov_rows = []
    total_phase_covered = 0.0

    for i, (s, e) in enumerate(merged, start=1):
        width = e - s                     # Δφ
        total_phase_covered += width
        time_days = width * P_ORB_DAYS
        time_hours = time_days * 24.0
        time_minutes = time_hours * 60.0

        cov_rows.append(
            {
                "SEGMENT_ID": i,
                "PHASE_START": s,
                "PHASE_END": e,
                "PHASE_WIDTH": width,
                "TIME_DAYS_EQUIV": time_days,
                "TIME_HOURS_EQUIV": time_hours,
                "TIME_MINUTES_EQUIV": time_minutes,
            }
        )

    cov_df = pd.DataFrame(cov_rows)
    cov_df.to_csv(COVERAGE_FILE, index=False)

    print("\nMerged orbital phase coverage segments:")
    for _, r in cov_df.iterrows():
        print(
            f"Segment {int(r['SEGMENT_ID']):2d}: "
            f"phase {r['PHASE_START']:.3f} – {r['PHASE_END']:.3f} "
            f"(Δφ = {r['PHASE_WIDTH']:.3f}, "
            f"≈ {r['TIME_MINUTES_EQUIV']:.1f} min)"
        )

    print(
        f"\nTotal phase covered (union): {total_phase_covered:.3f} of orbit "
        f"({total_phase_covered * 100:.1f} %)"
    )
    print(f"Coverage segments table written to:\n  {COVERAGE_FILE}")


if __name__ == "__main__":
    main()
