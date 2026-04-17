# title: Build Segment/Gap orbit table from merged segments + plot 1D timeline

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------- USER PATHS ----------------
BASE_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs"

# Put the relative filename from your folder here:
SEGMENTS_REL = r"7) gx3394_orbital_phase_coverage_segments.csv"

# Outputs (same folder path as requested)
OUTPUT_TABLE_REL = r"8) gx3394_orbit_segments_and_gaps.csv"
OUTPUT_PLOT_REL  = r"8) GX339-4_Orbit_Segments_Gaps_Timeline.png"
# -------------------------------------------

segments_path = os.path.join(BASE_DIR, SEGMENTS_REL)
output_table_path = os.path.join(BASE_DIR, OUTPUT_TABLE_REL)
output_plot_path = os.path.join(BASE_DIR, OUTPUT_PLOT_REL)

# Load merged segments
seg = pd.read_csv(segments_path)

required = ["PHASE_START", "PHASE_END"]
for c in required:
    if c not in seg.columns:
        raise KeyError(f"Column '{c}' not found in {segments_path}")

# Sort by phase start
seg = seg.sort_values("PHASE_START").reset_index(drop=True)

# Clean numeric types
seg["PHASE_START"] = seg["PHASE_START"].astype(float)
seg["PHASE_END"]   = seg["PHASE_END"].astype(float)

# Small helper
P_DAYS = 1.7587
def phase_to_minutes(dphi: float) -> float:
    return dphi * P_DAYS * 24.0 * 60.0

ORBIT_MINUTES = P_DAYS * 24.0 * 60.0

rows = []
seg_id = 0
gap_id = 0

# Build alternating Segment / Gap, including the wrap-around gap
for i in range(len(seg)):
    # Segment i
    seg_id += 1
    s0 = float(seg.loc[i, "PHASE_START"])
    s1 = float(seg.loc[i, "PHASE_END"])
    dphi = max(0.0, s1 - s0)

    rows.append({
        "Label": f"Segment {seg_id}",
        "Status": "Observed",
        "Phase_start": s0,
        "Phase_end": s1,
        "Delta_phase": dphi,
        "Duration_minutes": phase_to_minutes(dphi),
    })

    # Gap after segment i
    if i < len(seg) - 1:
        g0 = s1
        g1 = float(seg.loc[i + 1, "PHASE_START"])
        dphi_g = max(0.0, g1 - g0)

        gap_id += 1
        rows.append({
            "Label": f"Gap {gap_id}",
            "Status": "Unobserved",
            "Phase_start": g0,
            "Phase_end": g1,
            "Delta_phase": dphi_g,
            "Duration_minutes": phase_to_minutes(dphi_g),
        })
    else:
        # Wrap-around gap: last_end -> 1.0 plus 0.0 -> first_start
        last_end = s1
        first_start = float(seg.loc[0, "PHASE_START"])

        part1 = max(0.0, 1.0 - last_end)     # last_end -> 1
        part2 = max(0.0, first_start - 0.0)  # 0 -> first_start
        dphi_wrap = part1 + part2

        gap_id += 1
        rows.append({
            "Label": f"Gap {gap_id}",
            "Status": "Unobserved",
            "Phase_start": last_end,
            "Phase_end": first_start,  # special: indicates wrap
            "Delta_phase": dphi_wrap,
            "Duration_minutes": phase_to_minutes(dphi_wrap),
            "Wrap_gap": True,
            "Wrap_part1_start": last_end,
            "Wrap_part1_end": 1.0,
            "Wrap_part2_start": 0.0,
            "Wrap_part2_end": first_start
        })

orbit = pd.DataFrame(rows)

# Ensure wrap columns exist for non-wrap rows
for c in ["Wrap_gap","Wrap_part1_start","Wrap_part1_end","Wrap_part2_start","Wrap_part2_end"]:
    if c not in orbit.columns:
        orbit[c] = np.nan

orbit["Wrap_gap"] = orbit["Wrap_gap"].fillna(False)

# Round for neatness
orbit["Phase_start"] = orbit["Phase_start"].round(6)
orbit["Phase_end"]   = orbit["Phase_end"].round(6)
orbit["Delta_phase"] = orbit["Delta_phase"].round(6)
orbit["Duration_minutes"] = orbit["Duration_minutes"].round(1)

# Save table
orbit.to_csv(output_table_path, index=False)

# Print in the style you want
print("Orbit coverage in periodic order (Segment / Gap / Segment / Gap ...):\n")
for _, r in orbit.iterrows():
    if not r["Wrap_gap"]:
        print(f'{r["Label"]}: {r["Phase_start"]:.3f} – {r["Phase_end"]:.3f}  ({r["Duration_minutes"]:.1f} min)')
    else:
        # Wrap gap prints as two intervals
        a0, a1 = float(r["Wrap_part1_start"]), float(r["Wrap_part1_end"])
        b0, b1 = float(r["Wrap_part2_start"]), float(r["Wrap_part2_end"])
        print(f'{r["Label"]}: {a0:.3f} – {a1:.3f} and {b0:.3f} – {b1:.3f}  ({r["Duration_minutes"]:.1f} min)')

print(f"\nCSV saved to:\n{output_table_path}")

# ---------------- ANALYTICS (console only) ----------------
obs = orbit[orbit["Status"] == "Observed"].copy()
gaps = orbit[orbit["Status"] == "Unobserved"].copy()

total_obs_phase = obs["Delta_phase"].sum()
total_gap_phase = gaps["Delta_phase"].sum()

total_obs_min = obs["Duration_minutes"].sum()
total_gap_min = gaps["Duration_minutes"].sum()

# sanity: should be ~1.0 and ~ORBIT_MINUTES
phase_total = total_obs_phase + total_gap_phase
minutes_total = total_obs_min + total_gap_min

print("\n--- Coverage analytics ---")
print(f"Orbital period: {P_DAYS:.4f} d  (= {ORBIT_MINUTES:.1f} min)")
print(f"Total (obs+gap) phase check: {phase_total:.6f} (should be 1.000000)")
print(f"Total (obs+gap) minutes check: {minutes_total:.1f} (should be {ORBIT_MINUTES:.1f})")

pct_obs = 100.0 * total_obs_phase
pct_gap = 100.0 * total_gap_phase

print(f"\n% Orbit Observed:   {pct_obs:.1f}%  ({total_obs_min:.1f} min)")
print(f"% Orbit Unobserved: {pct_gap:.1f}%  ({total_gap_min:.1f} min)")

print(f"\nNumber of observed segments: {len(obs)}")
print(f"Number of gaps:              {len(gaps)}")
print(f"Fragmentation (segments per orbit): {len(obs)}")

# Segment duration stats
seg_durs = obs["Duration_minutes"].values
gap_durs = gaps["Duration_minutes"].values

def stats_line(name, arr):
    return (
        f"{name}: mean {np.mean(arr):.1f} min, median {np.median(arr):.1f} min, "
        f"min {np.min(arr):.1f} min, max {np.max(arr):.1f} min"
    )

if len(seg_durs) > 0:
    print("\nObserved segment duration stats")
    print(stats_line("Segments", seg_durs))

if len(gap_durs) > 0:
    print("\nUnobserved gap duration stats")
    print(stats_line("Gaps", gap_durs))

# Largest gap details (phase and minutes)
if len(gaps) > 0:
    i_max = gaps["Delta_phase"].astype(float).values.argmax()
    g = gaps.iloc[i_max]

    if not bool(g["Wrap_gap"]):
        print("\nLargest gap")
        print(f'  {g["Label"]}: phase {float(g["Phase_start"]):.3f} – {float(g["Phase_end"]):.3f} '
              f'(Δφ = {float(g["Delta_phase"]):.3f}, ≈ {float(g["Duration_minutes"]):.1f} min)')
    else:
        a0, a1 = float(g["Wrap_part1_start"]), float(g["Wrap_part1_end"])
        b0, b1 = float(g["Wrap_part2_start"]), float(g["Wrap_part2_end"])
        print("\nLargest gap (wrap-around)")
        print(f'  {g["Label"]}: phase {a0:.3f} – {a1:.3f} and {b0:.3f} – {b1:.3f} '
              f'(Δφ = {float(g["Delta_phase"]):.3f}, ≈ {float(g["Duration_minutes"]):.1f} min)')

# Duty cycle is the same as percent observed, but we print it explicitly
print(f"\nDuty cycle (observed/total): {total_obs_phase:.3f}")

# ---------------- Plot 1D timeline ----------------
plt.figure(figsize=(12, 2.7))
y = 0.5
height = 0.45

# Plot segments (green)
for _, r in orbit.iterrows():
    if r["Status"] == "Observed":
        start = float(r["Phase_start"])
        width = float(r["Phase_end"] - r["Phase_start"])
        plt.broken_barh([(start, width)], (y - height/2, height),
                        facecolors="green", edgecolors="black", linewidth=1)

# Plot gaps (red) including wrap gap split into two pieces
for _, r in orbit.iterrows():
    if r["Status"] == "Unobserved":
        if not r["Wrap_gap"]:
            start = float(r["Phase_start"])
            width = float(r["Phase_end"] - r["Phase_start"])
            plt.broken_barh([(start, width)], (y - height/2, height),
                            facecolors="red", edgecolors="black", linewidth=1)
        else:
            a0, a1 = float(r["Wrap_part1_start"]), float(r["Wrap_part1_end"])
            b0, b1 = float(r["Wrap_part2_start"]), float(r["Wrap_part2_end"])
            if a1 > a0:
                plt.broken_barh([(a0, a1-a0)], (y - height/2, height),
                                facecolors="red", edgecolors="black", linewidth=1)
            if b1 > b0:
                plt.broken_barh([(b0, b1-b0)], (y - height/2, height),
                                facecolors="red", edgecolors="black", linewidth=1)

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.yticks([])
plt.xticks(np.linspace(0, 1, 11))
plt.grid(axis="x", alpha=0.25)

plt.xlabel("Orbital Phase (0 → 1)")
plt.title(f"GX 339-4 — Observed (green) vs Unobserved (red) Orbit Coverage (P = {P_DAYS} d)")

legend_elements = [
    Patch(facecolor="green", edgecolor="black", label="Observed (Segment)"),
    Patch(facecolor="red", edgecolor="black", label="Unobserved (Gap)")
]
plt.legend(handles=legend_elements, loc="upper right")

plt.tight_layout()
plt.savefig(output_plot_path, dpi=300)
plt.show()

print(f"\nPlot saved to:\n{output_plot_path}")

# --- Coverage quality metrics (position / spread) ---

# Coverage fraction (already)
C = float(total_obs_phase)  # in [0,1]

# Gap sizes in phase (include wrap gap already in 'gaps')
gap_sizes = gaps["Delta_phase"].astype(float).values

# 1) Uniformity of gap spacing
if len(gap_sizes) >= 2:
    cv_gap = np.std(gap_sizes) / np.mean(gap_sizes)
    Q_uniform = 1.0 / (1.0 + cv_gap)
else:
    cv_gap = np.nan
    Q_uniform = np.nan

# 2) Largest-gap penalty
G_max = float(np.max(gap_sizes)) if len(gap_sizes) > 0 else np.nan
Q_maxgap = 1.0 - G_max if np.isfinite(G_max) else np.nan

# 3) Overall suitability score
Q_overall = C * Q_uniform * Q_maxgap if np.isfinite(Q_uniform) and np.isfinite(Q_maxgap) else np.nan

print("\n--- Coverage quality (spread / orbit reconstruction suitability) ---")
print(f"Coverage fraction C: {C:.3f}")

if np.isfinite(cv_gap):
    print(f"Gap coefficient of variation (CV_gap): {cv_gap:.3f}  (lower is better)")
    print(f"Gap uniformity score Q_uniform = 1/(1+CV): {Q_uniform:.3f}")
else:
    print("Gap uniformity score: N/A (not enough gaps)")

if np.isfinite(G_max):
    print(f"Largest gap G_max: {G_max:.3f} of orbit")
    print(f"Largest-gap score Q_maxgap = 1 - G_max: {Q_maxgap:.3f}")
else:
    print("Largest-gap score: N/A")

if np.isfinite(Q_overall):
    print(f"Overall suitability score Q_overall = C * Q_uniform * Q_maxgap: {Q_overall:.3f}")
else:
    print("Overall suitability score: N/A")
