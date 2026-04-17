# Plot gx_over_star3_net_ratio vs Midpoint Orbital Timescale with OB regions

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Option A: load from your CSV
# -----------------------------
GX_CSV = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/6) Aperture Outputs/gx_all_frames.csv"

use_csv = True  # set False to use the hard coded arrays below

if use_csv:
    df = pd.read_csv(GX_CSV)

    # If your column names match what you showed, this should work.
    # If not, rename the strings below to match your actual CSV headers.
    x_mid = df["Midpoint Orbital Timescale (1.7587 Days)"].to_numpy(dtype=float)
    y = df["gx_over_star3_net_ratio"].to_numpy(dtype=float)
    yerr = df["gx_over_star3_net_ratio_err"].to_numpy(dtype=float)

    x_start = df["Start Orbital Timescale (1.7587 Days)"].to_numpy(dtype=float)
    x_end = df["End Orbital Timescale (1.7587 Days)"].to_numpy(dtype=float)

else:
    # --------------------------------
    # Option B: hard coded arrays
    # --------------------------------
    y = np.array([
        0.034953074, 0.034775473, 0.035978175, 0.033465159, 0.033089329,
        0.038097653, 0.042794078, 0.045403017, 0.042660798, 0.046620197,
        0.830779059, 1.752271683, 1.087227651
    ], dtype=float)

    yerr = np.array([
        8.64e-05, 9.59e-05, 1.19075e-04, 6.76e-05, 7.35e-05,
        7.06e-05, 6.40e-05, 1.01244e-04, 1.18254e-04, 8.50e-05,
        1.91618e-04, 3.71365e-04, 1.95879e-04
    ], dtype=float)

    x_mid = np.array([
        0.638061732, 0.747614386, 0.85505974, 0.302567166, 0.291310941,
        0.58656584, 0.831556261, 0.932111979, 0.599567639, 0.611886764,
        0.672268087, 0.197703659, 0.136477957
    ], dtype=float)

    x_start = np.array([
        0.630361917, 0.73991457, 0.847359925, 0.294867351, 0.283611124,
        0.578866023, 0.823856444, 0.924412164, 0.596902319, 0.605371534,
        0.664568272, 0.190003843, 0.12877814
    ], dtype=float)

    x_end = np.array([
        0.645761547, 0.755314203, 0.862759556, 0.310266981, 0.299010758,
        0.594265657, 0.839256078, 0.939811794, 0.60223296, 0.618401995,
        0.679967902, 0.205403474, 0.144177774
    ], dtype=float)

# -----------------------------
# Plot
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# Errorbar scatter
ax.errorbar(
    x_mid, y, yerr=yerr,
    fmt="o", markersize=5, capsize=3, elinewidth=1,
)

# OB regions: dashed black vertical lines at start and end, and transparent blue span between them
for s, e in zip(x_start, x_end):
    ax.axvline(s, linestyle="--", color="black", linewidth=1)
    ax.axvline(e, linestyle="--", color="black", linewidth=1)
    ax.axvspan(s, e, color="tab:blue", alpha=0.18)

ax.set_xlabel("Midpoint Orbital Timescale (1.7587 Days)")
ax.set_ylabel("gx_over_star3_net_ratio")
ax.set_title("gx_over_star3_net_ratio with uncertainty and OB regions")

ax.grid(True, alpha=0.25)

# Optional: make x limits slightly wider than the min/max OB region
xmin = np.nanmin(np.concatenate([x_start, x_end]))
xmax = np.nanmax(np.concatenate([x_start, x_end]))
pad = 0.02 * (xmax - xmin) if xmax > xmin else 0.01
ax.set_xlim(xmin - pad, xmax + pad)

plt.tight_layout()
plt.show()

# Optional save
# fig.savefig("gx_over_star3_net_ratio_vs_orbital_timescale.png", dpi=300, bbox_inches="tight")