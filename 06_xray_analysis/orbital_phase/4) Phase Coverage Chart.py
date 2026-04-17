####         4          ####

# title: Phase coverage by unique nights (Chart)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/3) eso_times_with_phase.csv"
OUTPUT_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/1) Observed Orbital Phase/Analysis Outputs/4) GX 339-4 — Phase Coverage Chart"

df = pd.read_csv(INPUT_FILE)

df["MJD_NIGHT"] = np.floor(df["MJD_UTC"])
night_phase = df.groupby("MJD_NIGHT")["PHASE"].mean().values

NBINS = 100
bins = np.linspace(0, 1, NBINS+1)

plt.figure(figsize=(9,5))
plt.hist(night_phase, bins=bins, edgecolor="black", linewidth=1)

plt.xlabel("Orbital Phase")
plt.ylabel("Number of Nights (unique epochs)")
plt.title("GX 339-4 — Phase Coverage by Night (P = 1.7587 d)")
plt.xticks(np.linspace(0, 1, 11))
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# save first, then show
plt.savefig(OUTPUT_FILE + ".png", dpi=300)
plt.show()

print(f"Figure saved to: {OUTPUT_FILE}.png")
