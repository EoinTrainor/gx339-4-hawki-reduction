# Create downloadable CSV file for star data

import pandas as pd

# Create dataframe
data = {
    "Star": ["Star 3", "Star 2", "Star 4", "Star 1", "Star 5"],
    "FWHM(px)": [7.25, 7.39, 7.37, 7.33, 7.37],
    "maxSNR3": [10124.092, 9060.960, 7346.895, 7343.828, 6570.243],
    "opt r(px)": [5.07, 5.17, 5.16, 5.13, 5.16],
    "sens": [0.005, 0.004, 0.005, 0.005, 0.004],
    "coreN": [2261, 2283, 2291, 2247, 2225],
    "sat": [0, 0, 0, 0, 0],
    "fit": [0, 0, 0, 0, 0],
    "ann(px)": ["22-37", "22-37", "22-37", "22-37", "22-37"],
    "score": [479202.80, 431132.24, 349980.11, 346470.11, 308635.14]
}

df = pd.DataFrame(data)

# Save to CSV
file_path = "C:/Users/40328449/OneDrive - University College Cork/GX 339-4/APERTURE SNR.csv"
df.to_csv(file_path, index=False)

file_path
