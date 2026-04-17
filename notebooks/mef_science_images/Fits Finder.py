# title: Check if specific FITS files are in the extracted timestamps CSV

import pandas as pd

CSV_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Analysis Outputs/1) eso_extracted_timestamps.csv"

df = pd.read_csv(CSV_FILE)

if "FILENAME" not in df.columns:
    raise KeyError("Column 'FILENAME' not found in the CSV. Make sure the FITS extraction script writes it.")

target_files = [
    "ADP.2025-07-08T07-31-50.422.fits",
    "ADP.2025-08-05T08-45-21.048.fits",
    "ADP.2025-08-05T08-46-56.121.fits",
    "ADP.2025-09-04T08-49-42.610.fits",
    "ADP.2025-10-07T08-32-26.876.fits",
]
    
print("Checking presence of target FITS in CSV:\n")

for name in target_files:
    present = (df["FILENAME"] == name).any()
    status = "FOUND" if present else "MISSING"
    print(f"{name}: {status}")
