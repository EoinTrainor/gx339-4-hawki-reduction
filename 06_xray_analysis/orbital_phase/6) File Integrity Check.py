####        6           ####

# title: Check HAWK-I FITS integrity and list bad files

import os
from astropy.io import fits

FITS_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/SI_Chronologic_DATE_OBS" 
FITS_EXTENSIONS = (".fits", ".fit", ".fz")

good_files = []
bad_files = []

for root, _, files in os.walk(FITS_DIR):
    for fname in files:
        if not fname.lower().endswith(FITS_EXTENSIONS):
            continue

        full_path = os.path.join(root, fname)

        try:
            with fits.open(full_path, memmap=False) as hdul:
                # Force a read of header/data to catch truncation
                _ = hdul[0].header
                # if you want, also touch data:
                # _ = hdul[0].data
            good_files.append(full_path)
        except Exception as e:
            print(f"[BAD] {full_path}: {e}")
            bad_files.append(full_path)

print()
print(f"Total FITS checked: {len(good_files) + len(bad_files)}")
print(f"Good FITS: {len(good_files)}")
print(f"Bad FITS: {len(bad_files)}")

if bad_files:
    print("\nList of bad files:")
    for bf in bad_files:
        print("  ", bf)
