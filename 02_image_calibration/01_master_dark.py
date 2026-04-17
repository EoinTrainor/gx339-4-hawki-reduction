"""
01_master_dark.py
-----------------
Step 1: Build master dark frame and pixel mask.

Median-combines all dark frames matched to the science exposure time.
Applies sigma-clipping to identify and mask hot/bad pixels.

Outputs saved to config.MASTERS_DIR:
  - master_dark.fits       : Median master dark
  - pixel_mask.fits        : Boolean bad pixel mask (True = bad)

Usage:
    python pipeline/01_master_dark.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# TODO: implement after running 00_diagnostics.py and confirming data structure
# Pipeline will be built here step by step.

def main():
    print("Step 1: Master dark — not yet implemented.")
    print("Run 00_diagnostics.py first to confirm your data structure.")

if __name__ == "__main__":
    main()
