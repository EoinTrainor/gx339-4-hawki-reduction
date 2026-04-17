# title: Extract ESO ZIP and fix invalid Windows characters in filenames

import zipfile
import os
from pathlib import Path
import shutil

# TODO: change this to the path of your ESO zip file:
ZIP_FILE = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/archive (1).zip"

# TODO: choose a short, local folder with enough space (avoid OneDrive):
OUTPUT_DIR = r"C:/Users/40328449/OneDrive - University College Cork/GX 339-4/Fits Test Set"

# Make sure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

def make_windows_safe(name: str) -> str:
    """
    Replace characters that Windows does not allow in filenames.
    Here we mainly care about ':' in the ESO timestamps.
    """
    bad_chars = {
        ":": "-",   # 2025-06-04T07:48:45.939 → 2025-06-04T07-48-45.939
        "*": "_",
        "?": "_",
        "\"": "_",
        "<": "_",
        ">": "_",
        "|": "_"
    }
    for bad, repl in bad_chars.items():
        name = name.replace(bad, repl)
    return name

with zipfile.ZipFile(ZIP_FILE, 'r') as z:
    for member in z.infolist():
        # Skip directories
        if member.is_dir():
            continue

        original_name = member.filename  # path inside the zip
        safe_name = make_windows_safe(original_name)

        # Preserve any subdirectories inside the zip (if there are any)
        rel_path = Path(safe_name)
        target_path = Path(OUTPUT_DIR) / rel_path

        # Ensure subdirectories exist
        target_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Extracting: {original_name}  ->  {target_path}")

        # Manually copy file contents to the new (safe) path
        with z.open(member, 'r') as src, open(target_path, 'wb') as dst:
            shutil.copyfileobj(src, dst)

print("Extraction complete.")
print(f"Files extracted to: {OUTPUT_DIR}")
