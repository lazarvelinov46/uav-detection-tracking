"""
scripts/prep_kaggle_upload.py

Packages the D: training dataset into a single .zip on C: for Kaggle upload.
Uses ZIP_STORED (no compression) because the data is mostly already-compressed
JPGs — compression would add time without saving meaningful space.
"""

import time
import zipfile
from pathlib import Path

SRC = Path("D:/uav-tracker-data/anti_uav_v4")
OUT = Path("C:/kaggle-upload-staging/anti_uav_v4.zip")

OUT.parent.mkdir(parents=True, exist_ok=True)

print(f"Zipping {SRC}")
print(f"     -> {OUT}\n")

start = time.time()
count = 0

with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
    for path in SRC.rglob("*"):
        if path.is_file():
            arcname = path.relative_to(SRC.parent)  # keeps "anti_uav_v4/" prefix inside the zip
            zf.write(path, arcname)
            count += 1
            if count % 10000 == 0:
                elapsed = time.time() - start
                print(f"  {count:>7,} files packaged  ({elapsed:.1f}s elapsed)")

elapsed = time.time() - start
size_gb = OUT.stat().st_size / 1e9
print(f"\n  Done: {count:,} files, {size_gb:.2f} GB, {elapsed:.1f}s")
