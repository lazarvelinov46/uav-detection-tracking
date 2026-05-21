"""
scripts/vendor_bytetrack.py

One-time setup: vendor the canonical ByteTrack tracker code from
FoundationVision/ByteTrack (formerly ifzhang/ByteTrack) into
src/tracking/bytetrack/.

Downloads four files and applies four patches needed for use outside
the YOLOX framework, on modern NumPy, on Windows without a C++
compiler, and with the Windows torch-vs-numpy DLL ordering issue:

  1. Relative imports: replace `from yolox.tracker import X` with
     `from . import X` in byte_tracker.py and matching.py.
  2. NumPy compatibility: replace deprecated `np.float` with `float`
     across all files (removed in NumPy >=1.24).
  3. cython_bbox: wrap the import in try/except, falling back to a
     pure-numpy IoU implementation if cython_bbox isn't installed.
     `pip install cython_bbox` gives us canonical perf when it works,
     without requiring it for the code to run.
  4. Windows DLL ordering: prepend `import torch` to byte_tracker.py
     so torch's DLLs register before numpy perturbs the DLL search
     path. Without this, `import torch` later in the file fails with
     WinError 127 on Windows.

Run once from project root:
    python scripts/vendor_bytetrack.py
"""

import re
import sys
import urllib.request
from pathlib import Path


REPO_BASE = "https://raw.githubusercontent.com/FoundationVision/ByteTrack/main/yolox/tracker"
FILES     = ["byte_tracker.py", "kalman_filter.py", "matching.py", "basetrack.py"]

TRACKING_DIR = Path("src/tracking")
TARGET_DIR   = Path("src/tracking/bytetrack")


# Patch 3 replacement: try cython_bbox first, fall back to pure-numpy IoU.
# Use the C extension if available (canonical, ~2-3x faster on small
# matrices); otherwise use numpy (negligible perf hit at our scale).
CYTHON_BBOX_FALLBACK = """\
# Patched: try cython_bbox (canonical, C-compiled) first, fall back to a
# pure-numpy IoU implementation if cython_bbox isn't installed. Both compute
# the same IoU; the numpy fallback is ~2-3x slower on small matrices but
# the difference is negligible at our scale (~30M pair-IoUs total).
try:
    from cython_bbox import bbox_overlaps as bbox_ious
except ImportError:
    def bbox_ious(boxes1, boxes2):
        boxes1 = np.asarray(boxes1, dtype=float)
        boxes2 = np.asarray(boxes2, dtype=float)
        if boxes1.size == 0 or boxes2.size == 0:
            return np.zeros((len(boxes1), len(boxes2)), dtype=float)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        lt = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
        wh = np.maximum(rb - lt, 0)
        inter = wh[..., 0] * wh[..., 1]
        union = area1[:, None] + area2[None, :] - inter
        return inter / np.maximum(union, 1e-9)\
"""


# Patch 4 prefix: prepended to byte_tracker.py so torch loads before numpy on Windows.
TORCH_FIRST_PREFIX = "import torch  # Patched: Windows DLL ordering — must precede numpy\n"


def download_file(url: str, dest: Path):
    print(f"  Downloading: {dest.name}")
    with urllib.request.urlopen(url) as response:
        dest.write_bytes(response.read())


def patch_file(path: Path):
    """Applies all relevant patches to a single vendored file. Idempotent."""
    content  = path.read_text(encoding="utf-8")
    original = content

    # Patch 1: relative imports (only present in byte_tracker.py and matching.py)
    content = content.replace(
        "from yolox.tracker import matching",
        "from . import matching",
    )
    content = content.replace(
        "from yolox.tracker import kalman_filter",
        "from . import kalman_filter",
    )

    # Patch 2: NumPy deprecation. Word boundary avoids matching np.float32/np.float64.
    content = re.sub(r"\bnp\.float\b", "float", content)

    # Patch 3: cython_bbox -> try/except with numpy fallback (matching.py only)
    content = content.replace(
        "from cython_bbox import bbox_overlaps as bbox_ious",
        CYTHON_BBOX_FALLBACK,
    )

    # Patch 4: Windows DLL ordering — prepend torch import to byte_tracker.py.
    # Check the first non-whitespace line to stay idempotent across re-runs.
    if path.name == "byte_tracker.py":
        if not content.lstrip().startswith("import torch"):
            content = TORCH_FIRST_PREFIX + content

    if content != original:
        path.write_text(content, encoding="utf-8")
        print(f"  Patched    : {path.name}")
    else:
        print(f"  No patches : {path.name}")


def main():
    print("=" * 60)
    print("Vendoring ByteTrack tracker code")
    print("=" * 60)
    print(f"  Source : {REPO_BASE}")
    print(f"  Target : {TARGET_DIR}")

    # --- Create directory structure and package __init__.py files ----
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    (TRACKING_DIR / "__init__.py").touch()
    (TARGET_DIR   / "__init__.py").touch()
    print("\n  Package directories ready.")

    # --- Download canonical files ------------------------------------
    print("\n  Downloading canonical files...")
    for filename in FILES:
        download_file(f"{REPO_BASE}/{filename}", TARGET_DIR / filename)

    # --- Apply patches -----------------------------------------------
    print("\n  Applying patches...")
    for filename in FILES:
        patch_file(TARGET_DIR / filename)

    # --- Verify import works -----------------------------------------
    # Patch 4 prepended `import torch` to byte_tracker.py so importing
    # BYTETracker triggers a clean torch load before numpy. We also
    # import torch explicitly here to match the project convention
    # of "torch first on Windows."
    print("\n  Verifying import...")
    sys.path.insert(0, str(Path.cwd()))
    try:
        import torch  # Required first on Windows for DLL ordering
        from src.tracking.bytetrack.byte_tracker import BYTETracker
        print("  [OK] BYTETracker imported successfully")
        print(f"       {BYTETracker}")
    except Exception as e:
        print("  [FAIL] Import error:")
        print(f"         {type(e).__name__}: {e}")
        print(f"\n  Inspect the vendored files in {TARGET_DIR}.")
        sys.exit(1)

    print("\n  Vendoring complete.\n")


if __name__ == "__main__":
    main()
