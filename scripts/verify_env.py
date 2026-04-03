"""
scripts/verify_env.py

Run this after setting up the environment on any new machine.
Checks every package in the stack and reports pass/fail clearly.

Usage:
    python scripts/verify_env.py
"""

import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  [PASS] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def main():
    print("=" * 55)
    print("UAV Tracker — Environment Verification")
    print("=" * 55)

    results = []

    # ── Python ────────────────────────────────────────────
    print("\n  Python")
    results.append(check(
        "Python version",
        lambda: sys.version.split()[0]
    ))

    # ── Core ML stack ─────────────────────────────────────
    print("\n  Core ML")
    import torch  # torch must come first on Windows

    results.append(check(
        "PyTorch",
        lambda: torch.__version__
    ))
    results.append(check(
        "CUDA available",
        lambda: str(torch.cuda.is_available())
    ))
    results.append(check(
        "torch.zeros (CPU compute check)",
        lambda: str(torch.zeros(3, 3).shape)
    ))

    # ── Computer Vision ───────────────────────────────────
    print("\n  Computer Vision")
    results.append(check(
        "OpenCV",
        lambda: __import__("cv2").__version__
    ))
    results.append(check(
        "Albumentations",
        lambda: __import__("albumentations").__version__
    ))
    results.append(check(
        "Pillow",
        lambda: __import__("PIL").__version__
    ))

    # ── Ultralytics ───────────────────────────────────────
    print("\n  Detection")
    results.append(check(
        "Ultralytics (YOLOv8)",
        lambda: __import__("ultralytics").__version__
    ))

    # ── Scientific stack ──────────────────────────────────
    print("\n  Scientific stack")
    results.append(check(
        "NumPy",
        lambda: __import__("numpy").__version__
    ))
    results.append(check(
        "SciPy",
        lambda: __import__("scipy").__version__
    ))
    results.append(check(
        "Pandas",
        lambda: __import__("pandas").__version__
    ))
    results.append(check(
        "Matplotlib",
        lambda: __import__("matplotlib").__version__
    ))

    # ── Tracking utilities ────────────────────────────────
    print("\n  Tracking utilities")
    results.append(check(
        "filterpy (Kalman filter)",
        lambda: __import__("filterpy").__version__
    ))
    results.append(check(
        "lap (linear assignment)",
        lambda: str(__import__("lap"))
    ))
    results.append(check(
        "motmetrics (MOTA/IDF1)",
        lambda: __import__("motmetrics").__version__
    ))

    # ── Summary ───────────────────────────────────────────
    passed = sum(results)
    total = len(results)
    print()
    print("=" * 55)
    if passed == total:
        print(f"  All {total} checks passed. Environment is ready.")
    else:
        print(f"  {passed}/{total} checks passed. Fix failures above.")
    print("=" * 55)


if __name__ == "__main__":
    main()
