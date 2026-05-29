"""
scripts/smoke_test_gmc.py

Synthetic smoke test for the GMC (Global / Camera Motion Compensation)
estimator. No data files needed: builds a textured image, warps it by a
KNOWN affine to make "frame t", and checks GMC.apply() recovers it.

Covers:
  - first frame returns identity
  - pure translation is recovered (sub-pixel)
  - translation + small rotation is recovered
  - a UAV moving independently is IGNORED when its box is masked, so the
    estimate still reflects background (camera) motion
  - a featureless frame falls back to identity (no garbage warp)

Run from project root:
    python scripts/smoke_test_gmc.py
"""

import torch  # noqa: F401  # CRITICAL on Windows: precede the numpy/cv2 chain

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from src.tracking.bytetrack.gmc import GMC


H, W = 512, 640
RNG = np.random.default_rng(0)


def textured_image() -> np.ndarray:
    """Mid-gray canvas scattered with sharp blobs so ORB finds repeatable corners."""
    img = np.full((H, W), 128, np.uint8)
    for _ in range(300):
        cv2.circle(
            img,
            (int(RNG.integers(0, W)), int(RNG.integers(0, H))),
            int(RNG.integers(3, 9)),
            int(RNG.integers(0, 256)),
            -1,
        )
    img = cv2.GaussianBlur(img, (3, 3), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def affine(tx: float, ty: float, deg: float = 0.0) -> np.ndarray:
    M = cv2.getRotationMatrix2D((W / 2, H / 2), deg, 1.0)
    M[0, 2] += tx
    M[1, 2] += ty
    return M.astype(np.float32)


def warp(img: np.ndarray, M: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REFLECT)


def check(name, H_est, M_true, gmc, tol_t=1.5, tol_r=0.02) -> bool:
    dt = float(np.hypot(H_est[0, 2] - M_true[0, 2], H_est[1, 2] - M_true[1, 2]))
    dr = float(np.abs(H_est[:2, :2] - M_true[:2, :2]).max())
    ok = dt < tol_t and dr < tol_r
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         translation err = {dt:5.2f}px (tol {tol_t})   "
          f"linear err = {dr:.4f} (tol {tol_r})")
    print(f"         est t=({H_est[0,2]:7.2f},{H_est[1,2]:7.2f})  "
          f"true t=({M_true[0,2]:7.2f},{M_true[1,2]:7.2f})")
    print(f"         stats = {gmc.last_stats}")
    return ok


def main() -> int:
    print("=" * 60)
    print("GMC estimator - synthetic smoke test")
    print("=" * 60 + "\n")

    base = textured_image()
    all_ok = True

    # 1. First frame -> identity
    g = GMC(downscale=2)
    H0 = g.apply(base)
    ok = np.allclose(H0, np.eye(2, 3), atol=1e-6)
    print(f"  [{'PASS' if ok else 'FAIL'}] first frame returns identity")
    print(f"         stats = {g.last_stats}")
    all_ok &= ok

    # 2. Pure translation
    g.reset()
    M = affine(12, -7)
    g.apply(base)
    all_ok &= check("pure translation (+12, -7)", g.apply(warp(base, M)), M, g)

    # 3. Translation + small rotation
    g.reset()
    M = affine(-9, 5, deg=2.0)
    g.apply(base)
    all_ok &= check("translation + 2deg rotation", g.apply(warp(base, M)), M, g)

    # 4. Independent UAV motion ignored via masking
    g.reset()
    M = affine(10, 0)                       # background / camera motion
    f0, f1 = base.copy(), warp(base, M)
    u0, u1 = (300, 256), (340, 226)         # UAV moves (+40, -30): NOT the camera
    cv2.circle(f0, u0, 6, 255, -1)
    cv2.circle(f1, u1, 6, 255, -1)
    d0 = np.array([[u0[0] - 8, u0[1] - 8, u0[0] + 8, u0[1] + 8]], np.float32)
    d1 = np.array([[u1[0] - 8, u1[1] - 8, u1[0] + 8, u1[1] + 8]], np.float32)
    g.apply(f0, d0)
    all_ok &= check("background motion recovered, UAV masked", g.apply(f1, d1), M, g)

    # 5. Featureless frame -> identity fallback
    g.reset()
    g.apply(base)
    H_flat = g.apply(np.full((H, W, 3), 128, np.uint8))
    ok = np.allclose(H_flat, np.eye(2, 3)) and g.last_stats["fallback"]
    print(f"  [{'PASS' if ok else 'FAIL'}] featureless frame falls back to identity")
    print(f"         stats = {g.last_stats}")
    all_ok &= ok

    print("\n" + "=" * 60)
    print(f"  {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
