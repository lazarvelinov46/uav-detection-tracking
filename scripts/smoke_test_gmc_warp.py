"""
scripts/smoke_test_gmc_warp.py

Unit test for multi_gmc -- the xyah-adapted Kalman-state warp that
applies the camera-motion affine H to each track before IoU association.

Uses lightweight stub tracks (objects with .mean and .covariance) so the
warp math is verified in isolation, with no tracker or detector involved.
Expectations are hand-computed.

Run from project root:
    python scripts/smoke_test_gmc_warp.py
"""

import torch  # noqa: F401  # CRITICAL on Windows: precede the numpy chain

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.tracking.bytetrack.gmc import multi_gmc


class StubTrack:
    """Minimal stand-in for STrack: just the two attributes multi_gmc touches."""
    def __init__(self, mean, cov):
        self.mean = np.array(mean, dtype=float)
        self.covariance = np.array(cov, dtype=float)


# state layout: [cx, cy, a, h, vcx, vcy, va, vh]
MEAN0 = [100, 200, 0.5, 40, 2, -1, 0.0, 0.5]
COV0 = np.diag([4, 4, 0.01, 9, 1, 1, 0.001, 0.25]).astype(float)


def approx(a, b, tol=1e-9):
    return np.allclose(a, b, atol=tol)


def main() -> int:
    print("=" * 60)
    print("multi_gmc (xyah warp) - unit test")
    print("=" * 60 + "\n")
    ok = True

    # 1. Pure translation (+10, -5): only the center shifts; cov unchanged.
    t = StubTrack(MEAN0, COV0)
    multi_gmc([t], np.array([[1, 0, 10], [0, 1, -5]], float))
    c1 = approx(t.mean, [110, 195, 0.5, 40, 2, -1, 0, 0.5]) and approx(t.covariance, COV0)
    print(f"  [{'PASS' if c1 else 'FAIL'}] translation: center shifts, rest + covariance unchanged")
    ok &= c1

    # 2. Pure scale (x2): center/height/velocity/vh scale; aspect invariant;
    #    covariance diagonal scales by the square of each block factor.
    t = StubTrack(MEAN0, COV0)
    multi_gmc([t], np.array([[2, 0, 0], [0, 2, 0]], float))
    exp_mean = [200, 400, 0.5, 80, 4, -2, 0, 1.0]
    exp_cov = np.diag([16, 16, 0.01, 36, 4, 4, 0.001, 1.0]).astype(float)
    c2 = approx(t.mean, exp_mean) and approx(t.covariance, exp_cov)
    print(f"  [{'PASS' if c2 else 'FAIL'}] scale x2: center/h/vel/vh x2, aspect invariant, cov diag x4")
    ok &= c2

    # 3. Rotation 90deg about origin (det=1 -> s=1): center & velocity rotate,
    #    height & aspect unchanged, covariance stays symmetric.
    t = StubTrack(MEAN0, COV0)
    multi_gmc([t], np.array([[0, -1, 0], [1, 0, 0]], float))
    c3 = approx(t.mean, [-200, 100, 0.5, 40, 1, 2, 0, 0.5]) and approx(t.covariance, t.covariance.T)
    print(f"  [{'PASS' if c3 else 'FAIL'}] rotation 90deg: center/vel rotate, h/a unchanged, cov symmetric")
    ok &= c3

    # 4. None H and empty list are no-ops.
    t = StubTrack(MEAN0, COV0)
    multi_gmc([t], None)
    multi_gmc([], np.eye(2, 3))
    c4 = approx(t.mean, MEAN0)
    print(f"  [{'PASS' if c4 else 'FAIL'}] None H and empty list are no-ops")
    ok &= c4

    print("\n" + "=" * 60)
    print(f"  {'ALL PASSED' if ok else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
