"""
scripts/smoke_test_cmc_tracker.py

Smoke test for BYTETrackerCMC -- proves the wiring is correct using
synthetic detections + a known per-frame H. Three checks:

  1. With H=None, output matches the vendored BYTETracker exactly.
     (No regression to baseline behavior.)
  2. With a known H matching the actual camera motion, an object that
     would otherwise be lost (IoU = 0 between prediction and detection)
     is correctly associated -- the track keeps its ID.
  3. Without CMC (H=None), the SAME scenario loses the track, confirming
     the test is exercising the right failure mode.

Scenario: one box at (100,100)-(120,120) on frame 1; camera pans +30px
in x for frame 2, so the same world object lands at (130,100)-(150,120).
ByteTrack's constant-velocity predict (starting from zero velocity)
holds the predicted box at the frame-1 position, so without
compensation the +30px shift drives IoU to zero -- exactly the camera-
motion failure mode we built CMC to fix.

Run from project root:
    python scripts/smoke_test_cmc_tracker.py
"""

import torch  # noqa: F401  # CRITICAL on Windows: precede the numpy chain

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.tracking.bytetrack.byte_tracker import BYTETracker
from src.tracking.bytetrack.basetrack import BaseTrack
from src.tracking.bytetrack.cmc_tracker import BYTETrackerCMC


# img_info == img_size yields scale = 1 in the vendored update (no rescale)
IMG = (1, 1)


def make_args():
    return SimpleNamespace(
        track_thresh=0.3,   # matches baseline ByteTrack-0.3
        track_buffer=30,
        match_thresh=0.8,
        mot20=False,
    )


def reset_ids():
    """Reset the global STrack ID counter so independent runs use ids 1, 2, ..."""
    BaseTrack._count = 0


def fmt_out(outs):
    """One-line summary of a frame's output_stracks list."""
    if not outs:
        return "[]"
    return "[" + ", ".join(f"id={t.track_id} tlwh=({t.tlwh[0]:.0f},{t.tlwh[1]:.0f},"
                            f"{t.tlwh[2]:.0f},{t.tlwh[3]:.0f})" for t in outs) + "]"


def main() -> int:
    print("=" * 60)
    print("BYTETrackerCMC - wiring smoke test")
    print("=" * 60 + "\n")

    # Frame 1: one box at (100,100)-(120,120), high conf.
    det1 = np.array([[100, 100, 120, 120, 0.9]], dtype=np.float32)
    # Frame 2: same world object, camera panned +30px in x => box at (130,100)-(150,120).
    det2 = np.array([[130, 100, 150, 120, 0.9]], dtype=np.float32)
    # H maps frame 1 image coords -> frame 2 image coords: pure +30 in x.
    H = np.array([[1, 0, 30], [0, 1, 0]], dtype=np.float32)

    all_ok = True

    # ---- Test 1: H=None equivalence with vendored BYTETracker -------
    reset_ids()
    base = BYTETracker(make_args(), frame_rate=30)
    base_outs = [base.update(d, IMG, IMG) for d in (det1, det2)]

    reset_ids()
    cmc_none = BYTETrackerCMC(make_args(), frame_rate=30)
    cmc_outs = [cmc_none.update(d, IMG, IMG, H=None) for d in (det1, det2)]

    same_counts = all(len(a) == len(b) for a, b in zip(base_outs, cmc_outs))
    same_ids = all([t.track_id for t in a] == [t.track_id for t in b]
                   for a, b in zip(base_outs, cmc_outs))
    same_boxes = all(
        np.allclose([t.tlwh for t in a], [t.tlwh for t in b], atol=1e-6)
        for a, b in zip(base_outs, cmc_outs)
    )
    ok1 = same_counts and same_ids and same_boxes
    print(f"  [{'PASS' if ok1 else 'FAIL'}] H=None reproduces vendored BYTETracker output exactly")
    print(f"         base frame 2: {fmt_out(base_outs[1])}")
    print(f"         CMC  frame 2: {fmt_out(cmc_outs[1])}")
    all_ok &= ok1

    # ---- Test 2: With correct H, ID persists across the +30 pan ----
    reset_ids()
    trk_on = BYTETrackerCMC(make_args(), frame_rate=30)
    f1_on = trk_on.update(det1, IMG, IMG)
    f2_on = trk_on.update(det2, IMG, IMG, H=H)
    ok2 = len(f1_on) == 1 and len(f2_on) == 1 and f1_on[0].track_id == f2_on[0].track_id
    print(f"\n  [{'PASS' if ok2 else 'FAIL'}] With CMC, ID persists across +30 camera pan")
    print(f"         frame 1: {fmt_out(f1_on)}")
    print(f"         frame 2: {fmt_out(f2_on)}")
    all_ok &= ok2

    # ---- Test 3: Without CMC, same scenario loses the track --------
    reset_ids()
    trk_off = BYTETrackerCMC(make_args(), frame_rate=30)
    f1_off = trk_off.update(det1, IMG, IMG)
    f2_off = trk_off.update(det2, IMG, IMG, H=None)
    # Expected: f2_off is empty (new track created but is_activated=False
    # on frame_id != 1, so filtered out of output_stracks), and the old
    # track is now in lost_stracks.
    id1 = f1_off[0].track_id if f1_off else None
    id2 = f2_off[0].track_id if f2_off else None
    track_dropped = (id2 is None) or (id2 != id1)
    ok3 = track_dropped
    print(f"\n  [{'PASS' if ok3 else 'FAIL'}] Without CMC, the same +30 pan drops the track")
    print(f"         frame 1: {fmt_out(f1_off)}")
    print(f"         frame 2: {fmt_out(f2_off)}")
    all_ok &= ok3

    print("\n" + "=" * 60)
    print(f"  {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
