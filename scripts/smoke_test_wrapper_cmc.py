"""
scripts/smoke_test_wrapper_cmc.py

End-to-end smoke test for ByteTrackCMCWrapper on REAL Anti-UAV data
(not synthetic). Runs the wrapper over the first N frames of one val
sequence and verifies:

  - the GMC -> tracker -> output pipeline runs without crashing
  - at least one track line is produced
  - GMC actually engages on most frames (not all fallback) -- if this
    fails, the thermal-IR-low-texture concern we flagged at design time
    is real and we need to revisit ratio / mask margin / min_inliers

Run from project root:
    python scripts/smoke_test_wrapper_cmc.py
or with overrides:
    python scripts/smoke_test_wrapper_cmc.py --seq MultiUAV-002 --n-frames 60
"""

import torch  # noqa: F401  # CRITICAL on Windows: precede the numpy chain

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2

from src.tracking.bytetrack.wrapper_cmc import ByteTrackCMCWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq",         default="MultiUAV-002")
    p.add_argument("--det-dir",     default="data/processed/detections/yolov8s_baseline")
    p.add_argument("--img-dir",     default="D:/uav-tracker-data/anti_uav_v4/images/val")
    p.add_argument("--n-frames",    type=int, default=60,
                   help="cap on number of frames to process (from frame 1)")
    # Match the locked baseline so this test is comparable to the saved bytetrack run.
    p.add_argument("--track-thresh", type=float, default=0.3)
    p.add_argument("--track-buffer", type=int,   default=30)
    p.add_argument("--match-thresh", type=float, default=0.8)
    return p.parse_args()


def load_detections_by_frame(path: Path) -> dict:
    """MOT detection file -> {frame_id: (N, 5) [x1, y1, x2, y2, conf]}."""
    by_frame = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        f = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6])
        by_frame.setdefault(f, []).append([x, y, x + w, y + h, conf])
    return {f: np.array(rows, dtype=np.float32) for f, rows in by_frame.items()}


def frame_index(path: Path) -> int:
    """MultiUAV-002_000123.jpg -> 123"""
    return int(path.stem.split("_")[-1])


def main() -> int:
    args = parse_args()
    print("=" * 60)
    print(f"ByteTrackCMCWrapper - end-to-end smoke on {args.seq}")
    print("=" * 60)

    det_path = Path(args.det_dir) / f"{args.seq}.txt"
    img_dir  = Path(args.img_dir)

    if not det_path.exists():
        print(f"[FAIL] missing detections: {det_path}")
        return 1
    if not img_dir.exists():
        print(f"[FAIL] missing image dir : {img_dir}")
        return 1

    detections = load_detections_by_frame(det_path)
    all_frames = sorted(img_dir.glob(f"{args.seq}_*.jpg"), key=frame_index)
    if not all_frames:
        print(f"[FAIL] no images found for {args.seq} in {img_dir}")
        return 1

    frames = all_frames[: args.n_frames]
    n_det_total = sum(len(v) for v in detections.values())
    print(f"\n  detections : {n_det_total:,} across {len(detections)} frames")
    print(f"  available  : {len(all_frames)} frames; processing first {len(frames)}")
    print(f"  args       : track_thresh={args.track_thresh}, "
          f"track_buffer={args.track_buffer}, match_thresh={args.match_thresh}\n")

    wrapper = ByteTrackCMCWrapper(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
    )

    n_lines = 0
    seen_ids = set()
    fallback_count = 0
    engaged_inliers = []   # only inlier counts on frames where GMC engaged
    engaged_keypoints = []

    for img_path in frames:
        fid = frame_index(img_path)
        dets = detections.get(fid, np.empty((0, 5), dtype=np.float32))
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] could not read {img_path}")
            continue

        tracks = wrapper.update(dets, img)
        for t in tracks:
            n_lines += 1
            seen_ids.add(t.track_id)

        s = wrapper.last_gmc_stats
        if s.get("fallback"):
            fallback_count += 1
        else:
            engaged_inliers.append(s.get("n_inliers", 0))
            engaged_keypoints.append(s.get("n_keypoints", 0))

    n_engaged = len(engaged_inliers)
    n_after_first = len(frames) - 1   # first frame is always fallback by design
    engage_rate = n_engaged / max(1, n_after_first)

    print("=" * 60)
    print("  Results")
    print("=" * 60)
    print(f"  track lines produced  : {n_lines:,}")
    print(f"  unique track IDs      : {len(seen_ids)}")
    print(f"  frames processed      : {len(frames)}")
    print(f"  GMC engaged           : {n_engaged}/{n_after_first} post-first-frame "
          f"({engage_rate*100:.0f}%)")
    print(f"  GMC fallbacks         : {fallback_count} total (incl. first frame)")
    if engaged_inliers:
        print(f"  GMC inliers (engaged) : mean {np.mean(engaged_inliers):.0f}, "
              f"min {np.min(engaged_inliers)}, max {np.max(engaged_inliers)}")
        print(f"  GMC keypoints (eng.)  : mean {np.mean(engaged_keypoints):.0f}, "
              f"min {np.min(engaged_keypoints)}, max {np.max(engaged_keypoints)}")
    print()

    ok1 = n_lines > 0
    print(f"  [{'PASS' if ok1 else 'FAIL'}] produced at least one track line")
    ok2 = engage_rate >= 0.5
    print(f"  [{'PASS' if ok2 else 'FAIL'}] GMC engaged on >=50% of post-first-frame frames")

    all_ok = ok1 and ok2
    print("\n" + "=" * 60)
    print(f"  {'ALL PASSED' if all_ok else 'SOME FAILED'}")
    print("=" * 60)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
