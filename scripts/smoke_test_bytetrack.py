"""
scripts/smoke_test_bytetrack.py

Smoke test for the ByteTrack wrapper.

Reads detections from one val sequence's MOT-format detection file
(produced by run_detection_inference.py), feeds them through the
ByteTrackWrapper frame by frame, and prints the track outputs for
the first few frames. Verifies:
  - The wrapper instantiates and runs.
  - Track IDs are assigned on frame 1.
  - Track IDs are preserved across consecutive frames (the whole
    point of tracking).
"""

import torch  # CRITICAL on Windows: must precede numpy in import chain

from pathlib import Path
import numpy as np

from src.tracking.bytetrack.wrapper import ByteTrackWrapper


DETECTIONS_PATH    = Path("data/processed/detections/yolov8s_baseline/MultiUAV-002.txt")
NUM_FRAMES_TO_SHOW = 3


def load_detections_by_frame(path: Path) -> dict:
    """
    Reads a MOT-format detection file and returns
    {frame_id: (N, 5) array of [x1, y1, x2, y2, conf]}.
    Frames not in the file (no detections) won't appear in the dict.
    """
    by_frame = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        # MOT detection line: frame, id, x, y, w, h, conf, x3d, y3d, z3d
        frame_id = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6])
        # Convert MOT xywh to ByteTrack xyxy
        by_frame.setdefault(frame_id, []).append([x, y, x + w, y + h, conf])
    return {f: np.array(rows, dtype=np.float32) for f, rows in by_frame.items()}


def main():
    print("=" * 55)
    print("ByteTrack Wrapper — Smoke Test")
    print("=" * 55)
    print(f"\n  Detections : {DETECTIONS_PATH}")
    assert DETECTIONS_PATH.exists(), f"Missing: {DETECTIONS_PATH}"

    # Load detections grouped by frame
    print("\n  Loading detections...")
    detections_by_frame = load_detections_by_frame(DETECTIONS_PATH)
    print(f"  Frames with dets : {len(detections_by_frame)}")
    print(f"  Max frame_id     : {max(detections_by_frame.keys())}")

    # Create wrapper (paper-default hyperparameters)
    tracker = ByteTrackWrapper(
        track_thresh=0.5,
        track_buffer=30,
        match_thresh=0.8,
        mot20=False,
        img_size=(512, 640),
    )
    print("\n  Tracker created  : track_thresh=0.5, buffer=30, match=0.8, mot20=False")

    # Feed frames 1 .. NUM_FRAMES_TO_SHOW
    print(f"\n  Processing first {NUM_FRAMES_TO_SHOW} frames...")
    seen_ids_per_frame = []
    for frame_id in range(1, NUM_FRAMES_TO_SHOW + 1):
        dets = detections_by_frame.get(frame_id, np.empty((0, 5), dtype=np.float32))
        tracks = tracker.update(dets)
        seen_ids_per_frame.append({t.track_id for t in tracks})

        print(f"\n  Frame {frame_id}: {len(dets)} detections -> {len(tracks)} tracks")
        for i, t in enumerate(tracks[:5]):
            print(f"    [{i}] id={t.track_id:>3d}  "
                  f"({t.x:7.2f}, {t.y:7.2f}, {t.w:6.2f}, {t.h:6.2f})  conf={t.score:.3f}")
        if len(tracks) > 5:
            print(f"    ... and {len(tracks) - 5} more")

    # Cross-frame ID continuity check
    if len(seen_ids_per_frame) >= 2:
        carried = seen_ids_per_frame[0] & seen_ids_per_frame[1]
        new_in_2 = seen_ids_per_frame[1] - seen_ids_per_frame[0]
        print(f"\n  ID continuity 1->2 : {len(carried)} carried, {len(new_in_2)} new")

    print("\n  Smoke test done.\n")


if __name__ == "__main__":
    main()
