"""
scripts/visualize_tracking.py

Renders detections or tracker outputs onto a sequence's frames and
writes an mp4 for visual inspection.

  --source detections : boxes colored by confidence band — green if
      conf >= 0.6 (would initialize a ByteTrack track), yellow 0.5-0.6,
      red < 0.5 (associate-only). Diagnostic for poor-tracking sequences.
  --source tracks      : boxes colored by track ID (each ID keeps a
      consistent color across frames). Qualitative tracking check.

Usage (from project root):
    python scripts/visualize_tracking.py --sequence MultiUAV-002 --source tracks
    python scripts/visualize_tracking.py --sequence MultiUAV-278 --source detections
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Visualize detections or tracks on frames.")
    p.add_argument("--sequence",       type=str, required=True, help="e.g. MultiUAV-278")
    p.add_argument("--source",         type=str, choices=["detections", "tracks"], default="tracks")
    p.add_argument("--images_dir",     type=str, default="D:/uav-tracker-data/anti_uav_v4/images/val")
    p.add_argument("--detections_dir", type=str, default="data/processed/detections/yolov8s_baseline")
    p.add_argument("--tracks_dir",     type=str, default="data/processed/tracker_outputs/bytetrack/yolov8s_baseline")
    p.add_argument("--output_dir",     type=str, default="outputs/visualizations")
    p.add_argument("--fps",            type=int, default=30)
    p.add_argument("--max_frames",     type=int, default=0, help="0 = all; else cap for a quick preview")
    return p.parse_args()


def load_boxes_by_frame(path: Path) -> dict:
    """Reads a MOT-format file -> {frame_id: [(obj_id, x, y, w, h, conf), ...]}."""
    by_frame = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        frame_id = int(parts[0])
        obj_id   = int(parts[1])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6])
        by_frame.setdefault(frame_id, []).append((obj_id, x, y, w, h, conf))
    return by_frame


def conf_to_color(conf: float) -> tuple:
    """BGR by confidence band, matching ByteTrack's init thresholds."""
    if conf >= 0.6:
        return (0, 200, 0)      # green: would initialize a track
    elif conf >= 0.5:
        return (0, 200, 200)    # yellow: high but won't initialize
    else:
        return (0, 0, 220)      # red: low, associate-only


def id_to_color(obj_id: int) -> tuple:
    """Deterministic BGR per track ID (stable across frames)."""
    rng = np.random.RandomState(obj_id)
    return tuple(int(c) for c in rng.randint(60, 256, size=3))


def main():
    args = parse_args()

    images_dir = Path(args.images_dir)
    seq        = args.sequence
    src_path   = (Path(args.detections_dir) if args.source == "detections"
                  else Path(args.tracks_dir)) / f"{seq}.txt"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{seq}_{args.source}.mp4"

    print("=" * 60)
    print(f"Visualizing {args.source} for {seq}")
    print("=" * 60)
    print(f"  Source : {src_path}")
    print(f"  Output : {out_path}")

    assert images_dir.exists(), f"Missing images dir: {images_dir}"
    assert src_path.exists(),   f"Missing source file: {src_path}"

    boxes_by_frame = load_boxes_by_frame(src_path)

    frame_paths = sorted(images_dir.glob(f"{seq}_*.jpg"))
    if args.max_frames > 0:
        frame_paths = frame_paths[:args.max_frames]
    assert frame_paths, f"No images found for {seq}"

    first = cv2.imread(str(frame_paths[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    if not writer.isOpened():
        print("  [ERROR] VideoWriter failed to open — check OpenCV ffmpeg support.")
        sys.exit(1)

    print(f"\n  Rendering {len(frame_paths)} frames at {w}x{h}...")

    drawn = 0
    for fp in frame_paths:
        frame_id = int(fp.stem.split("_")[-1])
        img = cv2.imread(str(fp))

        for (obj_id, bx, by, bw, bh, conf) in boxes_by_frame.get(frame_id, []):
            x1, y1 = int(round(bx)), int(round(by))
            x2, y2 = int(round(bx + bw)), int(round(by + bh))
            if args.source == "detections":
                color, label = conf_to_color(conf), f"{int(conf * 100)}"
            else:
                color, label = id_to_color(obj_id), f"{obj_id}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.putText(img, label, (x1, max(0, y1 - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            drawn += 1

        cv2.putText(img, f"{seq}  frame {frame_id}", (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)

    writer.release()
    print(f"  Boxes drawn : {drawn:,}")
    print(f"\n  Saved: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
