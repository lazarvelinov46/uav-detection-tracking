"""
scripts/convert_to_yolo.py

Converts Anti-UAV v4 Track 3 annotations from MOT format
to YOLO format (normalized center x, center y, width, height).

Usage:
    python scripts/convert_to_yolo.py \
        --data_root "C:/UAV Detection and Tracking/MultiUAV_Train" \
        --output_dir "data/processed/anti_uav_v4"
"""

import argparse
from pathlib import Path

# Fixed for all Anti-UAV v4 Track 3 videos (verified across all 200 sequences)
FRAME_WIDTH  = 640
FRAME_HEIGHT = 512
UAV_CLASS_ID = 0

def parse_args():
    parser = argparse.ArgumentParser(description="Convert Anti-UAV v4 annotations to YOLO format")
    # Path to the raw dataset (MultiUAV_Train folder)
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to MultiUAV_Train root folder"
    )
    # Where the converted YOLO label files will be written
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to write converted YOLO labels"
    )
    return parser.parse_args()

def parse_mot_label(label_path: Path):
    """
    Reads a MOT-format label file and returns a dict mapping
    frame_id -> list of YOLO-format annotation strings.
    Each annotation is: "class_id cx cy w h"
    """
    frames = {}  # frame_id -> list of yolo annotation strings

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(",")
            frame_id = int(parts[0])
            # parts[1] is track_id — not needed for detection, skipped
            x_left  = float(parts[2])
            y_top   = float(parts[3])
            w       = float(parts[4])
            h       = float(parts[5])

            # Convert absolute pixel coords to normalized YOLO center format
            cx = (x_left + w / 2) / FRAME_WIDTH
            cy = (y_top  + h / 2) / FRAME_HEIGHT
            nw = w / FRAME_WIDTH
            nh = h / FRAME_HEIGHT

            # Clamp to [0, 1] to handle boxes that slightly exceed frame boundary
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            yolo_line = f"{UAV_CLASS_ID} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"

            if frame_id not in frames:
                frames[frame_id] = []
            frames[frame_id].append(yolo_line)

    return frames

def main():
    args = parse_args()
    data_root  = Path(args.data_root)
    output_dir = Path(args.output_dir)

    # All raw annotation files live here
    labels_dir = data_root / "TrainLabels"

    # Collect all sequence label files, sorted for deterministic ordering
    label_files = sorted(labels_dir.glob("*.txt"))

    print("=" * 55)
    print("Anti-UAV v4 — YOLO Conversion")
    print("=" * 55)
    print(f"\n  Input  : {labels_dir}")
    print(f"  Output : {output_dir}")
    print(f"  Sequences to convert: {len(label_files)}")


if __name__ == "__main__":
    main()
