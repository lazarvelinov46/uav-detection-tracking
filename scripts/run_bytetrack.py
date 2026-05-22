"""
scripts/run_bytetrack.py

Runs ByteTrack on the per-sequence YOLOv8s detection files produced
by run_detection_inference.py. For each val sequence, instantiates a
fresh ByteTrackWrapper, feeds detections frame by frame (empty frames
included for counter alignment), and writes per-sequence MOT-format
tracker output files ready for TrackEval.

Output line format (MOT16 convention):
    frame, track_id, x, y, w, h, conf, -1, -1, -1

Usage (from project root):
    python scripts/run_bytetrack.py
or with overrides, e.g.:
    python scripts/run_bytetrack.py --mot20 --track_thresh 0.6
"""

import torch  # CRITICAL: must precede numpy in import chain on Windows

import argparse
import sys
from pathlib import Path

# Make `from src.tracking...` resolve when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import yaml

from src.tracking.bytetrack.wrapper import ByteTrackWrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ByteTrack on val-set detection files."
    )
    parser.add_argument("--detections_dir", type=str,   default="data/processed/detections/yolov8s_baseline")
    parser.add_argument("--images_dir",     type=str,   default="D:/uav-tracker-data/anti_uav_v4/images/val")
    parser.add_argument("--split_manifest", type=str,   default="configs/splits/anti_uav_v4_track3.yaml")
    parser.add_argument("--output_dir",     type=str,   default="data/processed/tracker_outputs/bytetrack/yolov8s_baseline")
    # ByteTrack hyperparameters
    parser.add_argument("--track_thresh",   type=float, default=0.5,  help="High/low detection split threshold")
    parser.add_argument("--track_buffer",   type=int,   default=30,   help="Frames to remember a lost track")
    parser.add_argument("--match_thresh",   type=float, default=0.8,  help="First-stage IoU matching threshold")
    parser.add_argument("--mot20",          action="store_true",      help="Crowded-scene mode")
    parser.add_argument("--frame_rate",     type=int,   default=30,   help="Scales track_buffer to time")
    return parser.parse_args()


def load_val_sequences(split_manifest: Path) -> list:
    with open(split_manifest, "r") as f:
        manifest = yaml.safe_load(f)
    return manifest["val"]


def load_detections_by_frame(path: Path) -> dict:
    """
    Reads a MOT-format detection file and returns
    {frame_id: (N, 5) np.array of [x1, y1, x2, y2, conf]}.
    Frames absent from the file (no detections) are absent from the dict.
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
        by_frame.setdefault(frame_id, []).append([x, y, x + w, y + h, conf])
    return {f: np.array(rows, dtype=np.float32) for f, rows in by_frame.items()}


def get_sequence_length(images_dir: Path, seq_name: str) -> int:
    """Counts .jpg files matching the sequence to determine its length."""
    return len(list(images_dir.glob(f"{seq_name}_*.jpg")))


def track_to_mot_line(frame_id: int, t) -> str:
    """Formats one Track as an MOT16-style tracker output line."""
    return (
        f"{frame_id},{t.track_id},"
        f"{t.x:.2f},{t.y:.2f},{t.w:.2f},{t.h:.2f},"
        f"{t.score:.4f},-1,-1,-1"
    )


def run_tracker_on_sequence(detections_by_frame, seq_length, args):
    """Tracks one sequence; returns (output_lines, unique_track_ids)."""
    tracker = ByteTrackWrapper(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=args.mot20,
        frame_rate=args.frame_rate,
    )

    output_lines = []
    seen_ids     = set()

    for frame_id in range(1, seq_length + 1):
        dets   = detections_by_frame.get(frame_id, np.empty((0, 5), dtype=np.float32))
        tracks = tracker.update(dets)
        for t in tracks:
            output_lines.append(track_to_mot_line(frame_id, t))
            seen_ids.add(t.track_id)

    return output_lines, seen_ids


def main():
    args = parse_args()

    detections_dir = Path(args.detections_dir)
    images_dir     = Path(args.images_dir)
    split_manifest = Path(args.split_manifest)
    output_dir     = Path(args.output_dir)

    print("=" * 60)
    print("ByteTrack — Per-Sequence Tracking (Val Set)")
    print("=" * 60)
    print(f"\n  Detections   : {detections_dir}")
    print(f"  Images       : {images_dir}")
    print(f"  Manifest     : {split_manifest}")
    print(f"  Output       : {output_dir}")
    print(f"  track_thresh : {args.track_thresh}")
    print(f"  match_thresh : {args.match_thresh}")
    print(f"  track_buffer : {args.track_buffer}")
    print(f"  mot20        : {args.mot20}")

    assert detections_dir.exists(), f"Missing detections directory: {detections_dir}"
    assert images_dir.exists(),     f"Missing images directory: {images_dir}"
    assert split_manifest.exists(), f"Missing split manifest: {split_manifest}"

    val_sequences = load_val_sequences(split_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Val sequences: {len(val_sequences)}\n")

    total_lines  = 0
    total_frames = 0
    total_tracks = 0

    for i, seq_name in enumerate(val_sequences, start=1):
        det_path = detections_dir / f"{seq_name}.txt"
        if not det_path.exists():
            print(f"  [WARN] No detection file for {seq_name}, skipping.")
            continue

        seq_length = get_sequence_length(images_dir, seq_name)
        if seq_length == 0:
            print(f"  [WARN] No images found for {seq_name}, skipping.")
            continue

        detections_by_frame = load_detections_by_frame(det_path)
        output_path = output_dir / f"{seq_name}.txt"

        print(f"  [{i:>2}/{len(val_sequences)}] {seq_name}  ({seq_length:>5} frames)")

        lines, seen_ids = run_tracker_on_sequence(detections_by_frame, seq_length, args)
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        total_lines  += len(lines)
        total_frames += seq_length
        total_tracks += len(seen_ids)
        print(f"        -> {len(lines):>7,} track lines, {len(seen_ids):>4} unique IDs")

    print("\n" + "=" * 60)
    print("  Done.")
    print(f"  Sequences processed : {len(val_sequences)}")
    print(f"  Frames processed    : {total_frames:,}")
    print(f"  Track lines written : {total_lines:,}")
    print(f"  Tracks (all seqs)   : {total_tracks:,}")
    if total_frames > 0:
        print(f"  Avg tracks/frame    : {total_lines / total_frames:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
