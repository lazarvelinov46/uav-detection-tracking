"""
scripts/extract_frames.py

Extracts frames from all Anti-UAV v4 videos, pairs them with their YOLO
labels, and partitions by split per the manifest at:
    configs/splits/anti_uav_v4_track3.yaml

Output structure under <output_dir>:
    images/train/<seq>_<frame_id:06d>.jpg
    images/val/<seq>_<frame_id:06d>.jpg
    labels/train/<seq>_<frame_id:06d>.txt
    labels/val/<seq>_<frame_id:06d>.txt
    data.yaml    (Ultralytics dataset config pointing at the above)

Usage:
    python scripts/extract_frames.py --output_dir "D:/uav-tracker-data/anti_uav_v4"
"""

import argparse
import shutil
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


JPG_QUALITY    = 95
PADDING_DIGITS = 6
NUM_CLASSES    = 1
CLASS_NAMES    = {0: "uav"}


def extract_sequence(
    seq_name: str,
    split: str,
    videos_dir: Path,
    processed_labels_dir: Path,
    output_dir: Path,
    jpg_quality: int = JPG_QUALITY,
) -> dict:
    """
    Extract all frames from one sequence's video, pair each with its
    existing YOLO label, and write to:
        output_dir/images/<split>/<seq>_<frame_id:06d>.jpg
        output_dir/labels/<split>/<seq>_<frame_id:06d>.txt

    Warnings (e.g. frame-count mismatches) are returned in the stats dict
    rather than printed, so the caller can render them cleanly without
    interfering with progress bars.
    """
    video_path = videos_dir / f"{seq_name}.mp4"
    label_dir  = processed_labels_dir / seq_name

    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Processed label directory not found: {label_dir}")

    label_count = len(list(label_dir.glob("*.txt")))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    reported_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    warnings = []
    if reported_frame_count != label_count:
        warnings.append(
            f"reported frame count ({reported_frame_count}) "
            f"!= label count ({label_count})"
        )

    frames_extracted    = 0
    labels_copied       = 0
    empty_labels_filled = 0

    frame_id = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        basename = f"{seq_name}_{frame_id:0{PADDING_DIGITS}d}"

        jpg_path = images_out / f"{basename}.jpg"
        ok = cv2.imwrite(str(jpg_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpg_quality])
        if not ok:
            raise RuntimeError(f"cv2.imwrite failed for {jpg_path}")
        frames_extracted += 1

        src_label = label_dir / f"{frame_id:0{PADDING_DIGITS}d}.txt"
        dst_label = labels_out / f"{basename}.txt"

        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("")
            empty_labels_filled += 1
        labels_copied += 1

        frame_id += 1

    cap.release()

    return {
        "seq_name":            seq_name,
        "split":               split,
        "reported_frame_count": reported_frame_count,
        "frames_extracted":     frames_extracted,
        "label_count_in_processed": label_count,
        "labels_copied":        labels_copied,
        "empty_labels_filled":  empty_labels_filled,
        "warnings":             warnings,
    }


def write_data_yaml(output_dir: Path) -> Path:
    """
    Write the Ultralytics data.yaml at the dataset root.

    Uses absolute path with forward slashes — works on Windows and
    avoids YAML escape ambiguity around backslashes.
    """
    data_yaml = {
        "path":  str(output_dir.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val":   "images/val",
        "nc":    NUM_CLASSES,
        "names": CLASS_NAMES,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False, default_flow_style=False)

    return yaml_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Manifest-driven frame extraction + label pairing for all sequences"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="configs/splits/anti_uav_v4_track3.yaml",
        help="Path to split manifest YAML"
    )
    parser.add_argument(
        "--videos_dir",
        type=str,
        default="data/raw/anti_uav_v4/TrainVideos",
        help="Directory containing .mp4 video files"
    )
    parser.add_argument(
        "--processed_labels_dir",
        type=str,
        default="data/processed/anti_uav_v4",
        help="Directory containing per-sequence YOLO label folders"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output root, e.g. D:/uav-tracker-data/anti_uav_v4"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    manifest_path        = Path(args.manifest)
    videos_dir           = Path(args.videos_dir)
    processed_labels_dir = Path(args.processed_labels_dir)
    output_dir           = Path(args.output_dir)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    train_seqs = manifest["train"]
    val_seqs   = manifest["val"]

    print("=" * 60)
    print("Anti-UAV v4 — Frame Extraction (Stage 2: full dataset)")
    print("=" * 60)
    print(f"\n  Manifest         : {manifest_path}")
    print(f"  Videos           : {videos_dir}")
    print(f"  Processed labels : {processed_labels_dir}")
    print(f"  Output           : {output_dir}")
    print(f"  Train sequences  : {len(train_seqs)}")
    print(f"  Val sequences    : {len(val_seqs)}\n")

    totals = {
        "train": {"frames": 0, "labels": 0, "empty_synth": 0},
        "val":   {"frames": 0, "labels": 0, "empty_synth": 0},
    }
    all_warnings = []

    for split, seqs in [("train", train_seqs), ("val", val_seqs)]:
        for seq_name in tqdm(seqs, desc=f"  {split:<5}", unit="seq"):
            stats = extract_sequence(
                seq_name             = seq_name,
                split                = split,
                videos_dir           = videos_dir,
                processed_labels_dir = processed_labels_dir,
                output_dir           = output_dir,
            )
            totals[split]["frames"]      += stats["frames_extracted"]
            totals[split]["labels"]      += stats["labels_copied"]
            totals[split]["empty_synth"] += stats["empty_labels_filled"]
            for w in stats["warnings"]:
                all_warnings.append(f"{seq_name}: {w}")

    yaml_path = write_data_yaml(output_dir)

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"\n  Train: {totals['train']['frames']:>7} frames, "
          f"{totals['train']['labels']:>7} labels "
          f"(synth empty: {totals['train']['empty_synth']})")
    print(f"  Val  : {totals['val']['frames']:>7} frames, "
          f"{totals['val']['labels']:>7} labels "
          f"(synth empty: {totals['val']['empty_synth']})")
    grand_total = totals['train']['frames'] + totals['val']['frames']
    print(f"  Total: {grand_total:>7} frames\n")

    if all_warnings:
        print(f"  Warnings ({len(all_warnings)}):")
        for w in all_warnings:
            print(f"    - {w}")
        print()

    print(f"  data.yaml written to: {yaml_path}")
    print("\n  Stage 2 complete.")


if __name__ == "__main__":
    main()
