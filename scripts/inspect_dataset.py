"""
scripts/inspect_dataset.py

Inspects the Anti-UAV v4 Track 3 dataset structure.
Reports sequence counts, paired files, and per-sequence statistics.

Usage:
    python scripts/inspect_dataset.py --data_root "C:/UAV Detection and Tracking/MultiUAV_Train"
"""

import argparse
from pathlib import Path

def find_sequences(data_root: Path):
    videos_dir = data_root / "TrainVideos"
    labels_dir = data_root / "TrainLabels"

    video_files = {p.stem: p for p in sorted(videos_dir.glob("*.mp4"))}
    label_files = {p.stem: p for p in sorted(labels_dir.glob("*.txt"))}

    return video_files, label_files

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect Anti-UAV v4 dataset structure")
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to the MultiUAV_Train root folder"
    )
    return parser.parse_args()

def inspect_sequence(label_path: Path):
    frame_ids = []
    track_ids = set()

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            frame_ids.append(int(parts[0]))
            track_ids.add(int(parts[1]))

    total_annotations = len(frame_ids)
    unique_frames = len(set(frame_ids))
    unique_tracks = len(track_ids)
    max_frame = max(frame_ids)
    annotations_per_frame = total_annotations / unique_frames if unique_frames > 0 else 0

    return {
        "unique_frames": unique_frames,
        "max_frame": max_frame,
        "unique_tracks": unique_tracks,
        "total_annotations": total_annotations,
        "avg_per_frame": annotations_per_frame,
    }

def main():
    args = parse_args()
    data_root = Path(args.data_root)

    print("=" * 55)
    print("Anti-UAV v4 — Dataset Inspector")
    print("=" * 55)
    print(f"\n  Root  : {data_root}")
    print(f"  Exists: {data_root.exists()}")

    video_files, label_files = find_sequences(data_root)

    print(f"\n  Videos found : {len(video_files)}")
    print(f"  Labels found : {len(label_files)}")

    paired = sorted(set(video_files) & set(label_files))
    only_video = sorted(set(video_files) - set(label_files))
    only_label = sorted(set(label_files) - set(video_files))

    print(f"\n  Paired sequences : {len(paired)}")
    print(f"  Video only       : {len(only_video)}")
    print(f"  Label only       : {len(only_label)}")

    if only_video:
        print(f"\n  [WARN] Videos without labels: {only_video}")
    if only_label:
        print(f"\n  [WARN] Labels without videos: {only_label}")

    print("\n" + "-" * 55)
    print("  Per-sequence statistics")
    print("-" * 55)

    stats = []
    for name in paired:
        s = inspect_sequence(label_files[name])
        stats.append(s)
        print(f"  {name}  frames={s['unique_frames']}  "
              f"tracks={s['unique_tracks']}  "
              f"avg_uav/frame={s['avg_per_frame']:.1f}")

    print("\n" + "=" * 55)
    print("  Dataset summary")
    print("=" * 55)

    all_frames = [s["unique_frames"] for s in stats]
    all_tracks = [s["unique_tracks"] for s in stats]
    all_avg = [s["avg_per_frame"] for s in stats]

    print(f"  Total sequences      : {len(stats)}")
    print(f"  Total frames         : {sum(all_frames)}")
    print(f"  Avg frames/sequence  : {sum(all_frames)/len(all_frames):.1f}")
    print(f"  Min/Max frames       : {min(all_frames)} / {max(all_frames)}")
    print(f"  Min/Max tracks       : {min(all_tracks)} / {max(all_tracks)}")
    print(f"  Avg UAVs/frame       : {sum(all_avg)/len(all_avg):.1f}")
    print(f"  Max UAVs/frame (avg) : {max(all_avg):.1f}")


if __name__ == "__main__":
    main()
