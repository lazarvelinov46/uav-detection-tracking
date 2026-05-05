"""
scripts/create_splits.py

Stratified train/val split for Anti-UAV v4 Track 3.

Reads raw MOT-format labels from TrainLabels/, computes per-sequence
mean UAV density, bins sequences into terciles by density, and produces
a deterministic 80/20 random split within each bin. Output is a YAML
manifest that drives both the YOLOv8 and YOLOX training pipelines.

Usage:
    python scripts/create_splits.py \
        --data_root "C:/UAV Detection and Tracking/MultiUAV_Train" \
        --output configs/splits/anti_uav_v4_track3.yaml
"""

import argparse
import random
from pathlib import Path

import numpy as np
import yaml


RANDOM_SEED  = 42
VAL_FRACTION = 0.2
NUM_BINS     = 3  # terciles: low / medium / high density


def compute_sequence_density(label_path: Path) -> dict:
    """
    Reads one MOT-format label file and returns density stats for the sequence.

    Density is computed as: total_annotations / max_frame_id.
    Using max_frame_id approximates true sequence length — safe given
    the 0.1% empty-frame rate observed in EDA.
    """
    seq_name = label_path.stem

    num_annotations = 0
    max_frame_id    = 0

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frame_id = int(line.split(",")[0])
            num_annotations += 1
            if frame_id > max_frame_id:
                max_frame_id = frame_id

    mean_density = num_annotations / max_frame_id if max_frame_id > 0 else 0.0

    return {
        "seq_name":        seq_name,
        "num_frames":      max_frame_id,
        "num_annotations": num_annotations,
        "mean_density":    mean_density,
    }


def assign_density_bin(density: float, low_max: float, high_min: float) -> str:
    """Map a density value to its tercile bin name."""
    if density <= low_max:
        return "low"
    elif density >= high_min:
        return "high"
    else:
        return "medium"


def stratified_split(stats: list, seed: int):
    """
    Bin sequences into terciles by density RANK (not by density value),
    then perform an 80/20 random split within each bin.

    Rank-based binning guarantees equal-sized bins regardless of ties
    in density values — important when many sequences share the same
    density (e.g. constant-count sequences clustering at d=5.00).

    Returns (train_list, val_list, distribution, bin_density_ranges)
    where bin_density_ranges documents the actual density spans of
    each bin for the manifest.
    """
    rng = random.Random(seed)
    n = len(stats)

    # Sort by (density, seq_name) — seq_name is a deterministic tiebreaker
    sorted_stats = sorted(stats, key=lambda s: (s["mean_density"], s["seq_name"]))

    # Cut into three roughly-equal chunks by position
    b1 = n // 3
    b2 = 2 * n // 3
    bin_groups = {
        "low":    sorted_stats[:b1],
        "medium": sorted_stats[b1:b2],
        "high":   sorted_stats[b2:],
    }

    # Document each bin's actual density range (for the manifest)
    bin_density_ranges = {
        name: {
            "min": round(group[0]["mean_density"],  4),
            "max": round(group[-1]["mean_density"], 4),
        }
        for name, group in bin_groups.items()
    }

    train, val = [], []
    distribution = {"train": {}, "val": {}}

    for bin_name in ["low", "medium", "high"]:
        names = sorted([s["seq_name"] for s in bin_groups[bin_name]])
        rng.shuffle(names)

        val_size   = max(1, round(len(names) * VAL_FRACTION))
        train_size = len(names) - val_size

        train.extend(names[:train_size])
        val.extend(names[train_size:])

        distribution["train"][bin_name] = train_size
        distribution["val"][bin_name]   = val_size

    train.sort()
    val.sort()

    return train, val, distribution, bin_density_ranges


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create stratified train/val split for Anti-UAV v4 Track 3"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Path to dataset root containing TrainLabels/"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/splits/anti_uav_v4_track3.yaml",
        help="Output path for the split manifest YAML"
    )
    return parser.parse_args()


def main():
    args        = parse_args()
    data_root   = Path(args.data_root)
    labels_dir  = data_root / "TrainLabels"
    output_path = Path(args.output)

    label_files = sorted(labels_dir.glob("*.txt"))

    print("=" * 60)
    print("Anti-UAV v4 — Stratified Train/Val Split")
    print("=" * 60)
    print(f"\n  Input     : {labels_dir}")
    print(f"  Output    : {output_path}")
    print(f"  Sequences : {len(label_files)}")
    print(f"  Strategy  : tercile stratification by mean UAV density")
    print(f"  Seed      : {RANDOM_SEED}\n")

    # Stage 1: density computation
    stats = [compute_sequence_density(p) for p in label_files]

    # Stage 2: tercile boundaries
    # densities = np.array([s["mean_density"] for s in stats])
    # low_max  = float(np.quantile(densities, 1.0 / 3.0))
    # high_min = float(np.quantile(densities, 2.0 / 3.0))

    # print(f"  Tercile boundaries:")
    # print(f"    low    : density <= {low_max:.2f}")
    # print(f"    medium : {low_max:.2f} < density < {high_min:.2f}")
    # print(f"    high   : density >= {high_min:.2f}\n")

    # Stage 3: stratified random split
    # Stage 2 + 3: rank-based bins, stratified split
    train, val, distribution, bin_ranges = stratified_split(stats, RANDOM_SEED)

    print(f"  Bin density ranges (rank-based terciles):")
    for bin_name in ["low", "medium", "high"]:
        r = bin_ranges[bin_name]
        print(f"    {bin_name:<8}: [{r['min']:.2f}, {r['max']:.2f}]")
    print()

    print(f"  Split sizes:")
    print(f"    {'bin':<10}{'train':>8}{'val':>8}{'total':>8}")
    print(f"    {'-'*10}{'-'*8}{'-'*8}{'-'*8}")
    for bin_name in ["low", "medium", "high"]:
        t = distribution["train"][bin_name]
        v = distribution["val"][bin_name]
        print(f"    {bin_name:<10}{t:>8}{v:>8}{t+v:>8}")
    print(f"    {'-'*10}{'-'*8}{'-'*8}{'-'*8}")
    print(f"    {'total':<10}{len(train):>8}{len(val):>8}{len(train)+len(val):>8}\n")

    # Sanity check: mean density in each split should be similar
    seq_to_density = {s["seq_name"]: s["mean_density"] for s in stats}
    train_mean = np.mean([seq_to_density[name] for name in train])
    val_mean   = np.mean([seq_to_density[name] for name in val])

    print(f"  Mean density per split (should be similar — sanity check):")
    print(f"    train : {train_mean:.2f}")
    print(f"    val   : {val_mean:.2f}\n")

    # Stage 4: write YAML manifest
    manifest = {
        "dataset":         "anti_uav_v4_track3",
        "random_seed":     RANDOM_SEED,
        "strategy":        "stratified_by_uav_density_terciles",
        "val_fraction":    VAL_FRACTION,
        "total_sequences": len(stats),
        "bin_density_ranges": bin_ranges,
        "binning_method":     "rank_based_terciles",
        "split_distribution": {
            "train": {**distribution["train"], "total": len(train)},
            "val":   {**distribution["val"],   "total": len(val)},
        },
        "train": train,
        "val":   val,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

    print(f"  Manifest written to: {output_path}")
    print("\n  Split complete.")


if __name__ == "__main__":
    main()
