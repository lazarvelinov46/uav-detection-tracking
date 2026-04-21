"""
tests/test_dataset.py

Smoke test for VideoDataset — verifies that the dataset
indexes correctly and returns well-formed samples.
"""

import torch
from src.data.dataset import VideoDataset

# Adjust these paths to match your local setup
PROCESSED_DIR = "data/processed/anti_uav_v4"
VIDEOS_DIR    = "data/raw/anti_uav_v4/TrainVideos"


def test_dataset_loads():
    dataset = VideoDataset(
        processed_dir=PROCESSED_DIR,
        videos_dir=VIDEOS_DIR
    )

    # Check total sample count matches what we expect
    print(f"\n  Total samples : {len(dataset)}")
    assert len(dataset) > 0, "Dataset is empty"

    # Load the very first sample
    image, labels, meta = dataset[0]

    print(f"  Image shape   : {image.shape}")
    print(f"  Image dtype   : {image.dtype}")
    print(f"  Image range   : [{image.min():.3f}, {image.max():.3f}]")
    print(f"  Labels shape  : {labels.shape}")
    print(f"  Meta          : {meta}")

    # Shape checks
    assert image.shape[0] == 3,  "Expected 3 channels"
    assert image.shape[1] == 512, "Expected height 512"
    assert image.shape[2] == 640, "Expected width 640"

    # Value range check
    assert image.min() >= 0.0 and image.max() <= 1.0, "Pixel values out of [0,1]"

    # Labels should be (N, 5) or (0, 5)
    assert labels.ndim == 2 and labels.shape[1] == 5, "Unexpected label shape"

    print("\n  All checks passed.")


if __name__ == "__main__":
    test_dataset_loads()
