"""
tests/test_augmentation.py

Smoke test for the augmentation pipeline.
Verifies that transforms run without errors and that
bounding box coordinates remain valid after augmentation.
"""
import torch  # must import before albumentations on Windows (DLL loading order)
import numpy as np
from src.data.augmentation import get_train_transforms, get_val_transforms


def test_transforms():
    # Simulate a single IR frame — float32, HWC, range [0, 1]
    image = np.random.rand(512, 640, 3).astype(np.float32)

    # Two small UAV boxes in YOLO format [cx, cy, w, h] normalized
    bboxes = [
        [0.5,  0.5,  0.02, 0.02],
        [0.25, 0.75, 0.015, 0.018],
    ]
    labels = [0, 0]  # both class UAV

    for name, transform in [("train", get_train_transforms()),
                             ("val",   get_val_transforms())]:
        result = transform(image=image, bboxes=bboxes, labels=labels)

        out_image  = result["image"]
        out_bboxes = result["bboxes"]

        # Image shape must be preserved
        assert out_image.shape == (512, 640, 3), f"{name}: unexpected image shape"

        # All returned boxes must have coordinates in [0, 1]
        for box in out_bboxes:
            assert all(0.0 <= v <= 1.0 for v in box), \
                f"{name}: box coordinate out of range: {box}"

        print(f"  [{name}] OK — {len(out_bboxes)} boxes retained")

    print("\n  All augmentation checks passed.")


if __name__ == "__main__":
    test_transforms()
