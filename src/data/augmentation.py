"""
src/data/augmentation.py

Albumentations augmentation pipelines for small UAV detection.
All transforms handle bounding box coordinate updates automatically.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transforms():
    """
    Augmentation pipeline for training.
    Designed for small object detection on IR imagery.

    All transforms use bbox_params to ensure bounding boxes
    are correctly updated alongside the image transformation.
    """
    return A.Compose([

        # ── Geometric transforms ──────────────────────────
        # Horizontal flip — most common and safest augmentation
        A.HorizontalFlip(p=0.5),

        # Slight zoom in/out — min_visibility drops boxes that become too small
        A.RandomScale(scale_limit=0.15, p=0.3),


        # replaces RandomSizedBBoxSafeCrop
        # Old
        # A.ShiftScaleRotate(
        #     shift_limit=0.05,   # translate up to 5% of image size
        #     scale_limit=0.15,   # zoom in/out up to 15%
        #     rotate_limit=10,    # rotate up to 10 degrees
        #     p=0.3
        # ),

        # replaces RandomSizedBBoxSafeCrop
        A.Affine(
            translate_percent=0.05, # translate up to 5% of image size
            scale=(0.85, 1.15),     # zoom in/out up to 15%
            rotate=(-10, 10),       # rotate up to 10 degrees
            p=0.3
        ),

        # ── Intensity transforms ──────────────────────────
        # IR imagery has no color — these operate on pixel intensity only
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Simulates sensor noise — teaches model to detect under noisy conditions
        # Old
        # A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
        A.GaussNoise(std_range=(0.02, 0.08), p=0.3),


        # Slight blur — simulates atmospheric haze or focus variation
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),

    ],
    bbox_params=A.BboxParams(
        format="yolo",            # expects [cx, cy, w, h] normalized
        min_visibility=0.3,       # drop boxes that are less than 30% visible after transform
        label_fields=["labels"]   # tells Albumentations which field holds class IDs
    ))

def get_val_transforms():
    """
    Validation pipeline — no augmentation, format only.
    Keeps validation deterministic and comparable across runs.
    """
    return A.Compose([
        A.NoOp()  # keeps bbox_params processor happy with no actual transformation
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        min_visibility=0.3,
        label_fields=["labels"]
    ))
