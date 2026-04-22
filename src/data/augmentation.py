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

        # Small rotations only — large rotations push tiny UAVs off frame
        A.Rotate(limit=10, p=0.3),

        # Slight zoom in/out — min_visibility drops boxes that become too small
        A.RandomScale(scale_limit=0.15, p=0.3),

        # Random crop that preserves at least 90% of the image area
        # Keeps small objects from being cropped out entirely
        A.RandomSizedBBoxSafeCrop(height=512, width=640, p=0.3),

        # ── Intensity transforms ──────────────────────────
        # IR imagery has no color — these operate on pixel intensity only
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),

        # Simulates sensor noise — teaches model to detect under noisy conditions
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),

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
    return A.Compose([],
        bbox_params=A.BboxParams(
            format="yolo",
            min_visibility=0.3,
            label_fields=["labels"]
        ))
