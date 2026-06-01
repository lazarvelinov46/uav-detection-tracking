"""
scripts/diagnose_gmc.py

Pinpoints WHERE the GMC pipeline collapsed on real thermal-IR frames.
The smoke test reported 0/59 engaged frames on MultiUAV-002 with a
static camera, which means the failure is upstream of motion (a static
pair should trivially RANSAC to identity). This script runs raw ORB on
one frame under several variants and reports keypoint counts side by
side so we can see exactly which step zeroes out.

Variants tested:
  - ORB on raw downscaled grayscale, no mask, default fastThreshold=20
  - Same, with the actual detection mask
  - Same, with lowered fastThreshold (5, 2)
  - Same, after CLAHE (Contrast Limited Adaptive Histogram Equalization)
    -- the standard preprocessing for low-dynamic-range thermal-IR ORB

Also prints frame intensity statistics: a tiny std (say < 15) means
thermal contrast is genuinely low and ORB's default FAST threshold of
20 is unreachable -- CLAHE is then almost certainly the fix.

Saves two PNG overlays to diagnostics_gmc/ so you can SEE where
keypoints land (or don't):
  - keypoints_raw.png      : raw downscaled gray + detected keypoints
  - keypoints_clahe.png    : CLAHE-enhanced  + detected keypoints

Run:
    python scripts/diagnose_gmc.py
or with overrides:
    python scripts/diagnose_gmc.py --seq MultiUAV-002 --frame 30
"""

import torch  # noqa: F401  # CRITICAL on Windows: precede the numpy chain

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seq",     default="MultiUAV-002")
    p.add_argument("--frame",   type=int, default=1)
    p.add_argument("--det-dir", default="data/processed/detections/yolov8s_baseline")
    p.add_argument("--img-dir", default="D:/uav-tracker-data/anti_uav_v4/images/val")
    p.add_argument("--out-dir", default="diagnostics_gmc")
    return p.parse_args()


def load_boxes_at_frame(det_path: Path, frame_id: int) -> np.ndarray:
    """Return (N, 4) xyxy boxes at a given frame from a MOT detection file."""
    boxes = []
    for line in det_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        if int(parts[0]) != frame_id:
            continue
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        boxes.append([x, y, x + w, y + h])
    return np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 4), dtype=np.float32)


def build_mask(shape, boxes_full, downscale=2) -> np.ndarray:
    """Build the same detection mask GMC uses (255 except inside boxes)."""
    h, w = shape
    mask = np.full((h, w), 255, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes_full:
        xi1 = max(0, int(np.floor(x1 / downscale)))
        yi1 = max(0, int(np.floor(y1 / downscale)))
        xi2 = min(w, int(np.ceil(x2 / downscale)))
        yi2 = min(h, int(np.ceil(y2 / downscale)))
        if xi2 > xi1 and yi2 > yi1:
            mask[yi1:yi2, xi1:xi2] = 0
    return mask


def n_kpts(name: str, img: np.ndarray, mask, *,
           fast_threshold: int = 20, n_features: int = 1000):
    orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=fast_threshold)
    kpts = orb.detect(img, mask)
    print(f"    {name:<60} : {len(kpts):>5}")
    return kpts


def main() -> int:
    args = parse_args()
    img_path = Path(args.img_dir) / f"{args.seq}_{args.frame:06d}.jpg"
    det_path = Path(args.det_dir) / f"{args.seq}.txt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    print("=" * 64)
    print(f"GMC diagnostic: {args.seq} frame {args.frame}")
    print("=" * 64)
    print(f"  image      : {img_path}")
    print(f"  detections : {det_path}")

    if not img_path.exists():
        print(f"[FAIL] image not found.")
        return 1
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"[FAIL] cv2.imread returned None.")
        return 1

    print(f"\n  raw load   : shape={bgr.shape}, dtype={bgr.dtype}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    gray_ds = cv2.resize(gray, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    print(f"  downscaled : shape={gray_ds.shape}")
    print(f"  intensity  : min={int(gray_ds.min())} max={int(gray_ds.max())} "
          f"mean={gray_ds.mean():.1f} std={gray_ds.std():.1f}")

    boxes = load_boxes_at_frame(det_path, args.frame)
    print(f"  det boxes @ frame {args.frame} : {len(boxes)}")

    mask = build_mask(gray_ds.shape, boxes)
    mask_frac = (mask == 0).sum() / mask.size
    print(f"  mask covers: {mask_frac*100:.1f}% of downscaled pixels")

    print("\n  Variant (downscaled gray)" + " " * 38 + "n_keypoints")
    print("  " + "-" * 62 + "  -----")
    n_kpts("ORB, no mask, default fastThreshold=20",                  gray_ds, None)
    n_kpts("ORB, det mask, default fastThreshold=20",                 gray_ds, mask)
    n_kpts("ORB, no mask, fastThreshold=5",                           gray_ds, None, fast_threshold=5)
    n_kpts("ORB, no mask, fastThreshold=2",                           gray_ds, None, fast_threshold=2)

    # CLAHE: boost local contrast (standard remedy for low-dynamic-range thermal IR).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_ds)
    print(f"\n  After CLAHE : min={int(gray_clahe.min())} max={int(gray_clahe.max())} "
          f"mean={gray_clahe.mean():.1f} std={gray_clahe.std():.1f}")
    print("\n  Variant (CLAHE-enhanced gray)" + " " * 34 + "n_keypoints")
    print("  " + "-" * 62 + "  -----")
    n_kpts("ORB, no mask, default fastThreshold=20",                  gray_clahe, None)
    n_kpts("ORB, det mask, default fastThreshold=20",                 gray_clahe, mask)
    n_kpts("ORB, det mask, fastThreshold=5",                          gray_clahe, mask, fast_threshold=5)

    # Save overlays for visual inspection.
    raw_kpts = cv2.ORB_create(nfeatures=1000, fastThreshold=20).detect(gray_ds, None)
    clahe_kpts = cv2.ORB_create(nfeatures=1000, fastThreshold=20).detect(gray_clahe, None)
    raw_bgr = cv2.cvtColor(gray_ds, cv2.COLOR_GRAY2BGR)
    clahe_bgr = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    overlay_raw = cv2.drawKeypoints(raw_bgr, raw_kpts, None, color=(0, 255, 0))
    overlay_clahe = cv2.drawKeypoints(clahe_bgr, clahe_kpts, None, color=(0, 255, 0))
    cv2.imwrite(str(out_dir / "keypoints_raw.png"), overlay_raw)
    cv2.imwrite(str(out_dir / "keypoints_clahe.png"), overlay_clahe)
    print(f"\n  Saved overlays:")
    print(f"    {out_dir / 'keypoints_raw.png'}")
    print(f"    {out_dir / 'keypoints_clahe.png'}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
