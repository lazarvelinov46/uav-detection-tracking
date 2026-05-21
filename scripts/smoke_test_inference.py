"""
scripts/smoke_test_inference.py

Smoke test for YOLOv8s detector inference.
Loads the Phase 3 best.pt checkpoint and runs it on a single validation
frame to verify: (1) the checkpoint loads, (2) inference produces sane
output, (3) the detection count is in the expected ballpark vs ground truth.

Run from the project root:
    python scripts/smoke_test_inference.py
"""
import torch
from pathlib import Path
from ultralytics import YOLO


# --- Configuration ---------------------------------------------------
# Paths are hardcoded for the smoke test — these become argparse args
# in the full per-sequence inference script later.

MODEL_PATH = Path("models/detection/yolov8s_baseline/best.pt")
IMAGE_PATH = Path("D:/uav-tracker-data/anti_uav_v4/images/val/MultiUAV-002_000001.jpg")
LABEL_PATH = Path("D:/uav-tracker-data/anti_uav_v4/labels/val/MultiUAV-002_000001.txt")

# Inference hyperparameters
CONF_THRESHOLD = 0.10   # permissive — ByteTrack needs the low-conf tail
IOU_THRESHOLD  = 0.45   # default Non-Maximum Suppression IoU threshold
IMG_SIZE       = 640    # match training resolution


def count_gt_boxes(label_path: Path) -> int:
    """Counts non-empty lines in a YOLO-format label file. Each line = one GT UAV."""
    if not label_path.exists():
        return -1
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def main():
    print("=" * 55)
    print("YOLOv8s Detector — Smoke Test Inference")
    print("=" * 55)

    # --- Path sanity ------------------------------------------------
    print(f"\n  Model : {MODEL_PATH}")
    print(f"  Image : {IMAGE_PATH}")
    assert MODEL_PATH.exists(), f"Missing model checkpoint: {MODEL_PATH}"
    assert IMAGE_PATH.exists(), f"Missing test image: {IMAGE_PATH}"

    # --- Load model -------------------------------------------------
    print("\n  Loading model...")
    model = YOLO(str(MODEL_PATH))
    print(f"  Device         : {model.device}")

    # --- Run inference ----------------------------------------------
    # model(...) returns a LIST — one Results object per input image.
    results = model(
        str(IMAGE_PATH),
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMG_SIZE,
        verbose=False,   # suppress Ultralytics' own per-image log line
    )
    boxes = results[0].boxes

    # --- Inspect output ---------------------------------------------
    n_pred = len(boxes)
    print(f"\n  Predictions    : {n_pred}")

    if n_pred > 0:
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()

        print(f"  Conf min/max   : {confs.min():.3f} / {confs.max():.3f}")
        print(f"  Conf mean      : {confs.mean():.3f}")
        print(f"  Classes (uniq) : {sorted(set(classes.tolist()))}")

        # Show first 3 boxes
        print(f"\n  First {min(3, n_pred)} boxes (x1, y1, x2, y2, conf):")
        for i in range(min(3, n_pred)):
            x1, y1, x2, y2 = xyxy[i]
            print(f"    [{i}] ({x1:7.2f}, {y1:7.2f}, {x2:7.2f}, {y2:7.2f})  conf={confs[i]:.3f}")

    # --- Cross-check against ground truth ---------------------------
    n_gt = count_gt_boxes(LABEL_PATH)
    print(f"\n  Ground truth   : {n_gt} UAV(s) in this frame")

    if n_gt > 0 and n_pred > 0:
        ratio = n_pred / n_gt
        print(f"  Pred/GT ratio  : {ratio:.2f}")
        if 0.7 <= ratio <= 1.5:
            print("  [OK] Detection count is in expected ballpark.")
        else:
            print("  [WARN] Detection count is unusual — worth investigating.")

    print("\n  Smoke test done.\n")


if __name__ == "__main__":
    main()
