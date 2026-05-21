"""
scripts/run_detection_inference.py

Runs YOLOv8s baseline detector inference on the 39 val sequences from the
Anti-UAV v4 Track 3 train/val split. Produces one MOT-format detection
file per sequence, ready to feed into ByteTrack or DeepSORT.

Each output line:
    frame_id, -1, x, y, w, h, conf, -1, -1, -1
where x, y are the top-left corner and w, h are width and height (pixels).
The -1 in column 2 is the identity placeholder (filled later by the tracker).
Trailing -1s are 3D coordinates unused in 2D Multi-Object Tracking.

Usage (from project root):
    python scripts/run_detection_inference.py
or with explicit overrides:
    python scripts/run_detection_inference.py --conf 0.05 --batch_size 32
"""

import torch  # CRITICAL on Windows: must be imported before ultralytics so
              # torch's DLLs register before cv2/others perturb the DLL search
              # path. Same constraint as for albumentations. Do not reorder.

import argparse
from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8s detector inference on Anti-UAV v4 val sequences."
    )
    parser.add_argument("--model_path",     type=str,   default="models/detection/yolov8s_baseline/best.pt")
    parser.add_argument("--images_dir",     type=str,   default="D:/uav-tracker-data/anti_uav_v4/images/val")
    parser.add_argument("--split_manifest", type=str,   default="configs/splits/anti_uav_v4_track3.yaml")
    parser.add_argument("--output_dir",     type=str,   default="data/processed/detections/yolov8s_baseline")
    parser.add_argument("--conf",           type=float, default=0.10, help="Confidence threshold (permissive for tracking)")
    parser.add_argument("--iou",            type=float, default=0.45, help="Non-Maximum Suppression IoU threshold")
    parser.add_argument("--imgsz",          type=int,   default=640,  help="Inference image size")
    parser.add_argument("--batch_size",     type=int,   default=16,   help="Inference batch size")
    return parser.parse_args()


def load_val_sequences(split_manifest: Path) -> list:
    """Reads the val sequence list from the split manifest YAML."""
    with open(split_manifest, "r") as f:
        manifest = yaml.safe_load(f)
    return manifest["val"]


def collect_sequence_images(images_dir: Path, seq_name: str) -> list:
    """Returns all .jpg files for `seq_name`, sorted by frame number.
       Lexicographic sort works because frame numbers are zero-padded."""
    return sorted(images_dir.glob(f"{seq_name}_*.jpg"))


def frame_id_from_path(image_path: Path) -> int:
    """Extracts the 1-based frame number from a path like .../MultiUAV-002_000001.jpg"""
    # stem = "MultiUAV-002_000001"; last underscore-delimited token = frame number
    return int(image_path.stem.split("_")[-1])


def detections_to_mot_lines(frame_id, boxes_xyxy, confs):
    """Converts (xyxy, conf) detections to MOT-format detection lines for one frame."""
    lines = []
    for (x1, y1, x2, y2), conf in zip(boxes_xyxy, confs):
        x = float(x1)
        y = float(y1)
        w = float(x2 - x1)
        h = float(y2 - y1)
        # Columns: frame, id(=-1), x, y, w, h, conf, x3d(-1), y3d(-1), z3d(-1)
        lines.append(f"{frame_id},-1,{x:.2f},{y:.2f},{w:.2f},{h:.2f},{float(conf):.4f},-1,-1,-1")
    return lines


def run_inference_on_sequence(model, images, output_path, conf, iou, imgsz, batch_size):
    """Runs inference on all frames of one sequence and writes its MOT detection file."""
    all_lines = []

    for batch_start in tqdm(range(0, len(images), batch_size),
                            desc="    batches",
                            leave=False,
                            unit="batch"):
        batch = images[batch_start:batch_start + batch_size]
        batch_paths = [str(p) for p in batch]

        # Ultralytics handles batching internally when given a list of paths
        results = model(batch_paths, conf=conf, iou=iou, imgsz=imgsz, verbose=False)

        for img_path, result in zip(batch, results):
            boxes = result.boxes
            if len(boxes) == 0:
                continue   # empty frames are implicit — no lines written

            frame_id = frame_id_from_path(img_path)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            all_lines.extend(detections_to_mot_lines(frame_id, xyxy, confs))

    output_path.write_text("\n".join(all_lines) + ("\n" if all_lines else ""))
    return len(all_lines)


def main():
    args = parse_args()

    model_path     = Path(args.model_path)
    images_dir     = Path(args.images_dir)
    split_manifest = Path(args.split_manifest)
    output_dir     = Path(args.output_dir)

    print("=" * 60)
    print("YOLOv8s Detector — Per-Sequence Inference (Val Set)")
    print("=" * 60)
    print(f"\n  Model     : {model_path}")
    print(f"  Images    : {images_dir}")
    print(f"  Manifest  : {split_manifest}")
    print(f"  Output    : {output_dir}")
    print(f"  Conf      : {args.conf}")
    print(f"  IoU       : {args.iou}")
    print(f"  ImgSz     : {args.imgsz}")
    print(f"  Batch     : {args.batch_size}")

    # --- Sanity checks ----------------------------------------------
    assert model_path.exists(),     f"Missing model: {model_path}"
    assert images_dir.exists(),     f"Missing images directory: {images_dir}"
    assert split_manifest.exists(), f"Missing split manifest: {split_manifest}"

    # --- Load val list and prepare output dir -----------------------
    val_sequences = load_val_sequences(split_manifest)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Val sequences : {len(val_sequences)}")

    # --- Load model -------------------------------------------------
    print("\n  Loading model...")
    model = YOLO(str(model_path))
    print(f"  Device        : {model.device}\n")

    # --- Per-sequence inference -------------------------------------
    total_detections = 0
    total_frames     = 0

    for i, seq_name in enumerate(val_sequences, start=1):
        images = collect_sequence_images(images_dir, seq_name)

        if not images:
            print(f"  [WARN] No images found for {seq_name}, skipping.")
            continue

        output_path = output_dir / f"{seq_name}.txt"
        print(f"  [{i:>2}/{len(val_sequences)}] {seq_name}  ({len(images):>5} frames)")

        n_dets = run_inference_on_sequence(
            model=model,
            images=images,
            output_path=output_path,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
        )

        total_detections += n_dets
        total_frames     += len(images)
        print(f"        -> {n_dets:>7,} detections written")

    # --- Summary ----------------------------------------------------
    print("\n" + "=" * 60)
    print("  Done.")
    print(f"  Sequences processed : {len(val_sequences)}")
    print(f"  Frames processed    : {total_frames:,}")
    print(f"  Total detections    : {total_detections:,}")
    if total_frames > 0:
        print(f"  Avg per frame       : {total_detections / total_frames:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
