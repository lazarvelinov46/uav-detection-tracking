"""Phase 4 Task B - real-time FPS benchmark.

Measures throughput of the detector + tracker pipeline frame-by-frame:
  - detector inference time (uses CUDA if available, falls back to CPU)
  - tracker association time (CPU-only by design)
  - end-to-end frames-per-second

The pipeline runs once and times each component separately, so a single run
yields all three numbers. For detector-only or tracker-only analysis, just
read the relevant column from the per-frame CSV.

Hardware-agnostic. Picks up GPU automatically. Run locally for tracker
overhead numbers; run on Kaggle T4 (or any GPU box) for the headline
detector / end-to-end numbers using the same script.

Usage (from project root):
    python scripts/benchmark_fps.py --detector yolov8s --tracker bytetrack
    python scripts/benchmark_fps.py --detector yolov8s --tracker deepsort
    python scripts/benchmark_fps.py --detector yolov8s --tracker none

Outputs:
    outputs/fps/<detector>_<tracker>_<device>.csv   - per-frame timings
    outputs/fps/summary.csv                          - one row per config, appended
    Printed summary with FPS and the 25 FPS real-time verdict.
"""
import torch  # MUST be the first non-stdlib import (Windows DLL ordering for
              # ultralytics / byte_tracker / albumentations).

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make `src.tracking.bytetrack...` importable when run from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REAL_TIME_FPS = 25  # paper's real-time bar


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------

def find_sequence(images_root: Path, sequence: str | None) -> str:
    """Return an existing MultiUAV-* sequence prefix from images_root."""
    if sequence:
        return sequence
    seqs = set()
    for p in images_root.glob("MultiUAV-*.jpg"):
        m = re.match(r"(MultiUAV-\d+)_", p.name)
        if m:
            seqs.add(m.group(1))
    if not seqs:
        raise FileNotFoundError(f"No MultiUAV-* frames found under {images_root}")
    return sorted(seqs)[0]


def load_frames(images_root: Path, sequence: str, n: int) -> list[np.ndarray]:
    """Pre-load the first n frames of a sequence (BGR uint8) into RAM."""
    paths = sorted(images_root.glob(f"{sequence}_*.jpg"))[:n]
    if not paths:
        raise FileNotFoundError(
            f"No frames matching {sequence}_*.jpg under {images_root}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            raise IOError(f"failed to read {p}")
        frames.append(img)
    return frames


# ---------------------------------------------------------------------------
# Detectors - each builder returns a callable infer(img_bgr) -> Nx5 array of
# [x1, y1, x2, y2, score] for the UAV class.
# ---------------------------------------------------------------------------

def make_detector_yolov8s(weights: Path, device: str,
                          imgsz: int = 640, conf: float = 0.05):
    from ultralytics import YOLO
    model = YOLO(str(weights))

    def infer(img):
        r = model.predict(img, imgsz=imgsz, conf=conf, verbose=False,
                          device=device)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)
        xyxy = r.boxes.xyxy.cpu().numpy()
        cf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
        return np.concatenate([xyxy, cf], axis=1).astype(np.float32)

    return infer


def make_detector_yolox(weights: Path, device: str,
                        imgsz: int = 640, conf: float = 0.05,
                        exp_file: Path = Path("configs/yolox_uav_s.py"),
                        nms_thresh: float = 0.45):
    """Build a YOLOX-S inference callable.

    Mirrors the project's Kaggle inference notebook:
      - ValTransform(legacy=False) preprocess (no ImageNet normalization)
      - test_size = (512, 640) from the EXP file (H, W), native thermal aspect
      - class-agnostic NMS for single-class UAV detection
      - score = obj_conf * class_conf

    `imgsz` is accepted for API symmetry with the YOLOv8s builder but ignored
    here: YOLOX input shape is fixed by the EXP file. Returns the same
    callable signature: infer(img_bgr) -> Nx5 [x1,y1,x2,y2,score] array.
    """
    # Local imports keep ultralytics-only environments from pulling in YOLOX.
    # cv2 was already imported at module top, which prevents the Windows
    # torchvision->cv2 DLL ordering issue from biting here.
    from yolox.exp import get_exp
    from yolox.utils import postprocess
    from yolox.data.data_augment import ValTransform

    exp = get_exp(str(exp_file), None)
    model = exp.get_model()
    # weights_only=False is required on torch 2.10+; the checkpoint is our own.
    ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    test_size = (512, 640)  # (H, W), matches EXP file
    num_classes = 1
    preproc = ValTransform(legacy=False)

    @torch.no_grad()
    def infer(img):
        t, _ = preproc(img, None, test_size)              # (3, H, W) float32
        ratio = min(test_size[0] / img.shape[0],
                    test_size[1] / img.shape[1])          # 1.0 at native 640x512
        batch = torch.from_numpy(t).unsqueeze(0).to(device)

        outs = model(batch)
        outs = postprocess(outs, num_classes, conf, nms_thresh,
                           class_agnostic=True)
        out = outs[0]
        if out is None:
            return np.zeros((0, 5), dtype=np.float32)
        out = out.cpu().numpy()
        boxes = out[:, 0:4] / ratio
        scores = (out[:, 4] * out[:, 5]).reshape(-1, 1)
        return np.concatenate([boxes, scores], axis=1).astype(np.float32)

    return infer


# ---------------------------------------------------------------------------
# Trackers - each builder returns a callable update(dets_xyxysc, img_bgr).
# img is unused by ByteTrack; DeepSORT needs it for embedding extraction.
# ---------------------------------------------------------------------------

def make_tracker_bytetrack(track_thresh: float = 0.3):
    from src.tracking.bytetrack.byte_tracker import BYTETracker

    class A:  # ByteTrack expects an args-namespace object
        pass
    a = A()
    a.track_thresh = track_thresh
    a.match_thresh = 0.8
    a.track_buffer = 30
    a.mot20 = False
    trk = BYTETracker(a)

    def update(dets, img):
        h, w = img.shape[:2]
        return trk.update(dets, (h, w), (h, w))  # no scaling: dets already in image coords

    return update


def make_tracker_bytetrack_cmc(track_thresh: float = 0.3):
    """ByteTrack + Camera Motion Compensation (GMC).

    Uses ByteTrackCMCWrapper at the headline configuration from the analysis:
    track_thresh=0.3, CLAHE on, FAST threshold=7, mask conf threshold=0.3 -
    matches the CLI defaults of scripts/run_bytetrack_cmc.py that produced
    HOTA 65.82 / AssA 63.02. Per-frame tracker time captures BOTH the GMC
    estimate (CLAHE + downscale + ORB/FAST + matching + RANSAC) and the
    ByteTrack association step, so the measurement is the true cost of CMC.
    """
    from src.tracking.bytetrack.wrapper_cmc import ByteTrackCMCWrapper

    wrapper = ByteTrackCMCWrapper(
        track_thresh=track_thresh,
        # Thermal-IR-tuned GMC defaults that produced the headline result.
        gmc_use_clahe=True,
        gmc_fast_threshold=7,
        gmc_mask_conf_threshold=0.3,
    )

    def update(dets, img):
        return wrapper.update(dets, img)

    return update


def make_tracker_deepsort():
    from deep_sort_realtime.deepsort_tracker import DeepSort
    trk = DeepSort(max_age=30, n_init=3, embedder="mobilenet",
                   embedder_gpu=False, half=False)

    def update(dets, img):
        if len(dets) == 0:
            return trk.update_tracks([], frame=img)
        rows = [([float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                 float(c), 0)
                for x1, y1, x2, y2, c in dets]
        return trk.update_tracks(rows, frame=img)

    return update


# ---------------------------------------------------------------------------
# Benchmark loop
# ---------------------------------------------------------------------------

def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def benchmark(infer_fn, update_fn, frames, device, warmup):
    """Time detector and tracker per measured frame.

    The first `warmup` frames are run through the full pipeline but not timed
    (CUDA kernel compile + ultralytics first-call setup + tracker first-update
    pollute the early samples). Returns (det_ms, trk_ms) numpy arrays of
    length len(frames) - warmup.
    """
    # Warmup
    for img in frames[:warmup]:
        dets = infer_fn(img)
        _sync(device)
        if update_fn is not None:
            update_fn(dets, img)

    det_ms, trk_ms = [], []
    for img in frames[warmup:]:
        # Detector
        _sync(device)
        t0 = time.perf_counter()
        dets = infer_fn(img)
        _sync(device)
        det_ms.append((time.perf_counter() - t0) * 1000.0)

        # Tracker
        if update_fn is not None:
            t0 = time.perf_counter()
            update_fn(dets, img)
            trk_ms.append((time.perf_counter() - t0) * 1000.0)
        else:
            trk_ms.append(0.0)

    return np.array(det_ms), np.array(trk_ms)


def summarize(arr_ms, name: str) -> dict:
    a = np.asarray(arr_ms, dtype=float)
    if a.size == 0 or a.mean() == 0:
        return dict(name=name, n=int(a.size), mean_ms=0.0, median_ms=0.0,
                    p99_ms=0.0, fps=float("inf"))
    return dict(
        name=name, n=int(a.size),
        mean_ms=float(a.mean()),
        median_ms=float(np.median(a)),
        p99_ms=float(np.percentile(a, 99)),
        fps=1000.0 / float(a.mean()),
    )


def print_summary(det_s, trk_s, e2e_s, detector, tracker, device):
    print(f"\nResults (n={det_s['n']} measured frames, device={device}):")
    print(f"  Detector ({detector:<8}): "
          f"mean {det_s['mean_ms']:7.2f} ms | median {det_s['median_ms']:7.2f} ms "
          f"| p99 {det_s['p99_ms']:7.2f} ms | {det_s['fps']:6.1f} FPS")
    print(f"  Tracker  ({tracker:<8}): "
          f"mean {trk_s['mean_ms']:7.2f} ms | median {trk_s['median_ms']:7.2f} ms "
          f"| p99 {trk_s['p99_ms']:7.2f} ms")
    print(f"  End-to-end          : "
          f"mean {e2e_s['mean_ms']:7.2f} ms | {e2e_s['fps']:6.1f} FPS")
    verdict = (f"\u2713 real-time (>= {REAL_TIME_FPS} FPS)"
               if e2e_s["fps"] >= REAL_TIME_FPS
               else f"\u2717 below real-time bar (< {REAL_TIME_FPS} FPS)")
    print(f"  Real-time: {verdict}")


def write_outputs(det_ms, trk_ms, det_s, trk_s, e2e_s,
                  detector, tracker, device, out: Path):
    out.mkdir(parents=True, exist_ok=True)

    per_frame = out / f"{detector}_{tracker}_{device}.csv"
    with open(per_frame, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "detector_ms", "tracker_ms", "total_ms"])
        for i, (d, t) in enumerate(zip(det_ms, trk_ms)):
            w.writerow([i, f"{d:.4f}", f"{t:.4f}", f"{d + t:.4f}"])
    print(f"[csv] wrote {per_frame}")

    summary = out / "summary.csv"
    new_file = not summary.exists()
    with open(summary, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow([
                "detector", "tracker", "device", "n",
                "det_mean_ms", "det_p99_ms", "det_fps",
                "trk_mean_ms", "trk_p99_ms",
                "e2e_mean_ms", "e2e_fps",
            ])
        w.writerow([
            detector, tracker, device, det_s["n"],
            f"{det_s['mean_ms']:.3f}", f"{det_s['p99_ms']:.3f}", f"{det_s['fps']:.2f}",
            f"{trk_s['mean_ms']:.3f}", f"{trk_s['p99_ms']:.3f}",
            f"{e2e_s['mean_ms']:.3f}", f"{e2e_s['fps']:.2f}",
        ])
    print(f"[csv] appended row to {summary}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "yolov8s": Path("models/detection/yolov8s_baseline/best.pt"),
    "yolox":   Path("models/detection/yolox_s_baseline/best_ckpt.pth"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detector", choices=["yolov8s", "yolox"], default="yolov8s")
    parser.add_argument("--tracker",
                        choices=["bytetrack", "bytetrack_cmc", "deepsort", "none"],
                        default="bytetrack")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--images-root", type=Path,
                        default=Path("D:/uav-tracker-data/anti_uav_v4/images/val"))
    parser.add_argument("--sequence", default=None,
                        help="MultiUAV-XXX prefix; auto-picks first available if omitted")
    parser.add_argument("--num-frames", type=int, default=500,
                        help="frames timed (not counting warmup)")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--device", default=None,
                        help="cuda / cpu (auto if omitted)")
    parser.add_argument("--out", type=Path, default=Path("outputs/fps"))
    parser.add_argument("--exp-file", type=Path,
                        default=Path("configs/yolox_uav_s.py"),
                        help="YOLOX EXP config (used only for --detector yolox)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    weights = args.weights or DEFAULT_WEIGHTS[args.detector]
    if not weights.exists():
        sys.exit(f"weights not found: {weights}")

    seq = find_sequence(args.images_root, args.sequence)
    n_total = args.num_frames + args.warmup
    print(f"Device     : {device}")
    print(f"Detector   : {args.detector}  ({weights})")
    print(f"Tracker    : {args.tracker}")
    print(f"Sequence   : {seq}  ({args.num_frames} measured + {args.warmup} warmup frames)")

    frames = load_frames(args.images_root, seq, n_total)
    if len(frames) < n_total:
        print(f"[warn] only {len(frames)} frames available; reducing measurement window")

    print(f"Building detector ...")
    if args.detector == "yolov8s":
        infer_fn = make_detector_yolov8s(weights, device)
    else:
        infer_fn = make_detector_yolox(weights, device, exp_file=args.exp_file)

    print(f"Building tracker  ...")
    if args.tracker == "bytetrack":
        update_fn = make_tracker_bytetrack()
    elif args.tracker == "bytetrack_cmc":
        update_fn = make_tracker_bytetrack_cmc()
    elif args.tracker == "deepsort":
        update_fn = make_tracker_deepsort()
    else:
        update_fn = None

    print(f"Running benchmark ...")
    det_ms, trk_ms = benchmark(infer_fn, update_fn, frames, device, args.warmup)
    total_ms = det_ms + trk_ms

    det_s = summarize(det_ms, "detector")
    trk_s = summarize(trk_ms, "tracker")
    e2e_s = summarize(total_ms, "end-to-end")

    write_outputs(det_ms, trk_ms, det_s, trk_s, e2e_s,
                  args.detector, args.tracker, device, args.out)
    print_summary(det_s, trk_s, e2e_s, args.detector, args.tracker, device)


if __name__ == "__main__":
    main()
