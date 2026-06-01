"""
scripts/run_bytetrack_cmc.py

Runs ByteTrack with Camera Motion Compensation (CMC) on the per-sequence
YOLOv8s detection files. For each val sequence, instantiates a fresh
ByteTrackCMCWrapper (tracker + GMC estimator), feeds detections + frames
frame by frame, and writes per-sequence MOT-format tracker outputs to a
SEPARATE output namespace so the baseline ByteTrack run is preserved
untouched.

Output namespace (default):
    data/processed/tracker_outputs/bytetrack_cmc/yolov8s_baseline/<seq>.txt

After this runs, scripts/build_trackeval_trackers.py will auto-discover
the new run under tracker_outputs/, and TrackEval can then be re-run
to compare bytetrack vs bytetrack_cmc side by side.

Per-sequence diagnostics: GMC engagement rate, fallback breakdown by
failure stage, mean inliers on engaged frames, throughput (fps).
End-of-run summary aggregates engagement and timing across sequences --
a sequence with very low engagement is a signal to revisit GMC params
for that scene type.

Usage (from project root):
    # full val set
    python scripts/run_bytetrack_cmc.py

    # single sequence sanity-check before committing to the full run
    python scripts/run_bytetrack_cmc.py --seqs MultiUAV-002

    # ablation: turn CLAHE off
    python scripts/run_bytetrack_cmc.py --gmc_use_clahe 0

Output line format (MOT16 convention):
    frame, track_id, x, y, w, h, conf, -1, -1, -1
"""

import torch  # CRITICAL on Windows: must precede the numpy chain

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import yaml

from src.tracking.bytetrack.wrapper_cmc import ByteTrackCMCWrapper


def parse_args():
    p = argparse.ArgumentParser(
        description="Run ByteTrack + CMC over the val set."
    )
    # I/O
    p.add_argument("--detections_dir", type=str,   default="data/processed/detections/yolov8s_baseline")
    p.add_argument("--images_dir",     type=str,   default="D:/uav-tracker-data/anti_uav_v4/images/val")
    p.add_argument("--split_manifest", type=str,   default="configs/splits/anti_uav_v4_track3.yaml")
    p.add_argument("--output_dir",     type=str,   default="data/processed/tracker_outputs/bytetrack_cmc/yolov8s_baseline")
    p.add_argument("--seqs",           nargs="*",  default=None,
                   help="optional subset, e.g. --seqs MultiUAV-002 MultiUAV-204")
    # ByteTrack hyperparameters (baseline defaults)
    p.add_argument("--track_thresh",   type=float, default=0.3,  help="High/low detection split threshold")
    p.add_argument("--track_buffer",   type=int,   default=30,   help="Frames to remember a lost track")
    p.add_argument("--match_thresh",   type=float, default=0.8,  help="First-stage IoU matching threshold")
    p.add_argument("--mot20",          action="store_true",      help="Crowded-scene mode")
    p.add_argument("--frame_rate",     type=int,   default=30,   help="Scales track_buffer to time")
    # GMC hyperparameters (validated thermal-IR-tuned defaults)
    p.add_argument("--gmc_use_clahe",        type=int,   default=1,    help="1=CLAHE on, 0=off")
    p.add_argument("--gmc_fast_threshold",   type=int,   default=7,    help="ORB FAST corner threshold")
    p.add_argument("--gmc_mask_conf_thresh", type=float, default=0.3,  help="Min det conf to mask a box")
    p.add_argument("--gmc_downscale",        type=int,   default=2)
    p.add_argument("--gmc_n_features",       type=int,   default=1000)
    p.add_argument("--gmc_min_inliers",      type=int,   default=6)
    p.add_argument("--gmc_det_mask_margin",  type=float, default=0.0,  help="Fractional pad around det boxes")
    return p.parse_args()


def load_val_sequences(split_manifest: Path) -> list:
    with open(split_manifest, "r") as f:
        return yaml.safe_load(f)["val"]


def load_detections_by_frame(path: Path) -> dict:
    """MOT detection file -> {frame_id: (N, 5) [x1, y1, x2, y2, conf]}."""
    by_frame = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",")
        f = int(parts[0])
        x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
        conf = float(parts[6])
        by_frame.setdefault(f, []).append([x, y, x + w, y + h, conf])
    return {f: np.array(rows, dtype=np.float32) for f, rows in by_frame.items()}


def frame_index(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def classify_fallback(stats: dict, is_first: bool) -> str:
    if is_first:
        return "fresh"
    if stats.get("n_keypoints", 0) == 0:
        return "no_kpts"
    if stats.get("n_matches", 0) < 10:  # mirrors GMC.min_matches default
        return "no_matches"
    return "no_inliers"


def track_to_mot_line(frame_id, t) -> str:
    return (f"{frame_id},{t.track_id},"
            f"{t.x:.2f},{t.y:.2f},{t.w:.2f},{t.h:.2f},"
            f"{t.score:.4f},-1,-1,-1")


def run_sequence(seq_name: str, det_path: Path, img_dir: Path,
                 out_path: Path, args) -> tuple:
    """Track one sequence. Returns (n_lines, n_ids, summary, elapsed)."""
    detections = load_detections_by_frame(det_path)
    frames = sorted(img_dir.glob(f"{seq_name}_*.jpg"), key=frame_index)
    if not frames:
        return 0, 0, None, 0.0

    wrapper = ByteTrackCMCWrapper(
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        mot20=args.mot20,
        frame_rate=args.frame_rate,
        gmc_downscale=args.gmc_downscale,
        gmc_n_features=args.gmc_n_features,
        gmc_min_inliers=args.gmc_min_inliers,
        gmc_det_mask_margin=args.gmc_det_mask_margin,
        gmc_use_clahe=bool(args.gmc_use_clahe),
        gmc_fast_threshold=args.gmc_fast_threshold,
        gmc_mask_conf_threshold=args.gmc_mask_conf_thresh,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    seen_ids = set()
    fallback_reasons = defaultdict(int)
    engaged_inliers = []
    n_engaged = 0

    t0 = time.time()
    with open(out_path, "w") as out:
        for i, img_path in enumerate(frames):
            fid = frame_index(img_path)
            dets = detections.get(fid, np.empty((0, 5), dtype=np.float32))
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            tracks = wrapper.update(dets, img)
            for t in tracks:
                out.write(track_to_mot_line(fid, t) + "\n")
                n_lines += 1
                seen_ids.add(t.track_id)
            s = wrapper.last_gmc_stats
            if s.get("fallback"):
                fallback_reasons[classify_fallback(s, is_first=(i == 0))] += 1
            else:
                n_engaged += 1
                engaged_inliers.append(s.get("n_inliers", 0))
    elapsed = time.time() - t0

    n_after_first = len(frames) - 1
    summary = {
        "n_frames": len(frames),
        "engage_rate": n_engaged / max(1, n_after_first),
        "fallbacks": dict(fallback_reasons),
        "inliers_mean": float(np.mean(engaged_inliers)) if engaged_inliers else 0.0,
        "elapsed": elapsed,
        "fps": len(frames) / elapsed if elapsed > 0 else 0.0,
    }
    return n_lines, len(seen_ids), summary, elapsed


def main():
    args = parse_args()
    det_dir        = Path(args.detections_dir)
    img_dir        = Path(args.images_dir)
    split_manifest = Path(args.split_manifest)
    out_dir        = Path(args.output_dir)

    print("=" * 64)
    print("ByteTrack + CMC -- Per-Sequence Tracking (Val Set)")
    print("=" * 64)
    print(f"\n  Detections : {det_dir}")
    print(f"  Images     : {img_dir}")
    print(f"  Manifest   : {split_manifest}")
    print(f"  Output     : {out_dir}")
    print(f"  ByteTrack  : track_thresh={args.track_thresh}, "
          f"buffer={args.track_buffer}, match={args.match_thresh}, mot20={args.mot20}")
    print(f"  GMC        : clahe={bool(args.gmc_use_clahe)}, "
          f"fast_thresh={args.gmc_fast_threshold}, "
          f"mask_conf={args.gmc_mask_conf_thresh}, "
          f"downscale={args.gmc_downscale}, min_inliers={args.gmc_min_inliers}")

    assert det_dir.exists(),        f"Missing detections: {det_dir}"
    assert img_dir.exists(),        f"Missing images dir: {img_dir}"
    assert split_manifest.exists(), f"Missing manifest:   {split_manifest}"

    val_sequences = load_val_sequences(split_manifest)
    if args.seqs:
        wanted = set(args.seqs)
        val_sequences = [s for s in val_sequences if s in wanted]
        if not val_sequences:
            print(f"\n  [WARN] none of --seqs matched the manifest.")
            return
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Sequences  : {len(val_sequences)}\n")

    total_lines  = 0
    total_frames = 0
    total_ids    = 0
    total_elapsed = 0.0
    eng_rates    = []
    inlier_means = []
    fallback_global = defaultdict(int)

    for i, seq_name in enumerate(val_sequences, start=1):
        det_path = det_dir / f"{seq_name}.txt"
        if not det_path.exists():
            print(f"  [WARN] no detection file for {seq_name}, skipping.")
            continue
        out_path = out_dir / f"{seq_name}.txt"

        n_lines, n_ids, summary, elapsed = run_sequence(
            seq_name, det_path, img_dir, out_path, args
        )
        if summary is None:
            print(f"  [WARN] no images for {seq_name}, skipping.")
            continue

        total_lines  += n_lines
        total_frames += summary["n_frames"]
        total_ids    += n_ids
        total_elapsed += elapsed
        eng_rates.append(summary["engage_rate"])
        if summary["inliers_mean"] > 0:
            inlier_means.append(summary["inliers_mean"])
        for k, v in summary["fallbacks"].items():
            fallback_global[k] += v

        print(f"  [{i:>2}/{len(val_sequences)}] {seq_name}  "
              f"frames={summary['n_frames']:>5}  "
              f"lines={n_lines:>6,}  "
              f"ids={n_ids:>4}  "
              f"GMC={summary['engage_rate']*100:>3.0f}%  "
              f"inliers={summary['inliers_mean']:>5.0f}  "
              f"({summary['fps']:>4.1f} fps)")

    print("\n" + "=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  Sequences processed : {len(eng_rates)}")
    print(f"  Frames              : {total_frames:,}")
    print(f"  Track lines         : {total_lines:,}")
    print(f"  Unique IDs (total)  : {total_ids:,}")
    print(f"  Wall time           : {total_elapsed:.1f}s")
    if total_elapsed > 0:
        print(f"  Throughput          : {total_frames/total_elapsed:.1f} fps")
    if eng_rates:
        print(f"\n  GMC engagement (per-seq): "
              f"mean {np.mean(eng_rates)*100:.0f}%  "
              f"min {np.min(eng_rates)*100:.0f}%  "
              f"max {np.max(eng_rates)*100:.0f}%")
        print(f"  Fallback breakdown (totals): "
              f"fresh={fallback_global['fresh']}  "
              f"no_kpts={fallback_global['no_kpts']}  "
              f"no_matches={fallback_global['no_matches']}  "
              f"no_inliers={fallback_global['no_inliers']}")
    if inlier_means:
        print(f"  Inliers / engaged frame (per-seq mean): "
              f"mean {np.mean(inlier_means):.0f}  "
              f"min {np.min(inlier_means):.0f}  "
              f"max {np.max(inlier_means):.0f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
