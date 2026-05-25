# scripts/run_deepsort.py
import torch  # MUST be first: deep_sort_realtime pulls in torch; avoids WinError 127 on Windows

import argparse
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


def load_detections(det_file):
    """MOT detection file (frame,-1,x,y,w,h,conf,...) -> {frame: [([l,t,w,h], conf), ...]}."""
    by_frame = defaultdict(list)
    with open(det_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",")
            frame = int(float(p[0]))
            x, y, w, h, conf = (float(p[2]), float(p[3]), float(p[4]),
                                float(p[5]), float(p[6]))
            by_frame[frame].append(([x, y, w, h], conf))
    return by_frame


def frame_index(path):
    """MultiUAV-002_000123.jpg -> 123 (1-based)."""
    return int(path.stem.split("_")[-1])


def run_sequence(seq, det_file, img_dir, out_file, args):
    detections = load_detections(det_file)

    # The image directory is the authoritative frame list, so frames with no
    # detections are still stepped through and tracks age correctly.
    frames = {frame_index(p): p for p in img_dir.glob(f"{seq}_*.jpg")}
    if not frames:
        print(f"  [WARN] no images for {seq} in {img_dir}")
        return 0, 0

    # Black frame at the right resolution for empty-detection frames: the
    # embedder is never invoked when there are no boxes, so content is irrelevant.
    h, w = cv2.imread(str(frames[min(frames)])).shape[:2]
    blank = np.zeros((h, w, 3), dtype=np.uint8)

    tracker = DeepSort(
        max_age=args.max_age,                       # mirrors ByteTrack track_buffer=30
        n_init=args.n_init,                         # confirm a track after N consecutive hits
        max_cosine_distance=args.max_cosine_distance,
        nn_budget=None,
        embedder=args.embedder,                     # default bundled MobileNet Re-ID
        bgr=True,                                   # OpenCV frames are BGR
        embedder_gpu=False,                         # CPU-only tracking
        half=False,                                 # fp16 is a GPU path; off on CPU
    )

    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_lines, ids = 0, set()
    with open(out_file, "w") as out:
        for frame_no in sorted(frames):
            dets = detections.get(frame_no, [])
            if dets:
                img = cv2.imread(str(frames[frame_no]))
                raw = [([l, t, bw, bh], conf, 0) for (l, t, bw, bh), conf in dets]
            else:
                img, raw = blank, []

            tracks = tracker.update_tracks(raw, frame=img)

            for tr in tracks:
                if not tr.is_confirmed() or tr.time_since_update > 0:
                    continue  # only tracks updated by a real detection this frame
                l, t, bw, bh = tr.to_ltwh()
                conf = tr.det_conf if tr.det_conf is not None else -1
                tid = int(tr.track_id)
                ids.add(tid)
                out.write(f"{frame_no},{tid},{l:.2f},{t:.2f},{bw:.2f},{bh:.2f},"
                          f"{conf:.2f},-1,-1,-1\n")
                n_lines += 1
    return n_lines, len(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default="data/processed/detections/yolov8s_baseline")
    ap.add_argument("--img-dir", default="D:/uav-tracker-data/anti_uav_v4/images/val")
    ap.add_argument("--out-dir",
                    default="data/processed/tracker_outputs/deepsort/yolov8s_baseline")
    ap.add_argument("--max-age", type=int, default=30)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--max-cosine-distance", type=float, default=0.2)
    ap.add_argument("--embedder", default="mobilenet")
    ap.add_argument("--seqs", nargs="*", default=None,
                    help="optional subset, e.g. --seqs MultiUAV-002 MultiUAV-278")
    args = ap.parse_args()

    det_dir, img_dir, out_dir = Path(args.det_dir), Path(args.img_dir), Path(args.out_dir)
    det_files = sorted(det_dir.glob("*.txt"))
    if args.seqs:
        wanted = set(args.seqs)
        det_files = [f for f in det_files if f.stem in wanted]

    print(f"DeepSORT over {len(det_files)} sequences "
          f"(max_age={args.max_age}, n_init={args.n_init}, "
          f"max_cos={args.max_cosine_distance}, embedder={args.embedder})")
    grand, t0 = 0, time.time()
    for det_file in det_files:
        seq, s = det_file.stem, time.time()
        n_lines, n_ids = run_sequence(seq, det_file, img_dir,
                                      out_dir / f"{seq}.txt", args)
        grand += n_lines
        print(f"  {seq:<16} tracks={n_lines:>7}  ids={n_ids:>4}  ({time.time()-s:5.1f}s)")
    print(f"Done: {grand} track lines in {time.time()-t0:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
