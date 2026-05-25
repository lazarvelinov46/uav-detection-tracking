# scripts/build_trackeval_gt.py
"""
Build the TrackEval MOTChallenge ground-truth layout for the Anti-UAV v4 val split.

Creates, under <trackeval-root>/data/gt/mot_challenge/:
    <BENCHMARK>-<SPLIT>/<seq>/gt/gt.txt     (MOT17 GT format: frame,id,x,y,w,h,1,1,1)
    <BENCHMARK>-<SPLIT>/<seq>/seqinfo.ini
    seqmaps/<BENCHMARK>-<SPLIT>.txt

Val sequence list is taken from the detection output dir (those 39 files ARE the
val split), so the GT set is guaranteed to match what was tracked.
"""
import argparse
from pathlib import Path


def read_gt(gt_file):
    """Raw MOT GT -> list of (frame, id, x, y, w, h). Tolerates comma or whitespace."""
    rows = []
    with open(gt_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = line.split(",") if "," in line else line.split()
            if len(p) < 6:
                continue
            rows.append((int(float(p[0])), int(float(p[1])),
                         float(p[2]), float(p[3]), float(p[4]), float(p[5])))
    return rows


def seq_length(seq, img_dir, rows):
    """Max frame index from images if present, else from GT."""
    imgs = list(img_dir.glob(f"{seq}_*.jpg"))
    if imgs:
        return max(int(p.stem.split("_")[-1]) for p in imgs)
    return max((r[0] for r in rows), default=0)


def write_seqinfo(path, seq, length, fps, w, h):
    path.write_text(
        "[Sequence]\n"
        f"name={seq}\nimDir=img1\nframeRate={fps}\n"
        f"seqLength={length}\nimWidth={w}\nimHeight={h}\nimExt=.jpg\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det-dir", default="data/processed/detections/yolov8s_baseline",
                    help="source of the val sequence list (its .txt stems)")
    ap.add_argument("--gt-src", default="data/raw/anti_uav_v4/TrainLabels",
                    help="dir of raw MOT GT files named <seq>.txt")
    ap.add_argument("--img-dir", default="D:/uav-tracker-data/anti_uav_v4/images/val",
                    help="val frames, used to count seqLength")
    ap.add_argument("--trackeval-root", default="TrackEval", help="cloned TrackEval repo root")
    ap.add_argument("--benchmark", default="AntiUAV")
    ap.add_argument("--split", default="val")
    ap.add_argument("--frame-rate", type=int, default=30)
    ap.add_argument("--im-width", type=int, default=640)
    ap.add_argument("--im-height", type=int, default=512)
    args = ap.parse_args()

    det_dir, gt_src, img_dir = Path(args.det_dir), Path(args.gt_src), Path(args.img_dir)
    gt_set = f"{args.benchmark}-{args.split}"
    mot = Path(args.trackeval_root) / "data" / "gt" / "mot_challenge"
    gt_root, seqmap_dir = mot / gt_set, mot / "seqmaps"

    seqs = sorted(f.stem for f in det_dir.glob("*.txt"))
    if not seqs:
        raise SystemExit(f"No sequences found in {det_dir}")

    seqmap_dir.mkdir(parents=True, exist_ok=True)
    written, missing = [], []
    for seq in seqs:
        gt_file = gt_src / f"{seq}.txt"
        if not gt_file.exists():
            missing.append(seq)
            continue
        rows = read_gt(gt_file)
        length = seq_length(seq, img_dir, rows)
        seq_dir = gt_root / seq
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        with open(seq_dir / "gt" / "gt.txt", "w") as out:
            for frame, tid, x, y, w, h in rows:
                out.write(f"{frame},{tid},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1,1,1\n")
        write_seqinfo(seq_dir / "seqinfo.ini", seq, length,
                      args.frame_rate, args.im_width, args.im_height)
        written.append((seq, len(rows), length))

    with open(seqmap_dir / f"{gt_set}.txt", "w") as f:
        f.write("name\n")
        for seq, _, _ in written:
            f.write(f"{seq}\n")

    print(f"Benchmark dir: {gt_root}")
    for seq, n_gt, length in written:
        print(f"  {seq:<16} gt_rows={n_gt:>7}  seqLength={length:>5}")
    print(f"Seqmap: {seqmap_dir / (gt_set + '.txt')}  ({len(written)} sequences)")
    if missing:
        print(f"\n[WARN] no GT file in {gt_src} for {len(missing)} sequence(s):")
        print("       " + ", ".join(missing))
        print("       Repoint --gt-src to the dir holding <seq>.txt MOT labels "
              "(e.g. 'C:/UAV Detection and Tracking/MultiUAV_Train').")


if __name__ == "__main__":
    main()
