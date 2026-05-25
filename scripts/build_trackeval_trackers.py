# scripts/build_trackeval_trackers.py
"""
Place tracker outputs into TrackEval's TRACKERS_FOLDER layout:
    <trackers-folder>/<benchmark>-<split>/<tracker>/data/<seq>.txt
Auto-discovers every run under <src-root> that has a <detector>/ subdir,
using the run's directory name as the TrackEval tracker label.
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default="data/processed/tracker_outputs")
    ap.add_argument("--detector", default="yolov8s_baseline")
    ap.add_argument("--trackers-folder", default="data/eval/trackers")
    ap.add_argument("--benchmark", default="AntiUAV")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.trackers_folder) / f"{args.benchmark}-{args.split}"
    runs = [d for d in sorted(src_root.iterdir()) if (d / args.detector).is_dir()]
    if not runs:
        raise SystemExit(f"No runs with a '{args.detector}/' subdir under {src_root}")

    for run in runs:
        dst = dst_root / run.name / "data"
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for txt in (run / args.detector).glob("*.txt"):
            shutil.copy(txt, dst / txt.name)
            n += 1
        print(f"  {run.name:<18} -> {dst}  ({n} sequences)")
    print(f"Placed {len(runs)} trackers under {dst_root}")


if __name__ == "__main__":
    main()
