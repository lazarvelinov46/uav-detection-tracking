"""
scripts/renumber_track_ids.py

Renumber track IDs in MOT-format tracker output files so each sequence
starts from 1 (instead of inheriting a global counter that bleeds
across sequences in a single Python process).

VISUALIZATION-ONLY utility. TrackEval evaluates per sequence and only
requires IDs to be unique WITHIN a sequence -- both the original and
the renumbered files give identical TrackEval results. The point here
is purely legibility for rendered videos: the first UAV in any clip is
labelled "1" instead of "4,517" or similar.

For each input .txt file, IDs are remapped in order of FIRST APPEARANCE:
the first track to show up (lowest frame, then earliest in the file)
becomes 1, the second becomes 2, and so on.

Run from project root:

    # one sequence
    python scripts/renumber_track_ids.py \
        --input  data/processed/tracker_outputs/bytetrack_cmc/yolov8s_baseline/MultiUAV-204.txt \
        --output data/processed/tracker_outputs/bytetrack_cmc_renum/yolov8s_baseline/MultiUAV-204.txt

    # whole directory
    python scripts/renumber_track_ids.py \
        --input-dir  data/processed/tracker_outputs/bytetrack_cmc/yolov8s_baseline \
        --output-dir data/processed/tracker_outputs/bytetrack_cmc_renum/yolov8s_baseline
"""

import argparse
from pathlib import Path


def renumber_file(in_path: Path, out_path: Path) -> tuple[int, int]:
    """Renumber a single MOT file. Returns (n_lines, n_unique_ids)."""
    remap: dict[int, int] = {}
    out_lines: list[str] = []
    with in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            old_id = int(parts[1])
            if old_id not in remap:
                remap[old_id] = len(remap) + 1
            parts[1] = str(remap[old_id])
            out_lines.append(",".join(parts))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(out_lines), len(remap)


def parse_args():
    p = argparse.ArgumentParser(
        description="Renumber MOT track IDs to start from 1 per file."
    )
    p.add_argument("--input",      type=str, help="single MOT file to renumber")
    p.add_argument("--output",     type=str, help="output path for --input mode")
    p.add_argument("--input-dir",  type=str, help="directory of MOT .txt files to renumber")
    p.add_argument("--output-dir", type=str, help="output directory (mirrors --input-dir)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.input and args.output:
        n_lines, n_ids = renumber_file(Path(args.input), Path(args.output))
        print(f"  {Path(args.input).name}  lines={n_lines:,}  unique_ids={n_ids}")
        return
    if args.input_dir and args.output_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir)
        files = sorted(in_dir.glob("*.txt"))
        if not files:
            raise SystemExit(f"no .txt files found in {in_dir}")
        print(f"renumbering {len(files)} files: {in_dir}  ->  {out_dir}\n")
        for f in files:
            n_lines, n_ids = renumber_file(f, out_dir / f.name)
            print(f"  {f.name:<26}  lines={n_lines:>7,}  unique_ids={n_ids:>4}")
        print("\ndone.")
        return
    raise SystemExit(
        "specify either --input/--output (single file) "
        "OR --input-dir/--output-dir (whole directory)"
    )


if __name__ == "__main__":
    main()
