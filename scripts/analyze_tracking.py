"""Phase 4 analysis - load TrackEval outputs into tidy DataFrames.

Foundation layer for the tracking-results visualization. Parses the per-sequence
`pedestrian_detailed.csv` TrackEval writes per tracker, stacks them into one long
DataFrame, separates the COMBINED (headline) row, and maps TrackEval's raw column
names + 0-1 scale onto clean, paper-ready metric columns on a 0-100 scale.

Run from the project root:
    python scripts/analyze_tracking.py
    python scripts/analyze_tracking.py --eval-root path/to/AntiUAV-val

Schema notes (verified against the real CSVs):
  * First column is `seq`; the last data row is `COMBINED`.
  * Detailed CSV values are FRACTIONS (0-1). We scale rate metrics x100 so they
    match the summary.txt / reported numbers; integer counts are left raw.
  * There is no plain `HOTA`/`DetA`/`AssA` column - the alpha-averaged scalars are
    `HOTA___AUC`, `DetA___AUC`, `AssA___AUC`. MOTA / IDF1 / IDSW are plain columns.

No `import torch` here on purpose: pandas only, so the Windows DLL-ordering
constraint (ultralytics / byte_tracker / albumentations) does not apply.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_EVAL_ROOT = Path("data/eval/trackers/AntiUAV-val")
DETAILED_NAME = "pedestrian_detailed.csv"

# clean display name -> (source column in the detailed CSV, kind)
# kind "rate"  : 0-1 in the CSV, scaled x100 for reporting
# kind "count" : integer counts, left as-is
METRIC_MAP: dict[str, tuple[str, str]] = {
    "HOTA":   ("HOTA___AUC", "rate"),
    "DetA":   ("DetA___AUC", "rate"),
    "AssA":   ("AssA___AUC", "rate"),
    "DetRe":  ("DetRe___AUC", "rate"),
    "AssRe":  ("AssRe___AUC", "rate"),
    "LocA":   ("LocA___AUC", "rate"),
    "MOTA":   ("MOTA", "rate"),
    "IDF1":   ("IDF1", "rate"),
    "CLR_Re": ("CLR_Re", "rate"),   # detection recall (CLEAR)
    "CLR_Pr": ("CLR_Pr", "rate"),   # detection precision (CLEAR)
    "IDSW":   ("IDSW", "count"),
    "CLR_FN": ("CLR_FN", "count"),  # false negatives
    "CLR_FP": ("CLR_FP", "count"),  # false positives
    "Frag":   ("Frag", "count"),
    "GT_Dets": ("GT_Dets", "count"),
}

# The subset shown in the headline sanity print.
HEADLINE = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]


def load_detailed(eval_root: Path = DEFAULT_EVAL_ROOT) -> pd.DataFrame:
    """Read every <tracker>/pedestrian_detailed.csv under eval_root into one frame.

    Each row is a (tracker, sequence) pair plus all of TrackEval's raw columns.
    The COMBINED row is kept here and separated by split_combined().
    """
    eval_root = Path(eval_root)
    if not eval_root.exists():
        raise FileNotFoundError(f"Eval root does not exist: {eval_root}")

    frames = []
    for tracker_dir in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        csv_path = tracker_dir / DETAILED_NAME
        if not csv_path.exists():
            print(f"[skip] no {DETAILED_NAME} in {tracker_dir.name}")
            continue
        df = pd.read_csv(csv_path)
        df = df.rename(columns={df.columns[0]: "seq"})
        # Prepend the tracker column via concat (insert on a ~270-col frame
        # triggers a pandas fragmentation warning).
        tag = pd.DataFrame({"tracker": [tracker_dir.name] * len(df)})
        df = pd.concat([tag, df], axis=1)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No {DETAILED_NAME} files found under {eval_root}. "
            "Check the path or re-run eval with detailed output enabled."
        )
    return pd.concat(frames, ignore_index=True)


def split_combined(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (per_sequence, combined). The combined frame is the headline."""
    combined = df[df["seq"] == "COMBINED"].copy().reset_index(drop=True)
    per_seq = df[df["seq"] != "COMBINED"].copy().reset_index(drop=True)
    return per_seq, combined


def to_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Project raw TrackEval columns onto clean metric names (rates x100).

    Returns tracker, seq, and every metric in METRIC_MAP whose source column is
    present. Missing source columns are skipped (with a warning) rather than
    raising, so a slightly different TrackEval version still yields a usable frame.
    """
    out = df[["tracker", "seq"]].copy()
    for clean, (src, kind) in METRIC_MAP.items():
        if src not in df.columns:
            print(f"[warn] column {src!r} not found - skipping metric {clean!r}")
            continue
        vals = pd.to_numeric(df[src], errors="coerce")
        out[clean] = vals * 100.0 if kind == "rate" else vals
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    args = parser.parse_args()

    raw = load_detailed(args.eval_root)
    per_seq_raw, combined_raw = split_combined(raw)

    per_seq = to_metrics(per_seq_raw)
    combined = to_metrics(combined_raw)

    print(f"\nLoaded from: {args.eval_root}")
    print(f"Trackers found: {sorted(raw['tracker'].unique())}")
    print("Per-sequence row counts (expect 39 each):")
    print(per_seq.groupby("tracker").size().to_string())

    cols = [c for c in HEADLINE if c in combined.columns]
    headline = combined.set_index("tracker")[cols].round(2)
    print("\nHeadline (COMBINED) - sanity-check against your known table:")
    print(headline.to_string())


if __name__ == "__main__":
    main()
