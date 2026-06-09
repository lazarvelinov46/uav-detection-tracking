"""Phase 4 analysis - plots and the paper results table.

Imports the loader from analyze_tracking.py and produces, for now:
  1. A clean results table (CSV + LaTeX) of the COMBINED headline metrics.
  2. The cross-tracker bar chart: HOTA / MOTA / IDF1 grouped by metric.

Later pieces (HOTA decomposition, track_thresh sweep, per-sequence density view)
will be added here as new functions.

Run from the project root:
    python scripts/plot_tracking.py
    python scripts/plot_tracking.py --eval-root <path> --out <dir>

matplotlib + pandas only, so no `import torch` ordering constraint applies.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the sibling loader importable however this script is launched.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_tracking import load_detailed, split_combined, to_metrics  # noqa: E402

# Canonical tracker order + human-readable labels for tables and legends.
ORDER = ["bytetrack_t030", "bytetrack_t040", "bytetrack_t050",
         "bytetrack_cmc", "deepsort"]
LABELS = {
    "bytetrack_t030": "ByteTrack (\u03c4=0.3)",
    "bytetrack_t040": "ByteTrack (\u03c4=0.4)",
    "bytetrack_t050": "ByteTrack (\u03c4=0.5)",
    "bytetrack_cmc": "ByteTrack+CMC",
    "deepsort": "DeepSORT",
}
# Colourblind-friendly (Okabe-Ito), one per tracker.
COLORS = {
    "bytetrack_t030": "#0072B2",
    "bytetrack_t040": "#009E73",
    "bytetrack_t050": "#E69F00",
    "bytetrack_cmc": "#CC79A7",
    "deepsort": "#D55E00",
}

TABLE_COLS = ["HOTA", "DetA", "AssA", "MOTA", "IDF1", "IDSW"]
BAR_METRICS = ["HOTA", "MOTA", "IDF1"]


def ordered_trackers(df: pd.DataFrame) -> list[str]:
    """Trackers present, in canonical order, with any extras appended."""
    present = list(df["tracker"].unique())
    known = [t for t in ORDER if t in present]
    extra = [t for t in present if t not in ORDER]
    return known + extra


def export_table(combined: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Write the headline table as CSV + LaTeX (booktabs) and return it."""
    trackers = ordered_trackers(combined)
    cols = [c for c in TABLE_COLS if c in combined.columns]
    tbl = combined.set_index("tracker").loc[trackers, cols].copy()
    tbl.insert(0, "Tracker", [LABELS.get(t, t) for t in trackers])

    rate_cols = [c for c in cols if c != "IDSW"]
    tbl[rate_cols] = tbl[rate_cols].round(2)
    if "IDSW" in tbl.columns:
        tbl["IDSW"] = tbl["IDSW"].astype(int)

    csv_path = out / "results_table.csv"
    tex_path = out / "results_table.tex"
    tbl.to_csv(csv_path, index=False)
    tbl.to_latex(
        tex_path, index=False, escape=False, float_format="%.2f",
        caption="Combined tracking metrics on the Anti-UAV v4 Track 3 "
                "validation split (39 sequences).",
        label="tab:tracking_results",
    )
    print(f"[table] wrote {csv_path}")
    print(f"[table] wrote {tex_path}")
    print("\nResults table:")
    print(tbl.to_string(index=False))
    return tbl


def plot_headline_bars(combined: pd.DataFrame, out: Path) -> Path:
    """Grouped bar chart: HOTA / MOTA / IDF1, four trackers per metric group."""
    trackers = ordered_trackers(combined)
    metrics = [m for m in BAR_METRICS if m in combined.columns]
    data = combined.set_index("tracker").loc[trackers, metrics]

    n_groups = len(metrics)
    n_bars = len(trackers)
    group_w = 0.8
    bar_w = group_w / n_bars
    x = range(n_groups)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, t in enumerate(trackers):
        offs = [xi - group_w / 2 + bar_w * (i + 0.5) for xi in x]
        vals = data.loc[t].values
        bars = ax.bar(offs, vals, width=bar_w, label=LABELS.get(t, t),
                      color=COLORS.get(t), edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score (0\u2013100)")
    ax.set_ylim(0, 100)
    ax.set_title("Cross-tracker comparison \u2014 Anti-UAV v4 Track 3 (val)")
    ax.legend(frameon=False, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.08))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    png = out / "cross_tracker_bars.png"
    pdf = out / "cross_tracker_bars.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")  # vector, for the paper
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print(f"[plot] wrote {pdf}")
    return png


def plot_hota_decomposition(combined: pd.DataFrame, out: Path) -> Path:
    """Scatter each tracker in DetA-AssA space, with iso-HOTA reference curves.

    Per the HOTA paper, HOTA = sqrt(DetA * AssA) at a given threshold, so curves
    of constant HOTA are AssA = HOTA^2 / DetA. Plotting them shows that at this
    operating point the gradient is steeper along AssA than DetA - i.e. for these
    trackers, association is the cheaper axis on which to gain HOTA.
    """
    trackers = ordered_trackers(combined)
    df = combined.set_index("tracker").loc[trackers]
    det_vals = df["DetA"].to_numpy()
    ass_vals = df["AssA"].to_numpy()
    hota_vals = df["HOTA"].to_numpy()

    # Auto axes: pad each range proportionally, with a minimum floor so very
    # tight clusters still leave room for labels. Works for any tracker set.
    def _lim(arr, floor=2.5, frac=0.18):
        rng = float(arr.max() - arr.min())
        pad = max(floor, rng * frac)
        return (float(arr.min()) - pad, float(arr.max()) + pad)
    xlim, ylim = _lim(det_vals), _lim(ass_vals)

    fig, ax = plt.subplots(figsize=(7.5, 6))

    # Iso-HOTA curves: ~4-5 integer levels spanning the data range.
    lo = int(np.floor(hota_vals.min() - 1))
    hi = int(np.ceil(hota_vals.max() + 1))
    step = max(2, (hi - lo) // 4)
    levels = list(range(lo, hi + 1, step))

    det = np.linspace(*xlim, 300)
    for level in levels:
        ass = level ** 2 / det
        ax.plot(det, ass, color="0.82", lw=1, zorder=1)
        x_lab = level ** 2 / ylim[1]  # where the curve exits the top
        if xlim[0] < x_lab < xlim[1]:
            ax.annotate(f"HOTA {level}", (x_lab, ylim[1]), color="0.6",
                        fontsize=8, ha="center", va="bottom")

    # Per-tracker label offsets (the three strong trackers cluster tightly,
    # so they need fanning out) with faint leader lines back to each point.
    annot = {
        "bytetrack_t030": (16, -10, "left"),
        "deepsort": (10, -34, "left"),
        "bytetrack_t040": (-12, 20, "right"),
        "bytetrack_t050": (14, 8, "left"),
        "bytetrack_cmc": (14, 24, "left"),
    }
    for t in trackers:
        d, a, h = df.loc[t, "DetA"], df.loc[t, "AssA"], df.loc[t, "HOTA"]
        ax.scatter(d, a, s=140, color=COLORS.get(t), zorder=3,
                   edgecolor="white", linewidth=1.2)
        dx, dy, ha = annot.get(t, (9, 5, "left"))
        ax.annotate(f"{LABELS.get(t, t)}\nHOTA {h:.1f}", (d, a),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8.5,
                    ha=ha, va="center",
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.6))

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Detection Accuracy \u2014 DetA (0\u2013100)")
    ax.set_ylabel("Association Accuracy \u2014 AssA (0\u2013100)")
    ax.set_title("HOTA decomposition \u2014 detection vs association")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    png = out / "hota_decomposition.png"
    pdf = out / "hota_decomposition.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print(f"[plot] wrote {pdf}")
    return png


def _thresh_from_name(name: str) -> float:
    """bytetrack_t030 -> 0.30."""
    tag = name.split("_t")[-1]
    return int(tag) / 100.0


def plot_thresh_sweep(combined: pd.DataFrame, out: Path) -> Path:
    """Line plot of the ByteTrack track_thresh sweep.

    Left axis (0-100): HOTA, AssA, detection Recall. Right axis: false-negative
    count. Shows that raising the threshold sheds recall (FN climbs steeply) for
    only a marginal association gain, so HOTA falls and 0.3 is the best operating
    point. DeepSORT is excluded - it has no threshold sweep.
    """
    # Only the threshold-sweep ByteTrack variants (bytetrack_t<NNN>) belong
    # here; CMC or other variants must be excluded.
    sweep_pat = re.compile(r"^bytetrack_t\d+$")
    sub = combined[combined["tracker"].str.match(sweep_pat)].copy()
    sub["thresh"] = sub["tracker"].map(_thresh_from_name)
    sub = sub.sort_values("thresh")
    x = sub["thresh"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax2 = ax.twinx()

    series = [
        ("HOTA", "HOTA", "#222222", "o"),
        ("AssA", "AssA", "#009E73", "s"),
        ("Recall", "CLR_Re", "#0072B2", "^"),
    ]
    for label, col, color, marker in series:
        if col not in sub.columns:
            continue
        y = sub[col].to_numpy()
        ax.plot(x, y, marker=marker, color=color, label=label, lw=2, zorder=3)
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:.1f}", (xi, yi), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8, color=color)

    if "CLR_FN" in sub.columns:
        fn = sub["CLR_FN"].to_numpy()
        ax2.plot(x, fn, marker="D", color="0.55", ls="--", lw=1.6,
                 label="False negatives", zorder=2)
        for xi, yi in zip(x, fn):
            ax2.annotate(f"{yi/1000:.0f}k", (xi, yi), textcoords="offset points",
                         xytext=(0, -15), ha="center", fontsize=8, color="0.4")

    ax.set_xticks(x)
    ax.set_xlabel("ByteTrack track_thresh")
    ax.set_ylabel("Score (0\u2013100)")
    ax2.set_ylabel("False negatives (count)")
    ax.set_ylim(40, 95)
    ax.set_title("track_thresh sweep \u2014 recall traded for association")
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="center left")
    fig.tight_layout()

    png = out / "thresh_sweep.png"
    pdf = out / "thresh_sweep.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print(f"[plot] wrote {pdf}")
    return png


SEQ_RE = re.compile(r"(MultiUAV-\d+)")
DENSITY_COLORS = {"sparse": "#009E73", "medium": "#E69F00", "dense": "#D55E00"}


def _seq_of(f: Path) -> str | None:
    """Extract 'MultiUAV-XXX' from a label file's parent dir or filename."""
    m = SEQ_RE.search(f.parent.name) or SEQ_RE.search(f.name)
    return m.group(1) if m else None


def _looks_like_mot(f: Path) -> bool:
    """MOT gt.txt has >=7 comma/space fields per row; YOLO labels have 5."""
    if f.name.lower() == "gt.txt":
        return True
    first = next((ln for ln in f.read_text().splitlines() if ln.strip()), "")
    return len(first.replace(" ", ",").split(",")) >= 7


def _mot_density(f: Path) -> float:
    """Mean rows-per-frame for a single MOT-format file (frame is column 0)."""
    per_frame: dict[str, int] = defaultdict(int)
    for ln in f.read_text().splitlines():
        if ln.strip():
            per_frame[ln.replace(" ", ",").split(",")[0]] += 1
    return sum(per_frame.values()) / len(per_frame) if per_frame else 0.0


def sequence_density(labels_root: Path) -> dict[str, float]:
    """Mean UAV count per frame per sequence, recomputed from label files.

    Handles two layouts automatically:
      * YOLO per-frame labels (one .txt per frame, one line per box) -> mean
        non-empty lines per frame across the sequence.
      * A single MOT-format file per sequence (e.g. gt.txt) -> mean rows per
        frame, grouping by the frame column.
    Sequence name is parsed as 'MultiUAV-XXX' from the parent dir or filename.
    Returns {} (with a message) if the root is missing or yields nothing.
    """
    labels_root = Path(labels_root)
    if not labels_root.exists():
        print(f"[per-seq] labels root not found: {labels_root} - skipping plot")
        return {}

    groups: dict[str, list[Path]] = defaultdict(list)
    for f in labels_root.rglob("*.txt"):
        seq = _seq_of(f)
        if seq:
            groups[seq].append(f)

    density: dict[str, float] = {}
    for seq, files in groups.items():
        if len(files) == 1 and _looks_like_mot(files[0]):
            density[seq] = _mot_density(files[0])
        else:
            counts = [sum(1 for ln in f.read_text().splitlines() if ln.strip())
                      for f in files]
            density[seq] = sum(counts) / len(counts) if counts else 0.0

    if not density:
        print(f"[per-seq] no MultiUAV-* label files under {labels_root}")
    return density


def plot_per_sequence(per_seq: pd.DataFrame, density: dict[str, float],
                      out: Path, primary: str = "bytetrack_t030",
                      xlabel: str = "Mean UAV count per frame (sequence density)"
                      ) -> Path | None:
    """Scatter per-sequence HOTA vs density (primary tracker), binned by tercile.

    Makes the density effect legible: HOTA degrades as the mean UAV count per
    frame rises (sparse sequences track near-perfectly; dense ones fragment).
    The lowest-HOTA sequence is annotated to surface the detector failure case.
    """
    if not density:
        return None
    sub = per_seq[per_seq["tracker"] == primary][["seq", "HOTA"]].copy()
    sub["density"] = sub["seq"].map(density)
    sub = sub.dropna(subset=["density"])
    if sub.empty:
        print("[per-seq] no overlap between metric sequences and label files")
        return None

    # Tercile bins via percentile rank (robust to tied densities).
    rank = sub["density"].rank(pct=True, method="first")
    sub["bin"] = pd.cut(rank, [0, 1 / 3, 2 / 3, 1.0],
                        labels=["sparse", "medium", "dense"], include_lowest=True)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for b, c in DENSITY_COLORS.items():
        d = sub[sub["bin"] == b]
        ax.scatter(d["density"], d["HOTA"], s=60, color=c, label=b,
                   edgecolor="white", linewidth=0.5, zorder=3)

    if len(sub) >= 2:  # faint linear trend
        z = np.polyfit(sub["density"], sub["HOTA"], 1)
        xs = np.linspace(sub["density"].min(), sub["density"].max(), 50)
        ax.plot(xs, np.polyval(z, xs), color="0.5", ls="--", lw=1, zorder=2)

    worst = sub.loc[sub["HOTA"].idxmin()]
    ax.annotate(worst["seq"], (worst["density"], worst["HOTA"]),
                textcoords="offset points", xytext=(7, -2), fontsize=8,
                color="0.3")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("HOTA (0\u2013100)")
    ax.set_title(f"Per-sequence HOTA vs density \u2014 {LABELS.get(primary, primary)}")
    ax.legend(frameon=False, title="density")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    png = out / "per_sequence_density.png"
    pdf = out / "per_sequence_density.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print(f"[plot] wrote {pdf}")
    return png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path,
                        default=Path("data/eval/trackers/AntiUAV-val"))
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    parser.add_argument("--labels", type=Path,
                        default=Path("data/processed/anti_uav_v4/labels/val"),
                        help="Root of per-frame (or MOT) val labels for density")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = load_detailed(args.eval_root)
    per_seq_raw, combined_raw = split_combined(raw)
    combined = to_metrics(combined_raw)
    per_seq = to_metrics(per_seq_raw)

    export_table(combined, args.out)
    plot_headline_bars(combined, args.out)
    plot_hota_decomposition(combined, args.out)
    plot_thresh_sweep(combined, args.out)
    density = sequence_density(args.labels)
    if density:
        plot_per_sequence(per_seq, density, args.out)
    else:
        # Fallback: no per-frame labels found - use distinct UAVs per sequence
        # (GT_IDs) from the metrics CSV as the density measure.
        prim = per_seq[per_seq["tracker"] == "bytetrack_t030"]
        if "GT_IDs" in prim.columns:
            gt_density = dict(zip(prim["seq"], prim["GT_IDs"]))
            print("[per-seq] using GT_IDs (distinct UAVs/seq) as density fallback")
            plot_per_sequence(per_seq, gt_density, args.out,
                              xlabel="Distinct UAVs per sequence (GT_IDs)")
        else:
            print("[per-seq] no density source available - skipping plot")


if __name__ == "__main__":
    main()
