"""Phase 4 Task B - FPS visualization.

Reads outputs/fps/summary.csv (produced by benchmark_fps.py) and writes
two paper-ready figures:

  1. fps_comparison.{png,pdf}
     End-to-end throughput across every (detector, tracker, device) cell,
     log y-axis so the 200x range from DeepSORT (~1 FPS) to ByteTrack
     on T4 (~80 FPS) fits in one frame. 25 FPS reference line marks the
     real-time bar.

  2. fps_latency_composition.{png,pdf}
     Stacked detector+tracker latency for the real-time-relevant subset
     (ByteTrack + ByteTrack+CMC). DeepSORT is excluded - its ~1000 ms
     tracker time crushes the scale and the verdict for it is already
     in Plot 1. 40 ms reference line marks the same 25 FPS bar in time.

Run from project root:
    python scripts/plot_fps.py
    python scripts/plot_fps.py --summary outputs/fps/summary.csv --out outputs/fps

matplotlib + pandas only, no torch ordering constraint applies.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Tracker palette - same Okabe-Ito choices as plot_tracking.py so the
# colors carry meaning across the analysis and the FPS plots.
TRACKER_COLORS = {
    "bytetrack":     "#0072B2",  # blue
    "bytetrack_cmc": "#CC79A7",  # reddish purple
    "deepsort":      "#D55E00",  # vermillion
}
TRACKER_LABELS = {
    "bytetrack":     "ByteTrack",
    "bytetrack_cmc": "ByteTrack+CMC",
    "deepsort":      "DeepSORT",
}
DETECTOR_LABELS = {"yolov8s": "YOLOv8s", "yolox": "YOLOX-S"}
DEVICE_LABELS   = {"cpu": "local CPU", "cuda": "Kaggle T4 GPU"}

TRACKER_ORDER  = ["bytetrack", "bytetrack_cmc", "deepsort"]
DETECTOR_ORDER = ["yolov8s", "yolox"]
DEVICE_ORDER   = ["cpu", "cuda"]

REAL_TIME_FPS = 25
REAL_TIME_MS  = 1000.0 / REAL_TIME_FPS  # 40 ms


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If a config was benched more than once, keep the latest row.
    df = df.drop_duplicates(subset=["detector", "tracker", "device"], keep="last")
    return df


def _get(df, det, trk, dev, col):
    """Pull a single scalar; returns NaN if the config wasn't benched."""
    row = df[(df["detector"] == det) & (df["tracker"] == trk) & (df["device"] == dev)]
    return float(row[col].iloc[0]) if len(row) else float("nan")


def plot_fps_comparison(df: pd.DataFrame, out: Path) -> Path:
    """End-to-end FPS, all configs, log y. 25 FPS reference line."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    x = np.arange(len(TRACKER_ORDER))
    width = 0.38

    for ax, dev in zip(axes, DEVICE_ORDER):
        for di, det in enumerate(DETECTOR_ORDER):
            offs = x - width / 2 + di * width
            vals = [_get(df, det, t, dev, "e2e_fps") for t in TRACKER_ORDER]
            colors = [TRACKER_COLORS[t] for t in TRACKER_ORDER]
            bars = ax.bar(offs, vals, width=width,
                          color=colors, edgecolor="white", linewidth=0.5,
                          hatch="" if det == "yolov8s" else "///",
                          alpha=0.9)
            for b, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                ax.annotate(f"{v:.1f}",
                            (b.get_x() + b.get_width() / 2, v),
                            textcoords="offset points", xytext=(0, 4),
                            ha="center", fontsize=8)

        ax.set_yscale("log")
        ax.axhline(REAL_TIME_FPS, color="0.4", ls="--", lw=1)
        ax.text(len(TRACKER_ORDER) - 0.5, REAL_TIME_FPS,
                f" {REAL_TIME_FPS} FPS",
                fontsize=8, va="bottom", ha="right", color="0.4")
        ax.set_xticks(x)
        ax.set_xticklabels([TRACKER_LABELS[t] for t in TRACKER_ORDER])
        ax.set_title(DEVICE_LABELS[dev])
        ax.grid(axis="y", which="both", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("End-to-end FPS (log scale)")

    # One legend for detector encoding (color = tracker, hatch = detector).
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#888", edgecolor="white",
                      hatch="",    alpha=0.9),
        plt.Rectangle((0, 0), 1, 1, facecolor="#888", edgecolor="white",
                      hatch="///", alpha=0.9),
    ]
    fig.legend(handles, [DETECTOR_LABELS[d] for d in DETECTOR_ORDER],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()

    png = out / "fps_comparison.png"
    pdf = out / "fps_comparison.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print(f"[plot] wrote {pdf}")
    return png


def plot_latency_composition(df: pd.DataFrame, out: Path) -> Path:
    """Stacked detector+tracker latency for ByteTrack and ByteTrack+CMC.

    DeepSORT is excluded because its ~1000 ms tracker time crushes the
    scale and obscures the ByteTrack vs CMC comparison; its real-time
    verdict (always ~1 FPS) is already legible in fps_comparison.
    """
    trackers = ["bytetrack", "bytetrack_cmc"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=False)

    for ax, dev in zip(axes, DEVICE_ORDER):
        labels, det_vals, trk_vals, fps_vals, trk_colors = [], [], [], [], []
        for trk in trackers:
            for det in DETECTOR_ORDER:
                d = _get(df, det, trk, dev, "det_mean_ms")
                t = _get(df, det, trk, dev, "trk_mean_ms")
                f = _get(df, det, trk, dev, "e2e_fps")
                if np.isnan(d) or np.isnan(t):
                    continue
                labels.append(f"{DETECTOR_LABELS[det]}\n{TRACKER_LABELS[trk]}")
                det_vals.append(d)
                trk_vals.append(t)
                fps_vals.append(f)
                trk_colors.append(TRACKER_COLORS[trk])

        x = np.arange(len(labels))
        ax.bar(x, det_vals, color="#999999",
               edgecolor="white", linewidth=0.5, label="detector")
        ax.bar(x, trk_vals, bottom=det_vals, color=trk_colors,
               edgecolor="white", linewidth=0.5, label="tracker")

        # FPS + real-time verdict above each bar
        for i, (d, t, f) in enumerate(zip(det_vals, trk_vals, fps_vals)):
            verdict = "\u2713" if f >= REAL_TIME_FPS else "\u2717"
            ax.annotate(f"{verdict} {f:.1f} FPS", (i, d + t),
                        textcoords="offset points", xytext=(0, 4),
                        ha="center", fontsize=8.5)

        ax.axhline(REAL_TIME_MS, color="r", ls="--", lw=1, alpha=0.6)
        # Annotate the threshold line at the right edge of the panel.
        ax.text(len(labels) - 0.5, REAL_TIME_MS,
                f" {REAL_TIME_FPS} FPS = {REAL_TIME_MS:.0f} ms",
                fontsize=8, va="bottom", ha="right", color="r", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(DEVICE_LABELS[dev])
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        # Headroom for the annotation
        ymax = max((d + t for d, t in zip(det_vals, trk_vals)), default=50)
        ax.set_ylim(0, ymax * 1.18)

    axes[0].set_ylabel("Latency (ms)")
    fig.suptitle("Pipeline latency \u2014 detector + tracker "
                 "(DeepSORT excluded for scale)", y=1.02)
    fig.tight_layout()

    png = out / "fps_latency_composition.png"
    pdf = out / "fps_latency_composition.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {png}")
    print(f"[plot] wrote {pdf}")
    return png


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path,
                        default=Path("outputs/fps/summary.csv"))
    parser.add_argument("--out", type=Path, default=Path("outputs/fps"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = load_summary(args.summary)
    print(f"Loaded {len(df)} configurations from {args.summary}")
    plot_fps_comparison(df, args.out)
    plot_latency_composition(df, args.out)


if __name__ == "__main__":
    main()
