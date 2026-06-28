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


def plot_deepsort_embedder_ablation(df: pd.DataFrame, out: Path) -> Path | None:
    """DeepSORT embedder CPU vs GPU ablation, Kaggle T4 only.

    Two panels: tracker latency (mean bars + p99 whiskers) on the left,
    end-to-end FPS (log scale, 25 FPS bar) on the right. The story is the
    mean-vs-p99 split: GPU embedder improves mean ~4x but worsens the tail,
    and even the best mean stays far below real-time.

    This figure is self-contained - it does not touch the shared TRACKER_*
    constants used by the main FPS plots, so the canonical 3-tracker
    palette and ordering remain unchanged.
    """
    sub = df[(df["device"] == "cuda")
             & (df["tracker"].isin(["deepsort", "deepsort_embgpu"]))]
    if sub.empty:
        print("[ablation] no deepsort/deepsort_embgpu rows on cuda - "
              "skipping ablation plot")
        return None

    # Local colors/labels - not added to the shared dicts.
    embedders   = ["deepsort", "deepsort_embgpu"]
    emb_labels  = {"deepsort": "CPU embedder",
                   "deepsort_embgpu": "GPU embedder (fp16)"}
    emb_colors  = {"deepsort": "#D55E00",   # vermillion (matches main plots)
                   "deepsort_embgpu": "#56B4E9"}  # sky blue (ablation only)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.5))
    x = np.arange(len(DETECTOR_ORDER))
    width = 0.36

    # Panel 1: tracker latency, mean bars + p99 as upper whiskers
    all_p99 = []
    for i, emb in enumerate(embedders):
        means = [_get(df, det, emb, "cuda", "trk_mean_ms") for det in DETECTOR_ORDER]
        p99s  = [_get(df, det, emb, "cuda", "trk_p99_ms")  for det in DETECTOR_ORDER]
        all_p99.extend(p99s)
        offsets = x - width / 2 + i * width
        err_up = [max(0, p - m) for p, m in zip(p99s, means)]
        err_dn = [0.0] * len(means)
        bars = ax1.bar(offsets, means, width=width,
                       color=emb_colors[emb], alpha=0.9,
                       edgecolor="white", linewidth=0.5,
                       yerr=[err_dn, err_up], capsize=5,
                       error_kw=dict(ecolor="0.2", lw=1.2),
                       label=emb_labels[emb])
        for b, m in zip(bars, means):
            if not np.isnan(m):
                ax1.annotate(f"{m:.0f}", (b.get_x() + b.get_width() / 2, m),
                             textcoords="offset points", xytext=(0, 3),
                             ha="center", fontsize=8)
        for off, p in zip(offsets, p99s):
            if not np.isnan(p):
                ax1.annotate(f"p99 {p:.0f}", (off, p),
                             textcoords="offset points", xytext=(0, 5),
                             ha="center", fontsize=7.5, color="0.3")

    # 25% headroom above the tallest p99 label so it doesn't collide with title.
    valid_p99 = [p for p in all_p99 if not np.isnan(p)]
    if valid_p99:
        ax1.set_ylim(0, max(valid_p99) * 1.25)

    ax1.set_xticks(x)
    ax1.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTOR_ORDER])
    ax1.set_ylabel("Tracker latency (ms)")
    ax1.set_title("Tracker time \u2014 mean (bars) + p99 (whiskers)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)

    # Panel 2: end-to-end FPS, log scale, 25 FPS bar
    all_fps = []
    for i, emb in enumerate(embedders):
        fps_vals = [_get(df, det, emb, "cuda", "e2e_fps") for det in DETECTOR_ORDER]
        all_fps.extend(fps_vals)
        offsets = x - width / 2 + i * width
        bars = ax2.bar(offsets, fps_vals, width=width,
                       color=emb_colors[emb], alpha=0.9,
                       edgecolor="white", linewidth=0.5,
                       label=emb_labels[emb])
        for b, v in zip(bars, fps_vals):
            if not np.isnan(v):
                ax2.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                             textcoords="offset points", xytext=(0, 3),
                             ha="center", fontsize=8)

    ax2.set_yscale("log")
    # Fixed range with comfortable space above the 25 FPS line.
    valid_fps = [f for f in all_fps if not np.isnan(f) and f > 0]
    if valid_fps:
        ax2.set_ylim(min(valid_fps) * 0.6, REAL_TIME_FPS * 2.5)
    ax2.axhline(REAL_TIME_FPS, color="0.4", ls="--", lw=1)
    # 25 FPS reference label on the LEFT, above the line, using y-axis transform
    # (x in axes coords, y in data coords) so x position is independent of data.
    ax2.text(0.02, REAL_TIME_FPS, f" {REAL_TIME_FPS} FPS real-time bar",
             transform=ax2.get_yaxis_transform(),
             fontsize=8, va="bottom", ha="left", color="0.4")
    ax2.set_xticks(x)
    ax2.set_xticklabels([DETECTOR_LABELS[d] for d in DETECTOR_ORDER])
    ax2.set_ylabel("End-to-end FPS (log scale)")
    ax2.set_title("End-to-end throughput")
    ax2.grid(axis="y", which="both", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)

    # Single figure-level legend above both panels - removes the in-panel
    # legend that was colliding with p99 labels (left) and the 25 FPS line (right).
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=emb_colors[e],
                      alpha=0.9, edgecolor="white") for e in embedders
    ]
    fig.legend(handles, [emb_labels[e] for e in embedders],
               loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.94))

    fig.suptitle("DeepSORT embedder ablation \u2014 Kaggle T4", y=1.00)
    # Reserve top ~10% of figure for the suptitle + shared legend.
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    png = out / "fps_deepsort_ablation.png"
    pdf = out / "fps_deepsort_ablation.pdf"
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
    plot_deepsort_embedder_ablation(df, args.out)


if __name__ == "__main__":
    main()