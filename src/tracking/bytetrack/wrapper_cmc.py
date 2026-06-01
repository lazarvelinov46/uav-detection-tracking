"""
src/tracking/bytetrack/wrapper_cmc.py

Per-sequence wrapper around BYTETrackerCMC with an integrated Global
Motion Compensation (GMC) estimator. Same shape as ByteTrackWrapper
but with one extra parameter on update(): the current frame's image,
from which the camera-motion affine is estimated.

What this wrapper composes:
  - BYTETrackerCMC : ByteTrack with multi_gmc injected into multi_predict
                     (see cmc_tracker.py); wraps the vendored BYTETracker.
  - GMC            : ORB + Lowe + RANSAC affine estimator with detection
                     masking and identity fallback (see gmc.py).

Per frame:
  1. GMC estimates the affine H from the previous frame to this frame,
     masking out the detection boxes so the estimate reflects camera/
     background motion rather than UAV motion.
  2. The tracker is fed (detections, H); inside its update() the
     temporary STrack.multi_predict swap warps each track's Kalman
     state by H right after the constant-velocity predict and before
     the IoU association step.

Per-call GMC diagnostics (n_keypoints / n_matches / n_inliers /
fallback) are exposed on `.last_gmc_stats` so the runner can log how
often the estimator engages vs. falls back to identity -- the
diagnostic for the low-texture-thermal-IR failure mode flagged at
design time.

One wrapper instance tracks one sequence. Both the tracker AND the GMC
estimator are stateful (the GMC holds the previous frame's keypoints),
so create a fresh wrapper per sequence.
"""

import torch  # noqa: F401  # CRITICAL on Windows: precede the numpy chain

from types import SimpleNamespace

import numpy as np

from .cmc_tracker import BYTETrackerCMC
from .gmc import GMC
from .wrapper import Track   # reuse the same per-track output dataclass


# Same convention as ByteTrackWrapper: img_info == img_size yields
# scale = 1 inside the vendored BYTETracker.update(), disabling its
# YOLOX-era rescaling.
_SCALE_DISABLE = (1, 1)


class ByteTrackCMCWrapper:
    """
    Per-sequence ByteTrack-with-CMC wrapper.

    Constructor parameters mirror ByteTrackWrapper's tracker args,
    plus GMC hyperparameters with sensible defaults matching the
    validated estimator. All GMC defaults are tunable per run.

    Tracker args (paper defaults; baseline uses track_thresh=0.3):
        track_thresh, track_buffer, match_thresh, mot20, frame_rate

    GMC args (validated defaults):
        gmc_downscale            : 2
        gmc_n_features           : 1000
        gmc_ratio                : 0.75    (Lowe ratio-test threshold)
        gmc_ransac_reproj_thresh : 3.0     (pixels, downscaled space)
        gmc_min_matches          : 10      (attempt-fit threshold)
        gmc_min_inliers          : 6       (trust-fit threshold)
        gmc_det_mask_margin      : 0.0     (no padding around det boxes)
    """

    def __init__(
        self,
        # --- ByteTrack args (identical to ByteTrackWrapper) ----------
        track_thresh: float = 0.5,
        track_buffer: int   = 30,
        match_thresh: float = 0.8,
        mot20: bool         = False,
        frame_rate: int     = 30,
        # --- GMC args -------------------------------------------------
        gmc_downscale: int  = 2,
        gmc_n_features: int = 1000,
        gmc_ratio: float    = 0.75,
        gmc_ransac_reproj_thresh: float = 3.0,
        gmc_min_matches: int = 10,
        gmc_min_inliers: int = 6,
        gmc_det_mask_margin: float = 0.0,
    ):
        args = SimpleNamespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=mot20,
        )
        self.tracker = BYTETrackerCMC(args, frame_rate=frame_rate)
        self.gmc = GMC(
            downscale=gmc_downscale,
            n_features=gmc_n_features,
            ratio=gmc_ratio,
            ransac_reproj_thresh=gmc_ransac_reproj_thresh,
            min_matches=gmc_min_matches,
            min_inliers=gmc_min_inliers,
            det_mask_margin=gmc_det_mask_margin,
        )
        self.last_gmc_stats: dict = {}

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list:
        """
        Feed one frame's detections AND image to the tracker.

        Frames must be fed in chronological order. Empty-detection
        frames must still be fed so (a) the tracker's frame counter
        stays aligned and (b) GMC keeps its previous-frame buffer
        fresh -- otherwise a multi-frame detection gap would make GMC
        compare across the gap and recover an outsized "motion."

        Parameters
        ----------
        detections : np.ndarray
            (N, 5) [x1, y1, x2, y2, conf] in full-res image pixels.
            For empty frames pass np.empty((0, 5)).
        frame : np.ndarray
            HxWx3 BGR frame at full resolution (cv2.imread output).

        Returns
        -------
        list[Track]
            Active tracks at this frame with assigned track_id and
            current bounding box (top-left + width + height).
        """
        # Normalize the detections array shape.
        if detections.size == 0:
            detections = np.empty((0, 5), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)
            if detections.shape[1] != 5:
                raise ValueError(
                    f"Expected detections of shape (N, 5) "
                    f"[x1, y1, x2, y2, conf]; got {detections.shape}"
                )

        # Estimate camera motion from the frame. GMC.apply slices the
        # first 4 columns internally, so passing the full (N, 5) array
        # is fine; it also handles the empty case (returns identity).
        H = self.gmc.apply(frame, detections)
        self.last_gmc_stats = self.gmc.last_stats

        active_stracks = self.tracker.update(
            detections,
            img_info=_SCALE_DISABLE,
            img_size=_SCALE_DISABLE,
            H=H,
        )

        return [
            Track(
                track_id=int(t.track_id),
                x=float(t.tlwh[0]),
                y=float(t.tlwh[1]),
                w=float(t.tlwh[2]),
                h=float(t.tlwh[3]),
                score=float(t.score),
            )
            for t in active_stracks
        ]
