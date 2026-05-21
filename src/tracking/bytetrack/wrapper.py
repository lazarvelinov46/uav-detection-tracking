"""
src/tracking/bytetrack/wrapper.py

Thin wrapper around the canonical BYTETracker, exposing a clean
`update(detections) -> list[Track]` interface and hiding the
YOLOX-era `img_info`/`img_size` plumbing we don't need (Ultralytics
already returns boxes in original-image coordinates).

One wrapper instance tracks one sequence. Create a fresh wrapper for
each new sequence so the tracker's internal state (active and lost
tracks, frame counter) resets cleanly.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .byte_tracker import BYTETracker


@dataclass
class Track:
    """One active track at a given frame."""
    track_id: int
    x: float        # top-left x in image pixels
    y: float        # top-left y in image pixels
    w: float        # box width in pixels
    h: float        # box height in pixels
    score: float    # detection confidence that produced this track


class ByteTrackWrapper:
    """
    Thin per-sequence wrapper around BYTETracker.

    Hyperparameters (paper defaults shown):
        track_thresh : float, default 0.5
            Confidence threshold separating "high" and "low" detections.
            High-conf feed first-stage association; low-conf feed the
            second-stage association unique to ByteTrack.
        track_buffer : int, default 30
            Frames to remember a lost track before removing it. At
            frame_rate=30 this is 1 second of "lost" tolerance.
        match_thresh : float, default 0.8
            IoU threshold for first-stage detection-to-track matching.
        mot20 : bool, default False
            Crowded-scene mode (disables score-fusion in first
            association). Worth trying True later given our dense regime.
        frame_rate : int, default 30
            Used internally to scale track_buffer.
        img_size : (H, W), default (512, 640)
            Image height and width in pixels. We pass img_info == img_size
            to BYTETracker so its internal scale factor is 1 (i.e., no
            rescaling — Ultralytics already gives original-image coords).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int   = 30,
        match_thresh: float = 0.8,
        mot20: bool         = False,
        frame_rate: int     = 30,
        img_size: tuple     = (512, 640),
    ):
        args = SimpleNamespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=mot20,
        )
        self.tracker  = BYTETracker(args, frame_rate=frame_rate)
        self.img_size = img_size

    def update(self, detections: np.ndarray) -> list:
        """
        Feed one frame's detections to the tracker. Frames must be
        fed in chronological order; frames with no detections must
        still be fed (pass np.empty((0, 5))) so the tracker's internal
        frame counter stays aligned with external frame numbering.

        Parameters
        ----------
        detections : np.ndarray
            (N, 5) array of [x1, y1, x2, y2, conf] in original-image
            pixels. For empty frames pass np.empty((0, 5)).

        Returns
        -------
        list[Track]
            Active tracks at this frame with assigned track_id and
            current bounding box (top-left + width + height).
        """
        if detections.size == 0:
            detections = np.empty((0, 5), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)
            if detections.shape[1] != 5:
                raise ValueError(
                    f"Expected detections of shape (N, 5) "
                    f"[x1, y1, x2, y2, conf]; got {detections.shape}"
                )

        # img_info == img_size -> scale = 1 (no rescaling needed).
        active_stracks = self.tracker.update(
            detections,
            img_info=self.img_size,
            img_size=self.img_size,
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
