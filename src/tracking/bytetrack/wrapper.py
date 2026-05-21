"""
src/tracking/bytetrack/wrapper.py

Thin wrapper around the canonical BYTETracker, exposing a clean
`update(detections) -> list[Track]` interface and hiding the
YOLOX-era `img_info`/`img_size` plumbing we don't need (Ultralytics
already returns boxes in original-image coordinates).

Dataset-agnostic: BYTETracker's internal rescaling is disabled
unconditionally (see _SCALE_DISABLE below), so no image-size
configuration is required.

One wrapper instance tracks one sequence. Create a fresh wrapper
per sequence so the tracker's internal state resets cleanly.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from .byte_tracker import BYTETracker


# Passing img_info == img_size to BYTETracker yields scale = 1 regardless
# of the tuple's contents (scale = min(img_size[0]/img_info[0],
# img_size[1]/img_info[1]) = 1 when the tuples are equal). The values
# below are arbitrary — they only need to match each other.
_SCALE_DISABLE = (1, 1)


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
            Frames to remember a lost track before removing it.
        match_thresh : float, default 0.8
            IoU threshold for first-stage detection-to-track matching.
        mot20 : bool, default False
            Crowded-scene mode (disables score-fusion in first
            association). Worth trying True later for our dense regime.
        frame_rate : int, default 30
            Used internally to scale track_buffer to time-based units.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int   = 30,
        match_thresh: float = 0.8,
        mot20: bool         = False,
        frame_rate: int     = 30,
    ):
        args = SimpleNamespace(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            mot20=mot20,
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, detections: np.ndarray) -> list:
        """
        Feed one frame's detections to the tracker. Frames must be
        fed in chronological order; frames with no detections must
        still be fed (pass np.empty((0, 5))) so the tracker's internal
        frame counter stays aligned with external frame numbering.

        Parameters
        ----------
        detections : np.ndarray
            (N, 5) array of [x1, y1, x2, y2, conf] in image pixels.
            For empty frames pass np.empty((0, 5)).

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

        active_stracks = self.tracker.update(
            detections,
            img_info=_SCALE_DISABLE,
            img_size=_SCALE_DISABLE,
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
