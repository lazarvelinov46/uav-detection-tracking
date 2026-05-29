"""
src/tracking/bytetrack/cmc_tracker.py

ByteTrack with Camera Motion Compensation (CMC) -- a thin subclass of
the vendored BYTETracker that injects the GMC warp into the prediction
step WITHOUT modifying the vendored update() method.

Integration strategy
--------------------
The canonical BYTETracker.update() invokes `STrack.multi_predict(strack_pool)`
by class name. We temporarily swap that staticmethod, for the duration of
a single super().update() call, with a wrapped version that:
  1. runs the original constant-velocity predict on each track, then
  2. calls multi_gmc(strack_pool, H) to warp each track's Kalman state
     by the camera-motion affine H (no-op when H is None).

The wrapper is restored in a `finally` block so the global STrack class
is unchanged once update() returns. Net result: the vendored update()
body runs verbatim every call -- the only diff vs upstream is that
multi_predict has been transparently extended for the call's duration.

Why not copy the 80-line update() body and insert a line?
  - This way the vendored bytetrack/ package executes truly unmodified;
    the diff against upstream is zero lines of code, not "verbatim copy
    with one insertion" (which silently desyncs the day upstream changes).
  - The override here is ~10 lines of clear intent vs ~80 lines of
    "find the diff" -- easier to read, audit, and defend in writing.

Safety notes
------------
Not thread-safe by construction: the swap mutates the global STrack
class for the call's duration. This is fine for our setup (each tracker
runs serially in its own process during evaluation) but would need a
lock or thread-local indirection for concurrent use within one process.

Backward compatibility: update(...) accepts H as an optional kwarg
defaulting to None. With H=None, multi_gmc is a no-op and the tracker
behaves IDENTICALLY to the vendored BYTETracker -- a useful sanity-check
path for proving CMC is the sole source of any observed delta.
"""

import torch  # noqa: F401  # CRITICAL on Windows: precede the numpy chain.

from .byte_tracker import BYTETracker, STrack
from .gmc import multi_gmc


class BYTETrackerCMC(BYTETracker):
    """BYTETracker with optional Camera Motion Compensation (CMC).

    Pass a 2x3 affine H (mapping previous-frame -> current-frame image
    coordinates) per update() call. With H is None, behavior matches the
    base BYTETracker exactly.
    """

    def update(self, output_results, img_info, img_size, H=None):
        """Same signature as BYTETracker.update() plus optional H.

        Parameters
        ----------
        output_results, img_info, img_size :
            Same as vendored BYTETracker.update().
        H : np.ndarray or None
            2x3 affine (prev frame -> current frame) at FULL resolution,
            as produced by GMC.apply(). None disables compensation for
            this frame and reproduces exact baseline behavior.
        """
        # Inject multi_gmc at the multi_predict point of the vendored
        # update() by temporarily wrapping STrack.multi_predict. Restored
        # in `finally` so the global class is unchanged on return.
        original_multi_predict = STrack.multi_predict

        def predict_then_warp(stracks):
            original_multi_predict(stracks)
            multi_gmc(stracks, H)   # no-op when H is None or stracks is empty

        STrack.multi_predict = staticmethod(predict_then_warp)
        try:
            return super().update(output_results, img_info, img_size)
        finally:
            STrack.multi_predict = original_multi_predict
