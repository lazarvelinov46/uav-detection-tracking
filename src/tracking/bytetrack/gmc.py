"""
src/tracking/bytetrack/gmc.py

Global Motion Compensation (GMC), a.k.a. Camera Motion Compensation (CMC).

Estimates the frame-to-frame camera ego-motion as a 2x3 affine
(similarity) transform using ORB (Oriented FAST and Rotated BRIEF)
features matched between consecutive frames, robustly fitted with RANSAC
(Random Sample Consensus).

The returned transform H maps a point in the PREVIOUS frame's image
coordinates to the CURRENT frame's image coordinates:

    [x']   [ a  b ] [x]   [tx]
    [y'] = [ c  d ] [y] + [ty]

It will be consumed by STrack.multi_gmc (a later task) to warp each
track's Kalman state into the new frame before IoU association, so a
camera pan no longer destroys the predicted-box / detection overlap.

Design choices for the Anti-UAV v4 thermal-IR regime (~11px boxes):
  - Detection boxes are masked out before feature extraction, so the
    estimate reflects BACKGROUND (camera) motion, not UAV motion -- the
    UAVs are the one thing moving independently of the camera.
  - Graceful identity fallback: on the first frame, or whenever too few
    inliers are found (e.g. low-texture sky), apply() returns the
    identity transform -> no compensation that frame, never a garbage
    warp that would corrupt every track.
  - Per-call diagnostics (keypoints / matches / inliers / fallback) are
    stored on .last_stats so the runner can log when an estimate is weak.

cv2 + numpy only -- this module does NOT import torch. Entry-point
scripts that (transitively) import byte_tracker must still `import torch`
first on Windows for DLL ordering, but importing GMC alone is safe.
"""

from __future__ import annotations

import numpy as np
import cv2


class GMC:
    """Per-sequence camera-motion estimator. Reuse one instance per
    sequence and call reset() between sequences (motion is intra-sequence)."""

    IDENTITY_2x3 = np.eye(2, 3, dtype=np.float32)

    def __init__(
        self,
        downscale: int = 2,
        n_features: int = 1000,
        ratio: float = 0.75,
        ransac_reproj_thresh: float = 3.0,
        min_matches: int = 10,
        min_inliers: int = 6,
        det_mask_margin: float = 0.0,
    ):
        """
        Parameters
        ----------
        downscale : int
            Factor to shrink frames before feature work (speed). The
            linear (rotation/scale) block is scale-invariant; only the
            translation is rescaled back to full resolution.
        n_features : int
            Max ORB keypoints per frame.
        ratio : float
            Lowe ratio-test threshold for accepting a KNN match.
        ransac_reproj_thresh : float
            RANSAC inlier threshold in pixels (downscaled space).
        min_matches : int
            Minimum good matches required to even attempt a fit.
        min_inliers : int
            Minimum RANSAC inliers required to trust the fit; below this
            we fall back to identity.
        det_mask_margin : float
            Fractional padding around each detection box when masking
            (0.0 = exact box; 0.5 = box grown 50% each side).
        """
        self.downscale = max(1, int(downscale))
        self.ratio = ratio
        self.ransac_reproj_thresh = ransac_reproj_thresh
        self.min_matches = min_matches
        self.min_inliers = min_inliers
        self.det_mask_margin = det_mask_margin

        self.detector = cv2.ORB_create(nfeatures=n_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

        self._prev_kpts = None
        self._prev_desc = None
        self.last_stats: dict = {}

    def reset(self) -> None:
        """Clear stored frame state. Call once per sequence."""
        self._prev_kpts = None
        self._prev_desc = None
        self.last_stats = {}

    # -- internals ----------------------------------------------------

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _build_mask(self, shape, detections) -> np.ndarray:
        """255 everywhere except inside (margin-expanded) detection boxes,
        which are zeroed so ORB ignores the UAVs. `detections` is full-res
        [x1,y1,x2,y2,...]; coordinates are scaled into downscaled space."""
        h, w = shape
        mask = np.full((h, w), 255, dtype=np.uint8)
        if detections is None or len(detections) == 0:
            return mask
        s = self.downscale
        m = self.det_mask_margin
        for x1, y1, x2, y2 in np.asarray(detections, dtype=np.float32)[:, :4]:
            bw, bh = (x2 - x1), (y2 - y1)
            xi1 = max(0, int(np.floor((x1 - m * bw) / s)))
            yi1 = max(0, int(np.floor((y1 - m * bh) / s)))
            xi2 = min(w, int(np.ceil((x2 + m * bw) / s)))
            yi2 = min(h, int(np.ceil((y2 + m * bh) / s)))
            if xi2 > xi1 and yi2 > yi1:
                mask[yi1:yi2, xi1:xi2] = 0
        return mask

    # -- public API ---------------------------------------------------

    def apply(self, frame: np.ndarray, detections=None) -> np.ndarray:
        """
        Estimate the affine transform from the previous frame to `frame`.

        Parameters
        ----------
        frame : np.ndarray
            HxWx3 (BGR) or HxW grayscale current frame, FULL resolution.
        detections : np.ndarray or None
            (N, 4+) array of [x1, y1, x2, y2, ...] in FULL-res pixels,
            masked out before feature extraction. None -> no mask.

        Returns
        -------
        np.ndarray
            2x3 float32 affine mapping previous-frame -> current-frame in
            FULL-resolution pixels. Identity on the first frame or when
            the estimate is not trustworthy.
        """
        gray = self._to_gray(frame)
        if self.downscale > 1:
            gray = cv2.resize(
                gray,
                (gray.shape[1] // self.downscale, gray.shape[0] // self.downscale),
                interpolation=cv2.INTER_AREA,
            )

        mask = self._build_mask(gray.shape, detections)
        kpts, desc = self.detector.detectAndCompute(gray, mask)

        stats = {
            "n_keypoints": 0 if kpts is None else len(kpts),
            "n_matches": 0,
            "n_inliers": 0,
            "fallback": True,
        }

        # First frame, or nothing detected to match with/against.
        if self._prev_desc is None or desc is None or len(kpts) == 0:
            self._prev_kpts, self._prev_desc = kpts, desc
            self.last_stats = stats
            return self.IDENTITY_2x3.copy()

        # KNN match (k=2) + Lowe ratio test to keep only confident pairs.
        good = []
        for pair in self.matcher.knnMatch(self._prev_desc, desc, k=2):
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance:
                good.append(m)
        stats["n_matches"] = len(good)

        H = self.IDENTITY_2x3.copy()
        if len(good) >= self.min_matches:
            prev_pts = np.float32([self._prev_kpts[m.queryIdx].pt for m in good])
            curr_pts = np.float32([kpts[m.trainIdx].pt for m in good])
            H_est, inliers = cv2.estimateAffinePartial2D(
                prev_pts, curr_pts,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_reproj_thresh,
            )
            n_in = 0 if inliers is None else int(inliers.sum())
            stats["n_inliers"] = n_in
            if H_est is not None and n_in >= self.min_inliers:
                H = H_est.astype(np.float32)
                # Linear block is downscale-invariant; rescale translation.
                if self.downscale > 1:
                    H[0, 2] *= self.downscale
                    H[1, 2] *= self.downscale
                stats["fallback"] = False

        self._prev_kpts, self._prev_desc = kpts, desc
        self.last_stats = stats
        return H


def multi_gmc(stracks, H) -> None:
    """
    Warp a list of tracks' Kalman states by the camera-motion affine H,
    IN PLACE. Call this right after STrack.multi_predict and before IoU
    association, so each predicted box is moved into the new frame's
    coordinate system (cancelling the camera pan) before matching.

    The vendored ByteTrack Kalman state is **xyah**:

        mean = [cx, cy, a, h, vcx, vcy, va, vh]

    where (cx, cy) is the box CENTER, `a` is aspect ratio (w/h), `h` is
    height, followed by their velocities. This differs from BoT-SORT's
    xywh state, so we cannot copy BoT-SORT's kron(I4, R) verbatim -- the
    (a, h) pair is not a spatial vector and must not be rotated together.

    Decomposition of the affine H = [A | t], with A = H[:2,:2], t = H[:2,2]
    and isotropic scale s = sqrt(|det A|):
      - center (cx, cy)   : full affine        -> A @ center + t
      - aspect a          : invariant          -> unchanged
      - height h          : isotropic scale     -> s * h
      - velocity (vcx,vcy): linear part only    -> A @ velocity   (no t)
      - aspect rate va    : invariant          -> unchanged
      - height rate vh    : isotropic scale     -> s * vh
    (For a similarity transform -- what estimateAffinePartial2D returns --
    A = s*R exactly, so aspect is *exactly* preserved and s is the true
    scale. For a general affine, s is the geometric-mean scale, a sound
    isotropic approximation for small inter-frame motion.)

    Covariance propagates as  cov' = M @ cov @ M.T  with the same block
    matrix M; the additive translation b does not affect covariance.

    Parameters
    ----------
    stracks : list
        Objects with `.mean` (8,) and `.covariance` (8, 8) attributes
        (e.g. vendored STrack instances).
    H : np.ndarray or None
        2x3 affine (previous -> current frame), full resolution. None or
        identity is a no-op.
    """
    if H is None or len(stracks) == 0:
        return

    A = np.asarray(H, dtype=np.float64)[:2, :2]
    t = np.asarray(H, dtype=np.float64)[:2, 2]
    s = float(np.sqrt(abs(np.linalg.det(A))))

    M = np.eye(8, dtype=np.float64)
    M[0:2, 0:2] = A      # center
    M[3, 3] = s          # height
    M[4:6, 4:6] = A      # center velocity
    M[7, 7] = s          # height velocity
    # M[2,2] and M[6,6] stay 1 -> aspect and aspect-rate invariant.

    b = np.zeros(8, dtype=np.float64)
    b[0], b[1] = t[0], t[1]

    for st in stracks:
        if getattr(st, "mean", None) is None:
            continue
        st.mean = M @ np.asarray(st.mean, dtype=np.float64) + b
        st.covariance = M @ np.asarray(st.covariance, dtype=np.float64) @ M.T
