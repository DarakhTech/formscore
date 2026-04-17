# preprocessing/normalizer.py
"""
Coordinate normalization for FormScore.

Translates and scales each frame's 33 landmarks so that:
  - origin = hip midpoint (mean of landmarks 23 and 24)
  - scale  = torso height (shoulder midpoint → hip midpoint distance)

Input : np.ndarray [T, 33, 4]  (x, y, z, visibility)
Output: np.ndarray [T, 33, 4]  (normalized x, y, z; visibility unchanged)
"""

import numpy as np

# MediaPipe landmark indices
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24


def normalize(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize a landmark sequence to be translation- and scale-invariant.

    Parameters
    ----------
    landmarks : np.ndarray, shape [T, 33, 4]
        Raw MediaPipe landmarks (x, y, z, visibility) in image-normalized
        coordinates [0, 1].

    Returns
    -------
    np.ndarray, shape [T, 33, 4]
        Landmarks translated to hip-midpoint origin and scaled by torso
        height. Visibility channel is preserved unchanged.
    """
    if landmarks.ndim != 3 or landmarks.shape[1] != 33 or landmarks.shape[2] != 4:
        raise ValueError(f"Expected [T, 33, 4], got {landmarks.shape}")

    xyz = landmarks[:, :, :3].copy()       # [T, 33, 3]
    vis = landmarks[:, :, 3:4]             # [T, 33, 1]  — keep for output

    # ── Origin: hip midpoint ──────────────────────────────────────────
    hip_mid = (xyz[:, _LEFT_HIP, :] + xyz[:, _RIGHT_HIP, :]) / 2.0   # [T, 3]

    xyz -= hip_mid[:, np.newaxis, :]       # broadcast subtract over 33 landmarks

    # ── Scale: torso height ───────────────────────────────────────────
    shoulder_mid = (xyz[:, _LEFT_SHOULDER, :] + xyz[:, _RIGHT_SHOULDER, :]) / 2.0  # [T, 3]
    # hip_mid is now at origin, so torso vector is just shoulder_mid
    torso_height = np.linalg.norm(shoulder_mid, axis=1, keepdims=True)  # [T, 1]

    # Guard against degenerate frames (e.g. partial detections)
    torso_height = np.where(torso_height < 1e-6, 1.0, torso_height)    # [T, 1]

    xyz /= torso_height[:, np.newaxis, :]  # broadcast divide

    return np.concatenate([xyz, vis], axis=2)  # [T, 33, 4]