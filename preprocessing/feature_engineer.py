# preprocessing/feature_engineer.py
"""
Biomechanical feature engineering for FormScore.

Consumes normalized [T, 33, 4] landmarks and produces an [T, 8] feature
matrix:
  0-4  five joint angles  (exercise-specific; see configs/exercises.py)
  5    angle_velocity_left   (d/dt of col 0)
  6    angle_velocity_right  (d/dt of col 1)
  7    angle_symmetry        (|col 0 - col 1|)
"""

import numpy as np

from configs.exercises import EXERCISE_CONFIGS

# ── MediaPipe landmark indices ────────────────────────────
_L_SHOULDER = 11
_R_SHOULDER = 12
_L_HIP      = 23
_R_HIP      = 24
_L_KNEE     = 25
_R_KNEE     = 26
_L_ANKLE    = 27
_R_ANKLE    = 28

N_FEATURES = 8


def _angle_vec(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Vectorized angle at joint b formed by a-b-c across T frames.

    Parameters
    ----------
    a, b, c : np.ndarray [T, 3]

    Returns
    -------
    np.ndarray [T]  angles in degrees
    """
    ba = a - b                                          # [T, 3]
    bc = c - b                                          # [T, 3]

    dot    = np.einsum("ti,ti->t", ba, bc)              # [T]
    norm_a = np.linalg.norm(ba, axis=1)                 # [T]
    norm_c = np.linalg.norm(bc, axis=1)                 # [T]

    cosine = dot / (norm_a * norm_c + 1e-8)             # [T]
    cosine = np.clip(cosine, -1.0, 1.0)                 # domain guard
    return np.degrees(np.arccos(cosine))                # [T]


def compute_angles_for_exercise(landmarks: np.ndarray, exercise: str = "squat") -> np.ndarray:
    """
    Compute 5 joint angles for a given exercise using its angle_joints config.

    Parameters
    ----------
    landmarks : np.ndarray [T, 33, 4]
    exercise  : str  key in EXERCISE_CONFIGS

    Returns
    -------
    np.ndarray [T, 5]
        Columns 0-3: four explicit triplet angles.
        Column  4:   mean of cols 0 and 1 (composite signal per exercise).
    """
    cfg = EXERCISE_CONFIGS[exercise]
    xyz = landmarks[:, :, :3]   # [T, 33, 3]

    triplets = cfg["angle_joints"]   # list of 4 (a, b, c) tuples
    a0 = _angle_vec(xyz[:, triplets[0][0]], xyz[:, triplets[0][1]], xyz[:, triplets[0][2]])
    a1 = _angle_vec(xyz[:, triplets[1][0]], xyz[:, triplets[1][1]], xyz[:, triplets[1][2]])
    a2 = _angle_vec(xyz[:, triplets[2][0]], xyz[:, triplets[2][1]], xyz[:, triplets[2][2]])
    a3 = _angle_vec(xyz[:, triplets[3][0]], xyz[:, triplets[3][1]], xyz[:, triplets[3][2]])
    a4 = (a0 + a1) / 2.0   # composite: spine_tilt / body_alignment / overhead_ext

    return np.stack([a0, a1, a2, a3, a4], axis=1)  # [T, 5]


def compute_angles(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute 5 joint angles from normalized landmarks.

    Parameters
    ----------
    landmarks : np.ndarray [T, 33, 4]

    Returns
    -------
    np.ndarray [T, 5]
        Columns: knee_L, knee_R, hip_L, hip_R, spine_tilt
    """
    xyz = landmarks[:, :, :3]   # [T, 33, 3]

    knee_l  = _angle_vec(xyz[:, _L_HIP],      xyz[:, _L_KNEE],  xyz[:, _L_ANKLE])
    knee_r  = _angle_vec(xyz[:, _R_HIP],      xyz[:, _R_KNEE],  xyz[:, _R_ANKLE])
    hip_l   = _angle_vec(xyz[:, _L_SHOULDER], xyz[:, _L_HIP],   xyz[:, _L_KNEE])
    hip_r   = _angle_vec(xyz[:, _R_SHOULDER], xyz[:, _R_HIP],   xyz[:, _R_KNEE])
    spine   = (hip_l + hip_r) / 2.0

    return np.stack([knee_l, knee_r, hip_l, hip_r, spine], axis=1)  # [T, 5]

def compute_velocity(angles: np.ndarray) -> np.ndarray:
    """
    Angular velocity for knee_L and knee_R via np.diff, padded to T.

    Parameters
    ----------
    angles : np.ndarray [T, 5]  output of compute_angles()

    Returns
    -------
    np.ndarray [T, 2]  columns: knee_vel_left, knee_vel_right
    """
    knee_l = angles[:, 0]   # [T]
    knee_r = angles[:, 1]   # [T]

    vel_l = np.diff(knee_l, prepend=knee_l[0])   # [T]
    vel_r = np.diff(knee_r, prepend=knee_r[0])   # [T]

    return np.stack([vel_l, vel_r], axis=1)       # [T, 2]


def compute_symmetry_depth(angles: np.ndarray) -> np.ndarray:
    """
    Symmetry and depth ratio features.

    Parameters
    ----------
    angles : np.ndarray [T, 5]  output of compute_angles()

    Returns
    -------
    np.ndarray [T, 1]  columns: knee_symmetry
    """
    knee_symmetry = np.abs(angles[:, 0] - angles[:, 1])  # |knee_L - knee_R|
    return knee_symmetry[:, np.newaxis]                   # [T, 1]


def build_feature_matrix(landmarks: np.ndarray, exercise: str = "squat") -> np.ndarray:
    """
    Full [T, 8] feature matrix from normalized [T, 33, 4] landmarks.

    Feature columns:
      0-4  five joint angles  (exercise-specific)
      5    angle_velocity_left   (d/dt of col 0)
      6    angle_velocity_right  (d/dt of col 1)
      7    angle_symmetry        (|col 0 - col 1|)

    Parameters
    ----------
    landmarks : np.ndarray [T, 33, 4]  normalized landmarks
    exercise  : str  key in EXERCISE_CONFIGS (default "squat")

    Returns
    -------
    np.ndarray [T, 8]
    """
    angles   = compute_angles_for_exercise(landmarks, exercise)  # [T, 5]
    velocity = compute_velocity(angles)                          # [T, 2]
    symmetry = compute_symmetry_depth(angles)                    # [T, 1]

    return np.concatenate([angles, velocity, symmetry], axis=1)  # [T, 8]

def resample_to_60(feature_matrix: np.ndarray, n_frames: int = 60) -> np.ndarray:
    """
    Resample a variable-length feature matrix to exactly n_frames via
    linear interpolation.

    Parameters
    ----------
    feature_matrix : np.ndarray [T, 8]
    n_frames       : int, target frame count (default 60)

    Returns
    -------
    np.ndarray [60, 8]
    """
    T = feature_matrix.shape[0]
    if T == n_frames:
        return feature_matrix.copy()

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, n_frames)

    resampled = np.stack([
        np.interp(x_new, x_old, feature_matrix[:, i])
        for i in range(feature_matrix.shape[1])
    ], axis=1)

    return resampled   # [60, 8]