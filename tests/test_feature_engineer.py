# tests/test_feature_engineer.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from preprocessing.feature_engineer import (
    compute_angles, compute_velocity,
    compute_symmetry_depth, build_feature_matrix, N_FEATURES, resample_to_60
)


_T = 20


def _make_landmarks(t=_T, seed=7):
    rng = np.random.default_rng(seed)
    xyz = rng.random((t, 33, 3)).astype(np.float64)
    vis = np.ones((t, 33, 1), dtype=np.float64)
    return np.concatenate([xyz, vis], axis=2)


# ── Test 1: output shape ──────────────────────────────────
def test_angle_output_shape():
    angles = compute_angles(_make_landmarks())
    assert angles.shape == (_T, 5), f"Expected ({_T}, 5), got {angles.shape}"


# ── Test 2: all angles in [0, 180] ───────────────────────
def test_angles_in_valid_range():
    angles = compute_angles(_make_landmarks())
    assert angles.min() >= 0.0,   f"Angle below 0: {angles.min()}"
    assert angles.max() <= 180.0, f"Angle above 180: {angles.max()}"


# ── Test 3: known 90-degree squat frame ──────────────────
def test_known_90_degree_knee():
    """
    Construct a synthetic frame where left knee is exactly 90°:
      hip   = (0, 1, 0)   — directly above knee
      knee  = (0, 0, 0)   — origin
      ankle = (1, 0, 0)   — directly to the side

    hip→knee vector  = (0,-1, 0)
    knee→ankle vector= (1, 0, 0)
    dot product = 0  →  angle = 90°
    """
    lm = np.zeros((1, 33, 4))
    # Place only the 3 joints that matter for left knee angle
    lm[0, 23, :3] = [0.0, 1.0, 0.0]   # LEFT_HIP
    lm[0, 25, :3] = [0.0, 0.0, 0.0]   # LEFT_KNEE
    lm[0, 27, :3] = [1.0, 0.0, 0.0]   # LEFT_ANKLE

    angles = compute_angles(lm)
    knee_l = angles[0, 0]
    assert abs(knee_l - 90.0) < 1e-4, f"Expected 90.0°, got {knee_l:.4f}°"

# ── Test 4: velocity shape and first frame is zero ────────
def test_velocity_shape_and_first_frame():
    angles = compute_angles(_make_landmarks())
    vel = compute_velocity(angles)
    assert vel.shape == (_T, 2), f"Expected ({_T}, 2), got {vel.shape}"
    # prepend duplicates frame 0, so diff at index 0 = 0
    assert vel[0, 0] == 0.0 and vel[0, 1] == 0.0, "First frame velocity should be 0"


# ── Test 5: symmetry is always non-negative ───────────────
def test_symmetry_non_negative():
    angles = compute_angles(_make_landmarks())
    sym = compute_symmetry_depth(angles)
    assert sym.shape == (_T, 1)
    assert (sym >= 0).all(), "Symmetry values must be non-negative"


# ── Test 6: full feature matrix shape [T, 8] ─────────────
def test_feature_matrix_shape():
    lm = _make_landmarks()
    feat = build_feature_matrix(lm)
    assert feat.shape == (_T, N_FEATURES), \
        f"Expected ({_T}, {N_FEATURES}), got {feat.shape}"
        
# ── Test 7: short rep (20 frames) → [60, 8] ──────────────
def test_resample_short_rep():
    feat = build_feature_matrix(_make_landmarks(t=20))
    out  = resample_to_60(feat)
    assert out.shape == (60, 8), f"Expected (60, 8), got {out.shape}"


# ── Test 8: long rep (90 frames) → [60, 8] ───────────────
def test_resample_long_rep():
    feat = build_feature_matrix(_make_landmarks(t=90))
    out  = resample_to_60(feat)
    assert out.shape == (60, 8), f"Expected (60, 8), got {out.shape}"


# ── Test 9: exact 60 frames passes through unchanged ─────
def test_resample_exact_passthrough():
    feat = build_feature_matrix(_make_landmarks(t=60))
    out  = resample_to_60(feat)
    assert out.shape == (60, 8)
    assert np.allclose(out, feat), "60-frame input should pass through unchanged"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])