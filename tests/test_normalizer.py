# tests/test_normalizer.py
"""
3 unit tests for preprocessing/normalizer.py  (s1-a3 requirement)
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pytest
from preprocessing.normalizer import normalize

_T, _J = 30, 33  # frames, joints


def _make_landmarks(t=_T, seed=42):
    """Random [T, 33, 4] array with visibility=1."""
    rng = np.random.default_rng(seed)
    xyz = rng.random((t, _J, 3)).astype(np.float32)
    vis = np.ones((t, _J, 1), dtype=np.float32)
    return np.concatenate([xyz, vis], axis=2)


# ── Test 1: output shape is preserved ────────────────────────────────
def test_output_shape():
    out = normalize(_make_landmarks())
    assert out.shape == (_T, _J, 4), f"Expected ({_T}, {_J}, 4), got {out.shape}"


# ── Test 2: hip midpoint is at origin after normalization ─────────────
def test_hip_at_origin():
    out = normalize(_make_landmarks())
    LEFT_HIP, RIGHT_HIP = 23, 24
    hip_mid = (out[:, LEFT_HIP, :3] + out[:, RIGHT_HIP, :3]) / 2.0  # [T, 3]
    assert np.allclose(hip_mid, 0.0, atol=1e-5), \
        f"Hip midpoint not at origin. Max deviation: {np.abs(hip_mid).max()}"


# ── Test 3: torso height = 1.0 after normalization ────────────────────
def test_torso_height_is_unit():
    out = normalize(_make_landmarks())
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    shoulder_mid = (out[:, LEFT_SHOULDER, :3] + out[:, RIGHT_SHOULDER, :3]) / 2.0
    torso_heights = np.linalg.norm(shoulder_mid, axis=1)  # [T]
    assert np.allclose(torso_heights, 1.0, atol=1e-5), \
        f"Torso height not normalized to 1. Values: {torso_heights[:5]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])