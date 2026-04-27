"""
preprocessing/rule_scorer.py

Rule-based form scoring for FormScore, mirroring Good-GYM's approach.
Operates on the [T, 8] feature matrix produced by build_feature_matrix().

Feature indices (all angles in degrees):
  0  knee_angle_left   (or elbow_L for pushup/shoulder_press)
  1  knee_angle_right  (or elbow_R for pushup/shoulder_press)
  2  hip_angle_left    (or shoulder_L)
  3  hip_angle_right   (or shoulder_R)
  4  spine_tilt        (composite: mean of first two angles, exercise-specific)
  5  knee_velocity_left
  6  knee_velocity_right
  7  knee_symmetry     (|col 0 - col 1|)

Each exercise has 4 rules worth 0.25 each → total in [0, 1].
"""

import numpy as np


# ── Credit helpers ──────────────────────────────────────────────────────

def _credit_below(value: float, full_thresh: float, half_thresh: float) -> float:
    """Full credit if value < full_thresh, half if < half_thresh, else 0."""
    if value < full_thresh:
        return 1.0
    if value < half_thresh:
        return 0.5
    return 0.0


def _credit_above(value: float, full_thresh: float, half_thresh: float) -> float:
    """Full credit if value > full_thresh, half if > half_thresh, else 0."""
    if value > full_thresh:
        return 1.0
    if value > half_thresh:
        return 0.5
    return 0.0


# ── Per-exercise rule functions ─────────────────────────────────────────

def _squat_rules(feat: np.ndarray) -> float:
    """4 squat rules × 0.25 each."""
    knee_l = feat[:, 0]
    knee_r = feat[:, 1]
    spine  = feat[:, 4]
    sym    = feat[:, 7]

    depth   = _credit_below(min(knee_l.min(), knee_r.min()),  50.0,  70.0)
    spine_c = _credit_above(spine.mean(),                     80.0,  60.0)
    sym_c   = _credit_below(sym.mean(),                       10.0,  20.0)
    lockout = _credit_above(max(knee_l.max(), knee_r.max()), 110.0,  95.0)

    return (depth + spine_c + sym_c + lockout) * 0.25


def _pushup_rules(feat: np.ndarray) -> float:
    """4 push-up rules × 0.25 each. Feature 0 = elbow_L slot."""
    elbow_l = feat[:, 0]
    spine   = feat[:, 4]
    sym     = feat[:, 7]

    depth   = _credit_below(elbow_l.min(),  50.0,  70.0)
    body    = _credit_above(spine.mean(),   80.0,  65.0)
    sym_c   = _credit_below(sym.mean(),     10.0,  20.0)
    lockout = _credit_above(elbow_l.max(), 110.0,  95.0)

    return (depth + body + sym_c + lockout) * 0.25


def _shoulder_press_rules(feat: np.ndarray) -> float:
    """4 shoulder-press rules × 0.25 each. Features 0,1 = elbow_L, elbow_R."""
    elbow_l = feat[:, 0]
    elbow_r = feat[:, 1]
    spine   = feat[:, 4]
    sym     = feat[:, 7]

    ext     = _credit_above(max(elbow_l.max(), elbow_r.max()), 110.0,  95.0)
    start   = _credit_below(min(elbow_l.min(), elbow_r.min()),  50.0,  65.0)
    sym_c   = _credit_below(sym.mean(),                          10.0,  20.0)
    spine_c = _credit_above(spine.mean(),                        80.0,  65.0)

    return (ext + start + sym_c + spine_c) * 0.25


_RULE_FNS = {
    "squat":          _squat_rules,
    "pushup":         _pushup_rules,
    "shoulder_press": _shoulder_press_rules,
}


# ── Public API ──────────────────────────────────────────────────────────

def rule_score(feature_matrix: np.ndarray, exercise: str = "squat") -> float:
    """
    Rule-based form score from a [T, 8] feature matrix.

    Parameters
    ----------
    feature_matrix : np.ndarray [T, 8]  — typically already resampled to [60, 8]
    exercise       : str  — "squat", "pushup", or "shoulder_press"

    Returns
    -------
    float in [0, 1]
    """
    feat = np.asarray(feature_matrix, dtype=np.float32)
    fn   = _RULE_FNS.get(exercise)
    if fn is None:
        raise ValueError(
            f"Unknown exercise '{exercise}'. Valid: {list(_RULE_FNS)}"
        )
    return float(fn(feat))


def hybrid_score(
    bilstm_score: float,
    feature_matrix: np.ndarray,
    exercise: str = "squat",
    bilstm_weight: float = 0.8,
) -> dict:
    """
    Weighted combination of BiLSTM model score and rule-based score.

    Parameters
    ----------
    bilstm_score   : float       — scalar output of the BiLSTM model
    feature_matrix : np.ndarray  — [T, 8] feature matrix for the rep
    exercise       : str
    bilstm_weight  : float       — BiLSTM fraction (rules get 1 - bilstm_weight)

    Returns
    -------
    dict with keys:
        hybrid         : float — weighted combination, clipped to [0, 1]
        bilstm         : float — raw BiLSTM score
        rules          : float — raw rule score
        delta          : float — |bilstm - rules|
        agreement      : str   — "high" (<0.10) | "medium" (<0.20) | "low" (>=0.20)
        interpretation : str   — human-readable explanation of the agreement level
    """
    rule_s = rule_score(feature_matrix, exercise)
    hybrid = float(np.clip(
        bilstm_weight * bilstm_score + (1.0 - bilstm_weight) * rule_s,
        0.0, 1.0,
    ))
    delta = abs(bilstm_score - rule_s)

    if delta < 0.1:
        agreement      = "high"
        interpretation = "Both scorers agree — reliable score"
    elif delta < 0.2:
        agreement      = "medium"
        interpretation = "Minor disagreement — score is approximate"
    elif bilstm_score > rule_s:
        agreement      = "low"
        interpretation = "Temporal model more optimistic — movement quality good but check form basics"
    else:
        agreement      = "low"
        interpretation = "Rules passed but temporal model flagged movement quality issues"

    return {
        "hybrid":         round(hybrid, 3),
        "bilstm":         round(float(bilstm_score), 3),
        "rules":          round(float(rule_s), 3),
        "delta":          round(float(delta), 3),
        "agreement":      agreement,
        "interpretation": interpretation,
    }


# ── Self-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

    rng = np.random.default_rng(0)

    print("=== rule_scorer self-test ===\n")

    # ── Random feature matrices ───────────────────────────────────
    print("Random feature matrices:")
    for ex in ("squat", "pushup", "shoulder_press"):
        feat = rng.uniform(60.0, 170.0, size=(60, 8)).astype(np.float32)
        feat[:, 7] = rng.uniform(0.0, 30.0, 60)   # symmetry stays small
        bilstm = float(rng.uniform(0.4, 0.9))
        r_s    = rule_score(feat, exercise=ex)
        h      = hybrid_score(bilstm, feat, exercise=ex)
        print(
            f"  {ex:<16}  rule={r_s:.3f}  bilstm={bilstm:.3f}  "
            f"hybrid={h['hybrid']:.3f}  delta={h['delta']:.3f}  "
            f"agreement={h['agreement']}"
        )

    # ── Perfect squat simulation (calibrated to real normalized ranges) ──
    # Observed from squats.mp4: knee min≈27°, max≈120°, spine mean≈90°, sym≈2°
    print("\nPerfect squat simulation:")
    t             = np.linspace(0, 1, 60)
    knee_curve    = 120.0 - (120.0 - 27.0) * np.abs(np.sin(np.pi * t))

    feat_perfect  = np.zeros((60, 8), dtype=np.float32)
    feat_perfect[:, 0] = knee_curve   # knee_L: 27° at bottom, 120° at top
    feat_perfect[:, 1] = knee_curve   # knee_R: identical (perfect symmetry)
    feat_perfect[:, 4] = 90.0         # spine_tilt: upright in normalized coords
    feat_perfect[:, 7] = 2.0          # knee_symmetry: near-zero

    score_perfect = rule_score(feat_perfect, exercise="squat")
    h_perfect     = hybrid_score(0.95, feat_perfect, exercise="squat")

    print(f"  rule_score  : {score_perfect:.3f}")
    print(f"  hybrid_score: {h_perfect}")

    assert score_perfect > 0.9, f"Expected > 0.9, got {score_perfect}"
    print("\nPASS — perfect squat rule_score > 0.9")
