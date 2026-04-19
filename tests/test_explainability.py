# tests/test_explainability.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from explainability.shap_explainer import FormScoreExplainer, FEATURE_NAMES
from explainability.feedback_lookup import get_feedback, format_feedback


def dummy_model(X: np.ndarray) -> np.ndarray:
    """
    Dummy scorer — mimics a trained model.
    Penalizes high knee_symmetry (feature 7) and high spine_tilt (feature 4).
    """
    symmetry   = X[:, :, 7].mean(axis=1)   # [N]
    spine      = X[:, :, 4].mean(axis=1)   # [N]
    score      = 1.0 - 0.4 * symmetry - 0.3 * (spine / 180.0)
    return np.clip(score, 0.0, 1.0)


def test_explainer_output_shape():
    rng        = np.random.default_rng(42)
    background = rng.random((50, 60, 8)).astype(np.float32)
    rep        = rng.random((60, 8)).astype(np.float32)

    explainer = FormScoreExplainer(dummy_model, background, model_type="kernel")
    result    = explainer.explain(rep)

    assert result["shap_values"].shape == (60, 8),  \
        f"Expected (60,8), got {result['shap_values'].shape}"
    assert result["fault_vector"].shape == (8,),    \
        f"Expected (8,), got {result['fault_vector'].shape}"
    assert result["top_fault"] in FEATURE_NAMES,    \
        f"Unknown fault: {result['top_fault']}"
    assert 0 <= result["frame_peak"] < 60,          \
        f"frame_peak out of range: {result['frame_peak']}"
    print("  explainer output shape  : PASS")


def test_feedback_generation():
    rng        = np.random.default_rng(7)
    background = rng.random((50, 60, 8)).astype(np.float32)
    rep        = rng.random((60, 8)).astype(np.float32)

    explainer  = FormScoreExplainer(dummy_model, background, model_type="kernel")
    result     = explainer.explain(rep)
    score      = float(dummy_model(rep[np.newaxis])[0])
    feedback   = get_feedback(result, form_score=score)

    assert "overall"   in feedback
    assert "cues"      in feedback
    assert "top_fault" in feedback
    assert isinstance(feedback["cues"], list)
    print("  feedback generation     : PASS")
    print()
    print(format_feedback(feedback))


if __name__ == "__main__":
    print("\n=== test_explainability ===")
    test_explainer_output_shape()
    test_feedback_generation()
    print("\n=== PASS ===\n")