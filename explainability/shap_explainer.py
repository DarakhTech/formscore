"""
explainability/shap_explainer.py

SHAP-based explainability for FormScore.

Given a trained scorer model and a [60, 8] feature matrix for one rep,
produces:
  - shap_values  [60, 8]   per-frame per-feature contributions
  - fault_vector [8]       mean absolute SHAP per feature (importance ranking)
  - top_fault    str       name of the most important fault feature

The explainer is model-agnostic — works with RF, CNN, BiLSTM, or any
sklearn/PyTorch model wrapped with a predict function.
"""

import numpy as np
import shap

# ── Feature index → name mapping ─────────────────────────
FEATURE_NAMES = [
    "knee_angle_left",      # 0
    "knee_angle_right",     # 1
    "hip_angle_left",       # 2
    "hip_angle_right",      # 3
    "spine_tilt",           # 4
    "knee_velocity_left",   # 5
    "knee_velocity_right",  # 6
    "knee_symmetry",        # 7
]


class FormScoreExplainer:
    """
    Wraps a trained scorer and computes SHAP explanations over [60, 8] reps.

    Parameters
    ----------
    model_fn : callable
        Function that takes [N, 60, 8] numpy array and returns [N] scores.
        Works with RF (after flattening) or neural nets.
    background : np.ndarray [B, 60, 8]
        Background dataset for SHAP baseline — typically 50-100 random
        training reps. SHAP compares each rep against this background.
    model_type : str
        "tree"   → uses shap.TreeExplainer  (RF)
        "kernel" → uses shap.KernelExplainer (CNN, BiLSTM)
    """

    def __init__(self, model_fn, background: np.ndarray, model_type: str = "kernel"):
        self.model_fn   = model_fn
        self.model_type = model_type

        if model_type == "kernel":
            # KernelExplainer needs 2D input → flatten [B, 60, 8] → [B, 480]
            bg_flat = background.reshape(len(background), -1)   # [B, 480]
            self.explainer = shap.KernelExplainer(
                self._flat_predict,
                bg_flat,
            )
        elif model_type == "tree":
            # TreeExplainer works directly on the model object
            # model_fn here should be the actual sklearn model, not a wrapper
            self.explainer = shap.TreeExplainer(model_fn)
        else:
            raise ValueError(f"model_type must be 'tree' or 'kernel', got {model_type}")

    def _flat_predict(self, X_flat: np.ndarray) -> np.ndarray:
        """Unflatten [N, 480] → [N, 60, 8] then call model_fn."""
        X = X_flat.reshape(len(X_flat), 60, 8)
        return self.model_fn(X)

    def explain(self, rep: np.ndarray) -> dict:
        """
        Compute SHAP explanation for a single rep.

        Parameters
        ----------
        rep : np.ndarray [60, 8]

        Returns
        -------
        dict with keys:
            shap_values  [60, 8]   per-frame per-feature SHAP values
            fault_vector [8]       mean |SHAP| per feature
            top_fault    str       feature name with highest fault_vector
            top_fault_idx int      feature index of top fault
            frame_peak   int       frame index where top fault is worst
        """
        if rep.ndim == 2:
            rep = rep[np.newaxis, :]   # [1, 60, 8]

        if self.model_type == "kernel":
            rep_flat = rep.reshape(1, -1)   # [1, 480]
            raw = self.explainer.shap_values(rep_flat, nsamples=100)
            # raw shape: [1, 480] → reshape to [60, 8]
            shap_matrix = np.array(raw).reshape(60, 8)

        elif self.model_type == "tree":
            raw = self.explainer.shap_values(rep.reshape(1, -1))
            shap_matrix = np.array(raw).reshape(60, 8)

        # Mean absolute SHAP per feature → importance ranking
        fault_vector = np.abs(shap_matrix).mean(axis=0)   # [8]

        top_fault_idx = int(np.argmax(fault_vector))
        top_fault     = FEATURE_NAMES[top_fault_idx]

        # Frame where the top fault is most extreme
        frame_peak = int(np.argmax(np.abs(shap_matrix[:, top_fault_idx])))

        return {
            "shap_values":   shap_matrix,        # [60, 8]
            "fault_vector":  fault_vector,        # [8]
            "top_fault":     top_fault,           # str
            "top_fault_idx": top_fault_idx,       # int
            "frame_peak":    frame_peak,          # int
        }

    def explain_batch(self, reps: np.ndarray) -> list[dict]:
        """
        Explain a batch of reps.

        Parameters
        ----------
        reps : np.ndarray [N, 60, 8]

        Returns
        -------
        list of N explanation dicts
        """
        return [self.explain(reps[i]) for i in range(len(reps))]