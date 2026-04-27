"""
explainability/shap_explainer.py

SHAP-based explainability for FormScore.

Given a trained scorer model and a [60, 8] feature matrix for one rep,
produces:
  - shap_values  [60, 8]   per-frame per-feature contributions
  - fault_vector [8]       mean absolute SHAP per feature (importance ranking)
  - top_fault    str       name of the most important fault feature

Supported model types:
  "kernel"   → shap.KernelExplainer  (model-agnostic, slow ~5-30 s)
  "gradient" → shap.GradientExplainer (PyTorch only, fast ~50-200 ms)
  "tree"     → shap.TreeExplainer    (sklearn tree models)
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
    model_fn : callable | nn.Module
        "kernel" / "tree" : callable [N, 60, 8] → [N] numpy
        "gradient"        : torch.nn.Module (BiLSTM in eval mode)
    background : np.ndarray [B, 60, 8]
        Background dataset for SHAP baseline.
    model_type : str
        "kernel" | "gradient" | "tree"
    """

    def __init__(self, model_fn, background: np.ndarray, model_type: str = "kernel"):
        self.model_fn   = model_fn
        self.model_type = model_type

        if model_type == "kernel":
            bg_flat = background.reshape(len(background), -1)   # [B, 480]
            self.explainer = shap.KernelExplainer(self._flat_predict, bg_flat)

        elif model_type == "gradient":
            import torch
            if not isinstance(model_fn, torch.nn.Module):
                raise TypeError("model_fn must be a torch.nn.Module when model_type='gradient'")
            # SHAP GradientExplainer expects model outputs shaped [batch, outputs].
            # Our scorer returns [batch], so adapt to [batch, 1].
            class _GradientModelWrapper(torch.nn.Module):
                def __init__(self, base_model):
                    super().__init__()
                    self.base_model = base_model

                def forward(self, x):
                    y = self.base_model(x)
                    return y.unsqueeze(-1) if y.ndim == 1 else y

            self._gradient_model = _GradientModelWrapper(model_fn)
            bg_tensor = torch.FloatTensor(background)           # [B, 60, 8]
            self.explainer = shap.GradientExplainer(self._gradient_model, bg_tensor)

        elif model_type == "tree":
            self.explainer = shap.TreeExplainer(model_fn)

        else:
            raise ValueError(
                f"model_type must be 'kernel', 'gradient', or 'tree', got {model_type!r}"
            )

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
            shap_values   [60, 8]   per-frame per-feature SHAP values
            fault_vector  [8]       mean |SHAP| per feature
            top_fault     str       feature name with highest fault_vector
            top_fault_idx int       feature index of top fault
            frame_peak    int       frame index where top fault is worst
        """
        if rep.ndim == 2:
            rep = rep[np.newaxis, :]   # [1, 60, 8]

        if self.model_type == "kernel":
            rep_flat = rep.reshape(1, -1)   # [1, 480]
            raw = self.explainer.shap_values(rep_flat, nsamples=100)
            shap_matrix = np.array(raw).reshape(60, 8)

        elif self.model_type == "gradient":
            import torch
            rep_tensor = torch.FloatTensor(rep[np.newaxis]) if rep.shape[0] != 1 else torch.FloatTensor(rep)
            raw = self.explainer.shap_values(rep_tensor)
            # raw: list[[1,60,8]] or ndarray[1,60,8] depending on SHAP version
            shap_matrix = np.array(raw).squeeze(0).reshape(60, 8)

        elif self.model_type == "tree":
            raw = self.explainer.shap_values(rep.reshape(1, -1))
            shap_matrix = np.array(raw).reshape(60, 8)

        # Mean absolute SHAP per feature → importance ranking
        fault_vector = np.abs(shap_matrix).mean(axis=0)   # [8]

        top_fault_idx = int(np.argmax(fault_vector))
        top_fault     = FEATURE_NAMES[top_fault_idx]

        frame_peak = int(np.argmax(np.abs(shap_matrix[:, top_fault_idx])))

        return {
            "shap_values":   shap_matrix,        # [60, 8]
            "fault_vector":  fault_vector,        # [8]
            "top_fault":     top_fault,           # str
            "top_fault_idx": top_fault_idx,       # int
            "frame_peak":    frame_peak,          # int
        }

    def explain_batch(self, reps: np.ndarray) -> list[dict]:
        """Explain a batch of reps. reps: [N, 60, 8]"""
        return [self.explain(reps[i]) for i in range(len(reps))]


# ── Timing comparison ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import time
    import pathlib
    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    from modeling.load_model import get_model_and_predict_fn

    ckpt = pathlib.Path(__file__).parent.parent / "checkpoints" / "lstm_squat.pt"
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}. Run scripts/train_lstm_full.py first.")
        sys.exit(1)

    model, predict_fn = get_model_and_predict_fn(str(ckpt))

    rng        = np.random.default_rng(42)
    background = rng.random((10, 60, 8)).astype(np.float32)
    rep        = rng.random((60, 8)).astype(np.float32)

    # ── GradientExplainer ────────────────────────────────
    print("Building GradientExplainer (10 bg samples)…")
    t0       = time.perf_counter()
    grad_ex  = FormScoreExplainer(model, background, model_type="gradient")
    grad_ex.explain(rep)                             # warm-up / JIT
    t_warmup = (time.perf_counter() - t0) * 1000

    RUNS = 5
    t0 = time.perf_counter()
    for _ in range(RUNS):
        grad_ex.explain(rep)
    t_grad_ms = (time.perf_counter() - t0) * 1000 / RUNS
    print(f"  GradientExplainer — {t_grad_ms:.0f} ms/rep  (warm-up incl. init: {t_warmup:.0f} ms)")

    # ── KernelExplainer ──────────────────────────────────
    print("Building KernelExplainer (10 bg samples, nsamples=30)…")
    t0        = time.perf_counter()
    kernel_ex = FormScoreExplainer(predict_fn, background, model_type="kernel")
    raw = kernel_ex.explainer.shap_values(rep.reshape(1, -1), nsamples=30)
    t_kernel_ms = (time.perf_counter() - t0) * 1000
    print(f"  KernelExplainer   — {t_kernel_ms:.0f} ms/rep  (incl. init)")

    print(f"\nSpeedup: {t_kernel_ms / t_grad_ms:.1f}× faster with GradientExplainer")
