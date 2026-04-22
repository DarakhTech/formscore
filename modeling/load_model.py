"""
modeling/load_model.py

Load the best full-dataset BiLSTM checkpoint and expose a predict_fn.

Usage
-----
    from modeling.load_model import predict_fn

    preds = predict_fn(X)   # X: [N, 60, 8] numpy  →  preds: [N] numpy
"""

import pathlib
import numpy as np
import torch

from modeling.lstm_scorer import LSTMScorer

_CHECKPOINT = pathlib.Path("checkpoints/lstm_best_full.pt")
_device = "cuda" if torch.cuda.is_available() else "cpu"

_model: LSTMScorer | None = None


def _load() -> LSTMScorer:
    global _model
    if _model is None:
        if not _CHECKPOINT.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {_CHECKPOINT}\n"
                "Run scripts/train_lstm_full.py first."
            )
        m = LSTMScorer().to(_device)
        m.load_state_dict(torch.load(_CHECKPOINT, map_location=_device, weights_only=True))
        m.eval()
        _model = m
    return _model


def predict_fn(X: np.ndarray) -> np.ndarray:
    """
    Run inference with the best BiLSTM checkpoint.

    Parameters
    ----------
    X : np.ndarray [N, 60, 8]
        Preprocessed rep feature matrices (normalized, resampled to 60 frames).

    Returns
    -------
    np.ndarray [N]
        Predicted form scores in [0, 1].
    """
    model = _load()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32).to(_device)
        return model(x_t).cpu().numpy()
