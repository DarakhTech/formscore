"""
modeling/load_model.py

Load BiLSTM checkpoints and expose predict functions.

Usage
-----
    from modeling.load_model import predict_fn, get_predict_fn

    preds = predict_fn(X)                          # default full checkpoint
    fn    = get_predict_fn("checkpoints/lstm_squat.pt")
    preds = fn(X)   # X: [N, 60, 8] numpy  →  preds: [N] numpy
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

    Returns
    -------
    np.ndarray [N]  predicted form scores in [0, 1]
    """
    model = _load()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32).to(_device)
        return model(x_t).cpu().numpy()


def get_predict_fn(model_path: str):
    """
    Load a BiLSTM checkpoint from an explicit path and return a predict_fn.

    Parameters
    ----------
    model_path : str  path to .pt checkpoint

    Returns
    -------
    callable  [N, 60, 8] → [N] numpy predict function

    Raises
    ------
    FileNotFoundError if the checkpoint does not exist
    """
    ckpt = pathlib.Path(model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    m = LSTMScorer().to(_device)
    m.load_state_dict(torch.load(ckpt, map_location=_device, weights_only=True))
    m.eval()

    def _fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(_device)
            return m(x_t).cpu().numpy()

    return _fn


def get_model_and_predict_fn(model_path: str):
    """
    Load a BiLSTM checkpoint and return both the raw module and predict wrapper.

    Parameters
    ----------
    model_path : str
        Path to .pt checkpoint.

    Returns
    -------
    tuple[LSTMScorer, callable]
        (model, predict_fn) where model is an eval-mode nn.Module and
        predict_fn maps [N, 60, 8] numpy -> [N] numpy.
    """
    ckpt = pathlib.Path(model_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    m = LSTMScorer().to(_device)
    m.load_state_dict(torch.load(ckpt, map_location=_device, weights_only=True))
    m.eval()

    def _fn(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(_device)
            return m(x_t).cpu().numpy()

    return m, _fn
