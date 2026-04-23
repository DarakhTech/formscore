"""
modeling/lstm_scorer.py

Bidirectional LSTM form scorer for FormScore.

Architecture:
  Input [B, 60, 8]
  → BiLSTM(input=8,   hidden=64, bidirectional) → [B, 60, 128] + Dropout(0.3)
  → BiLSTM(input=128, hidden=32, bidirectional) → [B, 60, 64]  + Dropout(0.3)
  → last timestep → [B, 64]
  → Linear(64, 32) + ReLU
  → Linear(32, 1) → squeeze → sigmoid   output ∈ [0, 1]

Gradient clipping max_norm=1.0 is handled by TrainingLoop.
"""

import numpy as np
import torch
import torch.nn as nn

from modeling.evaluate import evaluate_model
from modeling.train_loop import TrainingLoop


class LSTMScorer(nn.Module):
    """Bidirectional two-layer LSTM form scorer with sigmoid output in [0, 1]."""

    def __init__(self, in_features: int = 8, dropout: float = 0.3):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=in_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=128,
            hidden_size=32,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout)

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 60, 8] → [B] predictions in [0, 1]"""
        out, _ = self.lstm1(x)          # [B, 60, 128]
        out    = self.drop1(out)
        out, _ = self.lstm2(out)        # [B, 60, 64]
        out    = self.drop2(out)
        out    = out[:, -1, :]          # [B, 64]  last timestep
        out    = self.head(out)         # [B, 1]
        return torch.sigmoid(out.squeeze(-1))  # [B] ∈ [0, 1]


class LSTMBaseline:
    """evaluate.py-compatible wrapper around LSTMScorer + TrainingLoop."""

    def __init__(
        self,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 20,
        device: str = None,
    ):
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.patience   = patience
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model     = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LSTMBaseline":
        """X_train: [N, 60, 8], y_train: [N] scores in [0, 1]"""
        self._model = LSTMScorer().to(self.device)
        loop = TrainingLoop(
            model=self._model,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            device=self.device,
            checkpoint_name="lstm_baseline",
        )
        loop.fit(X_train, y_train, verbose=False)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """X_test: [N, 60, 8] → [N] numpy predictions"""
        self._model.eval()
        with torch.no_grad():
            x_t = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            return self._model(x_t).cpu().numpy()

    def predict_fn(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """evaluate.py-compatible callable: fits on train split, predicts test."""
        self.fit(X_train, y_train)
        return self.predict(X_test)


if __name__ == "__main__":
    model = LSTMBaseline()
    results = evaluate_model(
        model_fn=model.predict_fn,
        model_name="E3_lstm",
    )
    print(f"\nMAE: {results['mae'].mean():.4f} ± {results['mae'].std():.4f}")
    print(f"R²:  {results['r2'].mean():.4f} ± {results['r2'].std():.4f}")