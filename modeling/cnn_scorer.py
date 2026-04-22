"""
modeling/cnn_scorer.py

1D-CNN form scorer for FormScore.

Architecture:
  Input [B, 60, 8]
  → permute [B, 8, 60]
  → Conv1d(8,  32, k=5, pad=2) + BN + ReLU
  → Conv1d(32, 64, k=3, pad=1) + BN + ReLU + MaxPool1d(2)
  → Conv1d(64,128, k=3, pad=1) + BN + ReLU
  → AdaptiveAvgPool1d(1) → flatten [B, 128]
  → Linear(128, 64) + ReLU + Dropout(0.3)
  → Linear(64, 1) → squeeze → sigmoid   output ∈ [0, 1]
"""

import numpy as np
import torch
import torch.nn as nn

from modeling.evaluate import evaluate_model
from modeling.train_loop import TrainingLoop


class CNNScorer(nn.Module):

    def __init__(self, in_features: int = 8, dropout: float = 0.3):
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),        # [B, 64, 30]
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 60, 8] → [B] predictions in [0, 1]"""
        x = x.permute(0, 2, 1)         # [B, 8, 60]
        x = self.conv_block(x)          # [B, 128, L]
        x = self.pool(x).squeeze(-1)    # [B, 128]
        x = self.head(x).squeeze(-1)    # [B]
        return torch.sigmoid(x)         # [B] ∈ [0, 1]


class CNNBaseline:
    """evaluate.py-compatible wrapper around CNNScorer + TrainingLoop."""

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

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CNNBaseline":
        """X_train: [N, 60, 8], y_train: [N] scores in [0, 1]"""
        self._model = CNNScorer().to(self.device)
        loop = TrainingLoop(
            model=self._model,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            device=self.device,
            checkpoint_name="cnn_baseline",
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
    model = CNNBaseline()
    results = evaluate_model(
        model_fn=model.predict_fn,
        model_name="E3_cnn",
    )
    print(f"\nMAE: {results['mae'].mean():.4f} ± {results['mae'].std():.4f}")
    print(f"R²:  {results['r2'].mean():.4f} ± {results['r2'].std():.4f}")