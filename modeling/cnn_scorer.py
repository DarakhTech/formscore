# modeling/cnn_scorer.py
"""
1D-CNN form scorer for FormScore.

Architecture:
  Input [B, 60, 8]
  -> Conv1D(32, k=5) + BN + ReLU
  -> Conv1D(64, k=3) + BN + ReLU + MaxPool(2)
  -> Conv1D(128, k=3) + BN + ReLU
  -> GlobalAvgPool
  -> Dense(64) + ReLU + Dropout(0.3)
  -> Dense(1)  -> score 0-100
"""

import numpy as np
import torch
import torch.nn as nn


class CNNScorer(nn.Module):

    def __init__(self, in_features: int = 8, dropout: float = 0.3):
        super().__init__()

        self.conv_block = nn.Sequential(
            # Block 1
            nn.Conv1d(in_features, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Block 2
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),   # [B, 64, 30]
            # Block 3
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 60, 8]  (batch, frames, features)
        returns: [B] scores
        """
        x = x.permute(0, 2, 1)          # [B, 8, 60] — Conv1d expects (B, C, L)
        x = self.conv_block(x)           # [B, 128, 30]
        x = x.mean(dim=-1)              # GlobalAvgPool -> [B, 128]
        x = self.head(x)                 # [B, 1]
        return x.squeeze(-1)             # [B]


class CNNScorerWrapper:
    """
    Sklearn-style wrapper so CNNScorer works with evaluate.cross_validate().
    Training happens in the shared TrainingLoop (s1-m6).
    This wrapper is for inference only during CV.
    """

    def __init__(
        self,
        in_features: int = 8,
        dropout: float = 0.3,
        epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 32,
        patience: int = 20,
        device: str = None,
    ):
        self.in_features = in_features
        self.dropout     = dropout
        self.epochs      = epochs
        self.lr          = lr
        self.batch_size  = batch_size
        self.patience    = patience
        self.device      = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_      = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        from modeling.train_loop import TrainingLoop
        self.model_ = CNNScorer(self.in_features, self.dropout).to(self.device)
        loop = TrainingLoop(
            model=self.model_,
            lr=self.lr,
            epochs=self.epochs,
            batch_size=self.batch_size,
            patience=self.patience,
            device=self.device,
        )
        loop.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model_.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model_(x_t).cpu().numpy()


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = CNNScorer(in_features=8).to(device)

    # Forward pass with dummy batch
    x = torch.randn(16, 60, 8).to(device)   # batch of 16 reps
    out = model(x)

    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Output range:  {out.min().item():.2f} - {out.max().item():.2f}")

    # Parameter count
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    print("CNN architecture OK")