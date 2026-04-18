# modeling/lstm_scorer.py
"""
Bidirectional LSTM form scorer for FormScore.

Architecture:
  Input [B, 60, 8]
  -> BiLSTM(64) + Dropout(0.3)
  -> BiLSTM(32) + Dropout(0.3)
  -> Dense(32) + ReLU
  -> Dense(1) -> score 0-100

Note: ~3x slower than CNN to train. Use Nebius GPU for full training runs.
Gradient clipping max_norm=1.0 applied in training loop.
"""

import numpy as np
import torch
import torch.nn as nn


class LSTMScorer(nn.Module):

    def __init__(self, in_features: int = 8, dropout: float = 0.3):
        super().__init__()

        # BiLSTM layer 1: hidden=64, bidirectional -> output 128
        self.lstm1 = nn.LSTM(
            input_size=in_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(dropout)

        # BiLSTM layer 2: input=128, hidden=32, bidirectional -> output 64
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
        """
        x: [B, 60, 8]
        returns: [B] scores
        """
        # LSTM layer 1
        out, _ = self.lstm1(x)           # [B, 60, 128]
        out    = self.drop1(out)

        # LSTM layer 2
        out, _ = self.lstm2(out)         # [B, 60, 64]
        out    = self.drop2(out)

        # Take last timestep
        out = out[:, -1, :]              # [B, 64]
        out = self.head(out)             # [B, 1]
        return out.squeeze(-1)           # [B]


class LSTMScorerWrapper:
    """
    Sklearn-style wrapper so LSTMScorer works with evaluate.cross_validate().
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
        self.model_ = LSTMScorer(self.in_features, self.dropout).to(self.device)
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

    model = LSTMScorer(in_features=8).to(device)

    x   = torch.randn(16, 60, 8).to(device)
    out = model(x)

    print(f"Input shape:   {x.shape}")
    print(f"Output shape:  {out.shape}")
    print(f"Output range:  {out.min().item():.2f} - {out.max().item():.2f}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_params:,}")
    print("LSTM architecture OK")