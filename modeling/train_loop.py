# modeling/train_loop.py
"""
Shared training loop for CNNScorer and LSTMScorer.

Features:
  - Adam optimizer
  - Cosine LR decay: 1e-3 -> 1e-5
  - MSE loss
  - Early stopping (patience=20 on val MAE)
  - Gradient clipping max_norm=1.0 (critical for LSTM stability)
  - Saves best checkpoint to checkpoints/<model_name>.pt
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path


CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


class TrainingLoop:
    """Training loop with Adam, cosine LR decay, early stopping, and gradient clipping."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 32,
        patience: int = 20,
        val_split: float = 0.15,
        device: str = None,
        checkpoint_name: str = None,
    ):
        self.model          = model
        self.lr             = lr
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.patience       = patience
        self.val_split      = val_split
        self.device         = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_name = checkpoint_name or model.__class__.__name__.lower()

        self.model.to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> dict:
        """
        X: [N, 60, 8]
        y: [N] scores 0-100

        Returns: dict with train_losses, val_losses, best_epoch
        """
        # Build dataset
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)

        # Train/val split
        val_size   = max(1, int(len(dataset) * self.val_split))
        train_size = len(dataset) - val_size
        train_ds, val_ds = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        # Optimizer + scheduler + loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=1e-5
        )
        criterion = nn.MSELoss()

        # Training state
        best_val_mae  = float("inf")
        best_epoch    = 0
        patience_ctr  = 0
        train_losses  = []
        val_losses    = []
        checkpoint_path = CHECKPOINT_DIR / f"{self.checkpoint_name}.pt"

        for epoch in range(self.epochs):
            # ── Train ──────────────────────────────────────────
            self.model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss  = criterion(preds, yb)
                loss.backward()
                # Gradient clipping — essential for LSTM
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item() * len(xb)

            train_loss /= train_size
            scheduler.step()

            # ── Validate ───────────────────────────────────────
            self.model.eval()
            val_mae = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    preds   = self.model(xb)
                    val_mae += torch.abs(preds - yb).sum().item()
            val_mae /= val_size

            train_losses.append(train_loss)
            val_losses.append(val_mae)

            # ── Early stopping ─────────────────────────────────
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch   = epoch
                patience_ctr = 0
                torch.save(self.model.state_dict(), checkpoint_path)
            else:
                patience_ctr += 1

            if verbose and (epoch + 1) % 10 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  Epoch {epoch+1:3d} | train_loss={train_loss:.3f} | "
                      f"val_mae={val_mae:.2f} | lr={lr_now:.6f} | "
                      f"patience={patience_ctr}/{self.patience}")

            if patience_ctr >= self.patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch+1} (best epoch={best_epoch+1}, "
                          f"best_val_mae={best_val_mae:.2f})")
                break

        # Reload best checkpoint
        self.model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        if verbose:
            print(f"\nBest epoch: {best_epoch+1} | Best val MAE: {best_val_mae:.2f}")
            print(f"Checkpoint saved: {checkpoint_path}")

        return {
            "train_losses": train_losses,
            "val_losses":   val_losses,
            "best_epoch":   best_epoch,
            "best_val_mae": best_val_mae,
        }


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    from modeling.cnn_scorer  import CNNScorer
    from modeling.lstm_scorer import LSTMScorer

    np.random.seed(42)
    X = np.random.randn(80, 60, 8).astype(np.float32)
    y = np.random.uniform(40, 95, 80).astype(np.float32)

    print("=" * 50)
    print("Testing CNN training loop")
    print("=" * 50)
    cnn  = CNNScorer()
    loop = TrainingLoop(cnn, epochs=30, patience=10, checkpoint_name="cnn_test")
    res  = loop.fit(X, y)
    print(f"CNN final best_val_mae: {res['best_val_mae']:.2f}\n")

    print("=" * 50)
    print("Testing LSTM training loop")
    print("=" * 50)
    lstm = LSTMScorer()
    loop = TrainingLoop(lstm, epochs=30, patience=10, checkpoint_name="lstm_test")
    res  = loop.fit(X, y)
    print(f"LSTM final best_val_mae: {res['best_val_mae']:.2f}")