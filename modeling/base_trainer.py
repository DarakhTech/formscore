"""
modeling/base_trainer.py

Shared trainer for CNNScorer and LSTMScorer.

Differences from TrainingLoop:
  - fit() accepts explicit X_val/y_val instead of splitting internally
  - load_best() is a classmethod for clean checkpoint restoration
  - Checkpoint saved as {checkpoint_dir}/{model_name}_best.pt
"""

import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class BaseTrainer:
    """Shared trainer for CNNScorer and LSTMScorer with explicit train/val splits."""

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        patience: int = 20,
        batch_size: int = 32,
        checkpoint_dir: str = "checkpoints/",
    ):
        self.model          = model
        self.lr             = lr
        self.patience       = patience
        self.batch_size     = batch_size
        self.checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.device         = next(model.parameters()).device

    def _checkpoint_path(self) -> pathlib.Path:
        name = self.model.__class__.__name__.lower()
        return self.checkpoint_dir / f"{name}_best.pt"

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 200,
        verbose: bool = True,
    ) -> dict:
        """
        Train with explicit train/val splits.

        Parameters
        ----------
        X_train, X_val : np.ndarray [N, 60, 8]
        y_train, y_val : np.ndarray [N]
        epochs         : int
        verbose        : bool  print every 10 epochs

        Returns
        -------
        dict with keys:
          train_loss : list[float]  per-epoch mean MSE on train set
          val_mae    : list[float]  per-epoch MAE on val set
          best_epoch : int
          best_val_mae : float
        """
        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5
        )
        criterion = nn.MSELoss()

        best_val_mae  = float("inf")
        best_epoch    = 0
        patience_ctr  = 0
        train_loss_history = []
        val_mae_history    = []
        ckpt_path = self._checkpoint_path()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                preds = self.model(xb)
                loss  = criterion(preds, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(xb)

            epoch_loss /= len(X_train)
            scheduler.step()

            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(X_val_t)
                val_mae   = torch.abs(val_preds - y_val_t).mean().item()

            train_loss_history.append(epoch_loss)
            val_mae_history.append(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_epoch   = epoch
                patience_ctr = 0
                torch.save(self.model.state_dict(), ckpt_path)
            else:
                patience_ctr += 1

            if verbose and (epoch + 1) % 10 == 0:
                print(
                    f"  Epoch {epoch+1:3d} | "
                    f"train_loss={epoch_loss:.4f} | "
                    f"val_mae={val_mae:.4f} | "
                    f"patience={patience_ctr}/{self.patience}"
                )

            if patience_ctr >= self.patience:
                if verbose:
                    print(
                        f"  Early stop at epoch {epoch+1} "
                        f"(best={best_epoch+1}, val_mae={best_val_mae:.4f})"
                    )
                break

        self.model.load_state_dict(torch.load(ckpt_path, weights_only=True))

        if verbose:
            print(f"  Checkpoint: {ckpt_path}")

        return {
            "train_loss":  train_loss_history,
            "val_mae":     val_mae_history,
            "best_epoch":  best_epoch,
            "best_val_mae": best_val_mae,
        }

    @classmethod
    def load_best(cls, model: nn.Module, model_name: str, checkpoint_dir: str = "checkpoints/") -> nn.Module:
        """
        Load best checkpoint weights into model in-place and return it.

        Parameters
        ----------
        model        : nn.Module  instantiated model (correct architecture)
        model_name   : str        e.g. "cnnscorer" or explicit filename stem
        checkpoint_dir : str

        Returns
        -------
        model with weights loaded, set to eval mode
        """
        ckpt_path = pathlib.Path(checkpoint_dir) / f"{model_name}_best.pt"
        model.load_state_dict(torch.load(ckpt_path, weights_only=True))
        model.eval()
        return model
