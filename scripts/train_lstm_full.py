"""
scripts/train_lstm_full.py

Retrain BiLSTM on the full dataset (no cross-validation).
A 10% random split is held out only for early-stopping; all data
informs the final model.

Saves checkpoint to: checkpoints/lstm_best_full.pt
"""

import sys
import pathlib
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from modeling.evaluate import _build_dataset
from modeling.lstm_scorer import LSTMScorer
from modeling.train_loop import TrainingLoop

CHECKPOINT_NAME = "lstm_best_full"
CHECKPOINT_PATH = pathlib.Path("checkpoints") / f"{CHECKPOINT_NAME}.pt"


def main():
    print("Loading dataset...")
    X, y, _ = _build_dataset()
    print(f"  X: {X.shape}  y: {y.shape}  y range: [{y.min():.3f}, {y.max():.3f}]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    model = LSTMScorer().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {n_params:,}")

    loop = TrainingLoop(
        model=model,
        lr=1e-3,
        epochs=200,
        batch_size=32,
        patience=20,
        val_split=0.10,
        device=device,
        checkpoint_name=CHECKPOINT_NAME,
    )

    print("\nTraining...")
    history = loop.fit(X, y, verbose=True)

    print(f"\nBest epoch : {history['best_epoch'] + 1}")
    print(f"Best val MAE: {history['best_val_mae']:.4f}")
    print(f"Checkpoint : {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
