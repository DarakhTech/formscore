"""
scripts/run_e4_ablation.py

E4 Feature Ablation: Angles-only [60,5] vs Full features [60,8].
Uses BiLSTM (LSTMScorer) with GroupKFold(5) for both configs.
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from modeling.evaluate import _build_dataset
from modeling.lstm_scorer import LSTMScorer
from modeling.train_loop import TrainingLoop

RESULTS_DIR = pathlib.Path("results")
N_SPLITS    = 5

CONFIGS = [
    {"name": "Angles only", "cols": slice(0, 5)},
    {"name": "Full [60,8]", "cols": slice(0, 8)},
]


def run_cv(X_full: np.ndarray, y: np.ndarray, groups: np.ndarray, cols: slice) -> pd.DataFrame:
    """GroupKFold CV for a feature slice. Returns per-fold DataFrame."""
    X       = X_full[:, :, cols]          # [N, 60, n_feat]
    n_feat  = X.shape[2]
    device  = "cuda" if torch.cuda.is_available() else "cpu"
    records = []

    for fold, (train_idx, test_idx) in enumerate(
        GroupKFold(N_SPLITS).split(X, y, groups), start=1
    ):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        model = LSTMScorer(in_features=n_feat).to(device)
        loop  = TrainingLoop(
            model=model,
            lr=1e-3,
            epochs=200,
            batch_size=32,
            patience=20,
            device=device,
            checkpoint_name=f"e4_ablation_feat{n_feat}_fold{fold}",
        )
        loop.fit(X_train, y_train, verbose=False)

        model.eval()
        with torch.no_grad():
            x_t   = torch.tensor(X_test, dtype=torch.float32).to(device)
            preds = model(x_t).cpu().numpy()

        records.append({
            "fold": fold,
            "mae":  mean_absolute_error(y_test, preds),
            "r2":   r2_score(y_test, preds),
        })
        print(f"    Fold {fold}  MAE={records[-1]['mae']:.4f}  R²={records[-1]['r2']:.4f}")

    return pd.DataFrame(records)


def main():
    print("Loading dataset...")
    X_full, y, groups = _build_dataset()
    print(f"  X: {X_full.shape}  reps: {len(y)}\n")

    summary_rows = []

    for cfg in CONFIGS:
        print(f"Running: {cfg['name']}")
        folds = run_cv(X_full, y, groups, cfg["cols"])
        row = {
            "config":   cfg["name"],
            "mae_mean": folds["mae"].mean(),
            "mae_std":  folds["mae"].std(),
            "r2_mean":  folds["r2"].mean(),
            "r2_std":   folds["r2"].std(),
        }
        summary_rows.append(row)
        print()

    summary = pd.DataFrame(summary_rows)

    # ── Print table ───────────────────────────────────────────────
    angles_mae = summary.loc[summary["config"] == "Angles only", "mae_mean"].values[0]
    full_mae   = summary.loc[summary["config"] == "Full [60,8]", "mae_mean"].values[0]
    delta      = angles_mae - full_mae
    pct        = (delta / angles_mae) * 100 if angles_mae > 0 else 0.0

    w = [16, 11, 11, 10]
    header = (
        f"{'Config':<{w[0]}}"
        f"{'MAE mean':<{w[1]}}"
        f"{'MAE std':<{w[2]}}"
        f"{'R2':<{w[3]}}"
    )
    rule = "─" * (sum(w) + 2)

    print("\n  E4 Feature Ablation Results")
    print(f"  {rule}")
    print(f"  {header}")
    for _, row in summary.iterrows():
        print(
            f"  {row['config']:<{w[0]}}"
            f"{row['mae_mean']:<{w[1]}.4f}"
            f"{row['mae_std']:<{w[2]}.4f}"
            f"{row['r2_mean']:<{w[3]}.4f}"
        )
    print(f"  {rule}")
    sign = "+" if delta >= 0 else "-"
    print(f"  {'Delta MAE':<{w[0]}}{sign}{abs(delta):.4f}   ({pct:.1f}% improvement from full features)")
    print()

    # ── Save ──────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "E4_ablation.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
