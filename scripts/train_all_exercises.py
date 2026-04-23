"""
scripts/train_all_exercises.py

Trains one BiLSTM model per exercise and saves exercise-specific checkpoints.
Skips an exercise if its checkpoint already exists (use --force to retrain).

Checkpoints saved to paths defined in EXERCISE_CONFIGS[exercise]["model_path"]:
  checkpoints/lstm_squat.pt
  checkpoints/lstm_pushup.pt
  checkpoints/lstm_shoulder_press.pt
"""

import argparse
import pathlib
import sys
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from configs.exercises import EXERCISE_CONFIGS
from data.synthetic_loader import load_synthetic_exercise
from modeling.base_trainer import BaseTrainer
from modeling.lstm_scorer import LSTMScorer
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60
from preprocessing.normalizer import normalize

EXERCISES  = ["squat", "pushup", "shoulder_press"]
VAL_SPLIT  = 0.10
EPOCHS     = 200
PATIENCE   = 20


def build_dataset(exercise: str):
    clips  = load_synthetic_exercise(exercise)
    X_list, y_list = [], []

    for clip in clips:
        norm_lm  = normalize(clip["landmarks"])                    # [T, 33, 4]
        features = build_feature_matrix(norm_lm, exercise=exercise)  # [T, 8]

        for start, end in clip["reps"]:
            rep_lm = norm_lm[start:end]
            if len(rep_lm) < 5:
                continue
            feat  = build_feature_matrix(rep_lm, exercise=exercise)
            X_list.append(resample_to_60(feat))
            y_list.append(float(clip["form_scores"][start:end].mean()))

    X = np.stack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, len(clips)


def train_exercise(exercise: str, device: str, force: bool) -> dict:
    model_path = pathlib.Path(EXERCISE_CONFIGS[exercise]["model_path"])

    if model_path.exists() and not force:
        print(f"  Checkpoint exists — skipping ({model_path})")
        # load val MAE from a quick eval so summary table is still populated
        X, y, n_clips = build_dataset(exercise)
        model = LSTMScorer(in_features=8).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X, dtype=torch.float32).to(device))
            val_mae = torch.abs(preds - torch.tensor(y).to(device)).mean().item()
        return {"exercise": exercise, "clips": n_clips, "reps": len(X), "val_mae": val_mae}

    print(f"  Building dataset...")
    X, y, n_clips = build_dataset(exercise)
    n_reps = len(X)
    print(f"  Clips={n_clips}  Reps={n_reps}  y=[{y.min():.3f}, {y.max():.3f}]")

    rng   = np.random.default_rng(42)
    idx   = rng.permutation(n_reps)
    split = max(1, int(n_reps * VAL_SPLIT))
    val_idx, train_idx = idx[:split], idx[split:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model   = LSTMScorer(in_features=8).to(device)
    trainer = BaseTrainer(
        model          = model,
        lr             = 1e-3,
        patience       = PATIENCE,
        batch_size     = 32,
        checkpoint_dir = str(model_path.parent),
    )

    history = trainer.fit(X_train, y_train, X_val, y_val, epochs=EPOCHS, verbose=True)

    # BaseTrainer saves to {dir}/{classname}_best.pt — move to the config path
    generic = model_path.parent / "lstmscorer_best.pt"
    if generic.exists() and generic != model_path:
        generic.rename(model_path)
    print(f"  Saved: {model_path}")

    return {"exercise": exercise, "clips": n_clips, "reps": n_reps,
            "val_mae": history["best_val_mae"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if checkpoint already exists")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    summary = []
    for exercise in EXERCISES:
        print(f"{'='*55}")
        print(f"  Exercise: {exercise}")
        print(f"{'='*55}")
        row = train_exercise(exercise, device, force=args.force)
        summary.append(row)

    print(f"\n{'='*55}")
    print(f"  {'Exercise':<18} {'Clips':>6}  {'Reps':>5}  {'Val MAE':>8}")
    print(f"  {'-'*18}  {'-'*5}  {'-'*5}  {'-'*8}")
    for row in summary:
        print(f"  {row['exercise']:<18} {row['clips']:>6}  {row['reps']:>5}  {row['val_mae']:>8.4f}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
