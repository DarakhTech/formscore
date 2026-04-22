"""
modeling/evaluate.py

Base cross-validation evaluator for FormScore models.

Usage
-----
    from modeling.evaluate import evaluate_model

    def my_model_fn(X_train, y_train, X_test):
        ...
        return predictions  # [N_test]

    evaluate_model(my_model_fn, model_name="my_model")
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

from data.synthetic_loader import load_synthetic_squats
from preprocessing.normalizer import normalize
from preprocessing.feature_engineer import build_feature_matrix, resample_to_60

SQUAT_DIR = "data/real-time-exercise-recognition-dataset/synthetic_dataset/synthetic_dataset/squat"
RESULTS_DIR = pathlib.Path("results")


def _build_dataset(squat_dir: str = SQUAT_DIR):
    """
    Returns X [N, 60, 8], y [N], groups [N] (video index per rep).
    """
    clips = load_synthetic_squats(squat_dir)

    X_list, y_list, groups_list = [], [], []

    for video_idx, clip in enumerate(clips):
        landmarks   = clip["landmarks"]    # [T, 33, 4]
        form_scores = clip["form_scores"]  # [T]
        reps        = clip["reps"]         # list of (start, end)

        norm_lm = normalize(landmarks)     # [T, 33, 4]
        features = build_feature_matrix(norm_lm)  # [T, 8]

        for start, end in reps:
            rep_features = features[start:end + 1]         # [rep_T, 8]
            rep_scores   = form_scores[start:end + 1]      # [rep_T]

            resampled = resample_to_60(rep_features)       # [60, 8]
            label     = float(np.mean(rep_scores))         # scalar

            X_list.append(resampled)
            y_list.append(label)
            groups_list.append(video_idx)

    X      = np.stack(X_list)                  # [N, 60, 8]
    y      = np.array(y_list, dtype=np.float32)  # [N]
    groups = np.array(groups_list, dtype=np.int32)  # [N]

    return X, y, groups


def evaluate_model(model_fn, model_name: str, squat_dir: str = SQUAT_DIR, n_splits: int = 5):
    """
    GroupKFold cross-validation for a FormScore model.

    Parameters
    ----------
    model_fn : callable
        Signature: model_fn(X_train, y_train, X_test) -> np.ndarray [N_test]
        Receives training and test splits; returns predictions for the test set.
    model_name : str
        Used for the output CSV filename.
    squat_dir : str
        Path to the synthetic squat dataset directory.
    n_splits : int
        Number of GroupKFold folds (default 5).

    Returns
    -------
    pd.DataFrame
        Fold-level results with columns: fold, mae, r2.
    """
    X, y, groups = _build_dataset(squat_dir)

    gkf     = GroupKFold(n_splits=n_splits)
    records = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        preds = model_fn(X_train, y_train, X_test)

        mae = mean_absolute_error(y_test, preds)
        r2  = r2_score(y_test, preds)

        records.append({"fold": fold, "mae": mae, "r2": r2})
        print(f"  Fold {fold}  MAE={mae:.4f}  R²={r2:.4f}")

    results = pd.DataFrame(records)

    print(f"\n--- {model_name} ---")
    print(f"  MAE: {results['mae'].mean():.4f} ± {results['mae'].std():.4f}")
    print(f"  R²:  {results['r2'].mean():.4f} ± {results['r2'].std():.4f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{model_name}_eval.csv"
    results.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return results
