# evaluate.py
"""
Shared evaluation framework for FormScore models.

Uses GroupKFold(5) with subject-stratified splits so the same
person never appears in both train and test — prevents data leakage.

Usage:
    from evaluate import cross_validate
    results = cross_validate(model, X, y, groups)
"""

import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score
from dataclasses import dataclass, field


@dataclass
class EvalResults:
    mae_per_fold:  list[float] = field(default_factory=list)
    r2_per_fold:   list[float] = field(default_factory=list)

    @property
    def mae_mean(self) -> float:
        return float(np.mean(self.mae_per_fold))

    @property
    def mae_std(self) -> float:
        return float(np.std(self.mae_per_fold))

    @property
    def r2_mean(self) -> float:
        return float(np.mean(self.r2_per_fold))

    @property
    def r2_std(self) -> float:
        return float(np.std(self.r2_per_fold))

    def summary(self, model_name: str = "model") -> str:
        return (
            f"{model_name:12s} | "
            f"MAE {self.mae_mean:.2f} ± {self.mae_std:.2f} | "
            f"R²  {self.r2_mean:.3f} ± {self.r2_std:.3f}"
        )


def cross_validate(
    model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    verbose: bool = True,
) -> EvalResults:
    """
    Subject-stratified k-fold cross validation.

    Args:
        model:    any model with fit(X, y) and predict(X) methods
        X:        [N, 60, 8] feature matrix
        y:        [N] form scores 0-100
        groups:   [N] subject/video IDs — same subject stays in same fold
        n_splits: number of folds (default 5)
        verbose:  print per-fold results

    Returns:
        EvalResults with mae and r2 per fold + mean/std
    """
    kf      = GroupKFold(n_splits=n_splits)
    results = EvalResults()

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        r2  = r2_score(y_val, preds)

        results.mae_per_fold.append(mae)
        results.r2_per_fold.append(r2)

        if verbose:
            print(f"  Fold {fold+1}: MAE={mae:.2f}  R²={r2:.3f}  "
                  f"(val_size={len(val_idx)}, subjects={len(np.unique(groups[val_idx]))})")

    return results


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np
    from modeling.rf_baseline import RFScorer

    np.random.seed(42)

    # Simulate 200 reps from 20 subjects (10 reps each)
    N_SUBJECTS = 20
    REPS_EACH  = 10
    N          = N_SUBJECTS * REPS_EACH

    X      = np.random.randn(N, 60, 8).astype(np.float32)
    y      = np.random.uniform(40, 95, N).astype(np.float32)
    groups = np.repeat(np.arange(N_SUBJECTS), REPS_EACH)  # subject IDs

    print(f"Dataset: {N} reps, {N_SUBJECTS} subjects, {REPS_EACH} reps each")
    print(f"Running 5-fold GroupKFold cross-validation...\n")

    model   = RFScorer()
    results = cross_validate(model, X, y, groups)

    print(f"\n{results.summary('RF baseline')}")
    print(f"\nMAE per fold: {[round(m,2) for m in results.mae_per_fold]}")
    print(f"R²  per fold: {[round(r,3) for r in results.r2_per_fold]}")