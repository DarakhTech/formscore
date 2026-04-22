"""
modeling/rf_baseline.py

Random Forest baseline for FormScore.

Feature extraction: [N, 60, 8] → per-feature summary stats → [N, 40]
  5 stats (mean, std, min, max, range) × 8 features = 40 dimensions
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from modeling.evaluate import evaluate_model


def _extract_features(X: np.ndarray) -> np.ndarray:
    """[N, 60, 8] → [N, 40] summary stat vector."""
    mean = X.mean(axis=1)   # [N, 8]
    std  = X.std(axis=1)    # [N, 8]
    mn   = X.min(axis=1)    # [N, 8]
    mx   = X.max(axis=1)    # [N, 8]
    rng  = mx - mn          # [N, 8]
    return np.concatenate([mean, std, mn, mx, rng], axis=1)  # [N, 40]


class RFBaseline:
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RFBaseline":
        """X_train: [N, 60, 8], y_train: [N]"""
        self._rf.fit(_extract_features(X_train), y_train)
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """X_test: [N, 60, 8] → [N] predictions"""
        return self._rf.predict(_extract_features(X_test))

    def predict_fn(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """evaluate.py-compatible callable: fits on train split, predicts test."""
        self.fit(X_train, y_train)
        return self.predict(X_test)


if __name__ == "__main__":
    model = RFBaseline()
    results = evaluate_model(
        model_fn=model.predict_fn,
        model_name="E3_rf",
    )
    print(f"\nMAE: {results['mae'].mean():.4f} ± {results['mae'].std():.4f}")
    print(f"R²:  {results['r2'].mean():.4f} ± {results['r2'].std():.4f}")