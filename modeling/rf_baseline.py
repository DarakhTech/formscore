# modeling/rf_baseline.py
"""
Random Forest baseline for FormScore.

Input:  [N, 60, 8] feature matrix (N reps, 60 frames, 8 features)
Output: predicted form score 0-100

Strategy: flatten temporal dimension into summary stats per feature
  mean, std, min, max, range = 5 stats x 8 features = 40-dim vector
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin


class RFScorer(BaseEstimator, RegressorMixin):

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        self.n_estimators  = n_estimators
        self.random_state  = random_state
        self._model        = None

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """
        [N, 60, 8] -> [N, 40] summary stat vector.
        Stats: mean, std, min, max, range per feature.
        """
        mean  = X.mean(axis=1)           # [N, 8]
        std   = X.std(axis=1)            # [N, 8]
        mn    = X.min(axis=1)            # [N, 8]
        mx    = X.max(axis=1)            # [N, 8]
        rng   = mx - mn                  # [N, 8]
        return np.concatenate([mean, std, mn, mx, rng], axis=1)  # [N, 40]

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: [N, 60, 8], y: [N] scores 0-100"""
        features = self._extract_features(X)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self._model.fit(features, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: [N, 60, 8] -> [N] predicted scores"""
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        features = self._extract_features(X)
        return self._model.predict(features)

    def feature_importances(self) -> np.ndarray:
        """Returns [40] importance vector."""
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return self._model.feature_importances_


# ── Smoke test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # Fake data: 50 reps, 60 frames, 8 features
    np.random.seed(42)
    X_dummy = np.random.randn(50, 60, 8).astype(np.float32)
    y_dummy = np.random.uniform(40, 95, 50).astype(np.float32)

    model = RFScorer()
    model.fit(X_dummy, y_dummy)
    preds = model.predict(X_dummy)

    print(f"Input shape:      {X_dummy.shape}")
    print(f"Feature shape:    {model._extract_features(X_dummy).shape}")
    print(f"Predictions:      {preds[:5].round(1)}")
    print(f"Pred range:       {preds.min():.1f} - {preds.max():.1f}")
    print(f"Top 3 feat idx:   {model.feature_importances().argsort()[-3:][::-1]}")
    print("RF baseline OK")