"""HMM wrapper for 2-state trend/range detection."""

from __future__ import annotations

import numpy as np

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as e:
    raise ImportError(
        "hmmlearn is required for the regime filter. "
        "Install it with: uv add hmmlearn"
    ) from e


class HMMTrendRangeModel:
    """Thin wrapper around a 2-state GaussianHMM."""

    def __init__(
        self,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self._model = GaussianHMM(
            n_components=2,
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self._fitted = False

    def fit(self, X: np.ndarray) -> None:
        """Fit the HMM on a z-scored feature matrix (W x 3).

        Creates a fresh model each time to avoid degenerate covariance carry-over.
        """
        self._model = GaussianHMM(
            n_components=self._model.n_components,
            covariance_type=self._model.covariance_type,
            n_iter=self._model.n_iter,
            random_state=self._model.random_state,
        )
        self._model.fit(X)
        self._fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities for each row of X.

        Returns shape (len(X), 2).
        """
        if not self._fitted:
            raise RuntimeError("HMM not fitted yet")
        return self._model.predict_proba(X)

    def get_trend_state(self, feature_names: list[str]) -> int:
        """Identify which HMM state corresponds to TREND.

        Heuristic: the state with higher mean Efficiency Ratio is the trend state.

        Parameters
        ----------
        feature_names : list[str]
            Column names used during fit — ER must be at index matching
            its position in the feature vector.
        """
        if not self._fitted:
            raise RuntimeError("HMM not fitted yet")
        er_idx = feature_names.index("ER")
        means = self._model.means_  # shape (2, n_features)
        return int(np.argmax(means[:, er_idx]))
