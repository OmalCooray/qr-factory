"""K-Means clustering regime detector."""

from __future__ import annotations

from collections import deque

import numpy as np

from .features import compute_regime_features
from .protocol import RegimeLabel


class KMeansDetector:
    """2-cluster K-Means on ER/vol/mom features.

    TREND = cluster with higher centroid ER.
    Uses hysteresis based on distance ratio to cluster centroids.
    """

    name: str = "KMeans"

    def __init__(
        self,
        k: int = 20,
        window: int = 5000,
        refit_every: int = 500,
        predict_every: int = 10,
        enter_th: float = 0.60,
        exit_th: float = 0.40,
    ) -> None:
        self._k = k
        self._window = window
        self._refit_every = refit_every
        self._predict_every = predict_every
        self._enter_th = enter_th
        self._exit_th = exit_th
        self.warmup_bars: int = k + window

        self._close_buf: deque[float] = deque(maxlen=k + 1)
        self._feat_buf: deque[tuple[float, float, float]] = deque(maxlen=window)

        self._km = None
        self._trend_cluster: int = 0
        self._ever_fitted: bool = False
        self._bars_since_fit: int = 0
        self._bars_since_predict: int = 0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self._cached_p_trend: float = 0.0
        self._trend_mode: bool = False

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        self._close_buf.append(close)

        if len(self._close_buf) < self._k + 1:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)

        closes_arr = np.array(self._close_buf, dtype=np.float64)
        er, vol, mom = compute_regime_features(closes_arr, self._k)
        self._feat_buf.append((er, vol, mom))
        self._bars_since_fit += 1
        self._bars_since_predict += 1

        if len(self._feat_buf) < self._window:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)

        need_refit = not self._ever_fitted or self._bars_since_fit >= self._refit_every
        if need_refit:
            self._fit()
            self._bars_since_predict = self._predict_every

        if self._bars_since_predict >= self._predict_every:
            X = np.array(self._feat_buf, dtype=np.float64)
            X_z = (X - self._mean) / self._std
            point = X_z[-1:]
            dists = self._km.transform(point)[0]  # distances to each centroid
            # Convert distances to soft assignment (closer = higher prob)
            total = dists.sum()
            if total > 0:
                # Invert: smaller distance = higher probability
                inv_dists = total - dists
                p_trend = float(inv_dists[self._trend_cluster] / inv_dists.sum())
            else:
                p_trend = 0.5
            self._cached_p_trend = p_trend
            self._bars_since_predict = 0

        p_trend = self._cached_p_trend

        if self._trend_mode:
            if p_trend < self._exit_th:
                self._trend_mode = False
        else:
            if p_trend > self._enter_th:
                self._trend_mode = True

        regime = "TREND" if self._trend_mode else "RANGE"
        conf = p_trend if regime == "TREND" else 1.0 - p_trend
        return RegimeLabel(regime=regime, confidence=conf, raw_value=p_trend)

    def _fit(self) -> None:
        from sklearn.cluster import KMeans

        X = np.array(self._feat_buf, dtype=np.float64)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_z = (X - self._mean) / self._std

        self._km = KMeans(n_clusters=2, random_state=42, n_init=10)
        self._km.fit(X_z)

        # Trend cluster = one with higher centroid ER (column 0 after z-score)
        self._trend_cluster = int(np.argmax(self._km.cluster_centers_[:, 0]))
        self._bars_since_fit = 0
        self._ever_fitted = True
