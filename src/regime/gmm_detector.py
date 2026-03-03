"""2-component Gaussian Mixture Model regime detector."""

from __future__ import annotations

from collections import deque

import numpy as np

from .features import compute_regime_features
from .protocol import RegimeLabel


class GMMDetector:
    """2-component GMM on ER/vol/mom features.

    TREND = component with higher mean ER.
    Uses hysteresis on the posterior probability.
    """

    name: str = "GMM"

    def __init__(
        self,
        k: int = 20,
        window: int = 5000,
        refit_every: int = 500,
        predict_every: int = 10,
        enter_th: float = 0.65,
        exit_th: float = 0.45,
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

        self._gmm = None
        self._trend_comp: int = 0
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
            probs = self._gmm.predict_proba(X_z)
            self._cached_p_trend = float(probs[-1, self._trend_comp])
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
        from sklearn.mixture import GaussianMixture

        X = np.array(self._feat_buf, dtype=np.float64)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        self._std[self._std == 0] = 1.0
        X_z = (X - self._mean) / self._std

        self._gmm = GaussianMixture(n_components=2, random_state=42, max_iter=200)
        self._gmm.fit(X_z)

        # Trend component = one with higher mean ER (column 0)
        self._trend_comp = int(np.argmax(self._gmm.means_[:, 0]))
        self._bars_since_fit = 0
        self._ever_fitted = True
