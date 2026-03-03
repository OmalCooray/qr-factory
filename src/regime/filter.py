"""Stateful regime filter with hysteresis — feeds bars, emits regime info."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .features import compute_regime_features
from .hmm_model import HMMTrendRangeModel


@dataclass(frozen=True)
class RegimeInfo:
    """Snapshot of regime state for the current bar."""

    p_trend: float       # posterior probability of TREND state
    regime: str          # "TREND" or "RANGE"
    trend_state: int     # which HMM state is the trend state (-1 if not yet fitted)
    raw_state_probs: tuple[float, float]  # (p_state0, p_state1)
    er: float
    vol: float
    mom: float


class TrendRangeFilter:
    """Rolling 2-state HMM filter with hysteresis thresholds.

    Parameters
    ----------
    k : int
        Number of log-returns for feature computation.
    window_W : int
        Number of feature rows to keep for HMM fitting.
    refit_every : int
        Refit HMM every N new feature rows.
    predict_every : int
        Run predict_proba every N bars (cache last result otherwise).
        Regime changes are slow, so predicting every 5-10 bars is fine.
    enter_th : float
        p_trend must exceed this to enter TREND mode.
    exit_th : float
        p_trend must fall below this to exit TREND mode.
    features : list[str]
        Feature names (must include "ER").
    covariance_type : str
        HMM covariance type.
    n_iter : int
        HMM EM iterations.
    random_state : int
        HMM random seed.
    """

    def __init__(
        self,
        k: int = 20,
        window_W: int = 5000,
        refit_every: int = 50,
        predict_every: int = 10,
        enter_th: float = 0.70,
        exit_th: float = 0.50,
        features: Optional[list[str]] = None,
        covariance_type: str = "full",
        n_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.k = k
        self.window_W = window_W
        self.refit_every = refit_every
        self.predict_every = predict_every
        self.enter_th = enter_th
        self.exit_th = exit_th
        self.feature_names = features or ["ER", "vol", "mom"]

        # Internal buffers
        self._close_buf: deque[float] = deque(maxlen=k + 1)
        self._feat_buf: deque[tuple[float, float, float]] = deque(maxlen=window_W)

        # Z-score parameters (computed at fit time)
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None

        # HMM model
        self._hmm = HMMTrendRangeModel(
            covariance_type=covariance_type,
            n_iter=n_iter,
            random_state=random_state,
        )
        self._trend_state: int = -1
        self._bars_since_fit: int = 0
        self._bars_since_predict: int = 0
        self._ever_fitted: bool = False

        # Hysteresis state
        self._trend_mode: bool = False

        # Cached prediction result
        self._cached_p_trend: float = 0.0
        self._cached_probs: tuple[float, float] = (0.0, 0.0)

    def update(self, close: float) -> RegimeInfo:
        """Process one new close price, return regime info."""
        self._close_buf.append(close)

        # Not enough closes for feature computation
        if len(self._close_buf) < self.k + 1:
            return self._warmup_info()

        # Compute features for this bar
        closes_arr = np.array(self._close_buf, dtype=np.float64)
        er, vol, mom = compute_regime_features(closes_arr, self.k)
        self._feat_buf.append((er, vol, mom))
        self._bars_since_fit += 1
        self._bars_since_predict += 1

        # Not enough feature rows for HMM fitting
        if len(self._feat_buf) < self.window_W:
            return RegimeInfo(
                p_trend=0.0,
                regime="RANGE",
                trend_state=-1,
                raw_state_probs=(0.0, 0.0),
                er=er, vol=vol, mom=mom,
            )

        # Refit on schedule
        need_refit = not self._ever_fitted or self._bars_since_fit >= self.refit_every
        if need_refit:
            self._fit()
            # Force predict after refit
            self._bars_since_predict = self.predict_every

        # Predict (only every predict_every bars, or after refit)
        if self._bars_since_predict >= self.predict_every:
            X = np.array(self._feat_buf, dtype=np.float64)
            X_z = self._standardize(X)
            probs = self._hmm.predict_proba(X_z)
            last_probs = probs[-1]  # shape (2,)
            self._cached_p_trend = float(last_probs[self._trend_state])
            self._cached_probs = (float(last_probs[0]), float(last_probs[1]))
            self._bars_since_predict = 0

        p_trend = self._cached_p_trend

        # Hysteresis
        if self._trend_mode:
            if p_trend < self.exit_th:
                self._trend_mode = False
        else:
            if p_trend > self.enter_th:
                self._trend_mode = True

        regime = "TREND" if self._trend_mode else "RANGE"

        return RegimeInfo(
            p_trend=p_trend,
            regime=regime,
            trend_state=self._trend_state,
            raw_state_probs=self._cached_probs,
            er=er, vol=vol, mom=mom,
        )

    def _fit(self) -> None:
        """Fit HMM on current feature buffer."""
        X = np.array(self._feat_buf, dtype=np.float64)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0)
        # Guard against zero std
        self._std[self._std == 0] = 1.0

        X_z = self._standardize(X)
        self._hmm.fit(X_z)
        self._trend_state = self._hmm.get_trend_state(self.feature_names)
        self._bars_since_fit = 0
        self._ever_fitted = True

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        """Z-score using stored parameters."""
        return (X - self._mean) / self._std

    def _warmup_info(self) -> RegimeInfo:
        """Return conservative RANGE info during warmup."""
        return RegimeInfo(
            p_trend=0.0,
            regime="RANGE",
            trend_state=-1,
            raw_state_probs=(0.0, 0.0),
            er=0.0, vol=0.0, mom=0.0,
        )
