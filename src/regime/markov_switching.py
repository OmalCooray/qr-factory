"""Markov Switching autoregression regime detector."""

from __future__ import annotations

import numpy as np

from .protocol import RegimeLabel


class MarkovSwitchingDetector:
    """Markov Switching AR(1) on log-returns.

    Batch detector: call fit_batch() with full price series before bar iteration.
    High-variance state → RANGE, low-variance state → TREND.
    """

    name: str = "MarkovSW"

    def __init__(
        self,
        max_fit_bars: int = 50_000,
        subsample_factor: int = 1,
    ) -> None:
        self._max_fit_bars = max_fit_bars
        self._subsample_factor = subsample_factor
        self.warmup_bars: int = 0  # batch: no streaming warmup

        self._labels: np.ndarray | None = None
        self._confidences: np.ndarray | None = None
        self._raw_values: np.ndarray | None = None
        self._fitted: bool = False

    def fit_batch(self, closes: np.ndarray) -> None:
        """Fit Markov Switching model on the full price series.

        Parameters
        ----------
        closes : np.ndarray
            Full close price series.
        """
        import warnings
        from statsmodels.tsa.regime_switching.markov_autoregression import (
            MarkovAutoregression,
        )

        log_returns = np.diff(np.log(closes))

        # Subsample for speed if needed
        if len(log_returns) > self._max_fit_bars:
            fit_data = log_returns[-self._max_fit_bars:]
        else:
            fit_data = log_returns

        if self._subsample_factor > 1:
            fit_data = fit_data[::self._subsample_factor]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = MarkovAutoregression(
                fit_data,
                k_regimes=2,
                order=1,
                switching_variance=True,
            )
            result = model.fit(maxiter=200, disp=False)

        # Get smoothed probabilities: ndarray shape (n_fit, k_regimes)
        smoothed_probs = result.smoothed_marginal_probabilities

        # Identify which regime is TREND (lower variance = more directional)
        # param_names lives on the model, not the result
        param_names = model.param_names
        sigma_0 = result.params[param_names.index("sigma2[0]")]
        sigma_1 = result.params[param_names.index("sigma2[1]")]
        trend_regime = 0 if sigma_0 < sigma_1 else 1

        # Build full-length labels array
        n_total = len(closes)
        self._labels = np.full(n_total, "RANGE", dtype="U5")
        self._confidences = np.full(n_total, 0.0)
        self._raw_values = np.full(n_total, 0.0)

        # smoothed_probs has one fewer row than fit_data due to AR(1) lag
        n_probs = smoothed_probs.shape[0]
        # Map fitted probs back to original indices
        # fit_data has len(fit_data) entries; probs may be shorter by `order`
        offset = n_total - 1 - len(fit_data)
        prob_offset = len(fit_data) - n_probs  # AR lag offset

        if self._subsample_factor > 1:
            for i in range(n_probs):
                p_trend = float(smoothed_probs[i, trend_regime])
                orig_start = offset + 1 + (i + prob_offset) * self._subsample_factor
                orig_end = min(orig_start + self._subsample_factor, n_total)
                for j in range(max(0, orig_start), orig_end):
                    self._labels[j] = "TREND" if p_trend > 0.5 else "RANGE"
                    self._confidences[j] = p_trend if p_trend > 0.5 else 1.0 - p_trend
                    self._raw_values[j] = p_trend
        else:
            for i in range(n_probs):
                idx = offset + 1 + i + prob_offset
                if 0 <= idx < n_total:
                    p_trend = float(smoothed_probs[i, trend_regime])
                    self._labels[idx] = "TREND" if p_trend > 0.5 else "RANGE"
                    self._confidences[idx] = p_trend if p_trend > 0.5 else 1.0 - p_trend
                    self._raw_values[idx] = p_trend

        self._fitted = True

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        if not self._fitted or self._labels is None:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)

        if bar_index < len(self._labels):
            return RegimeLabel(
                regime=str(self._labels[bar_index]),
                confidence=float(self._confidences[bar_index]),
                raw_value=float(self._raw_values[bar_index]),
            )
        return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)
