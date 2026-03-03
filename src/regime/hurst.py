"""Rolling Hurst exponent regime detector via rescaled-range (R/S) analysis."""

from __future__ import annotations

from collections import deque

import numpy as np

from .protocol import RegimeLabel


class HurstDetector:
    """Rolling R/S Hurst exponent.

    H > 0.55 → TREND (persistent), H < 0.45 → RANGE (mean-reverting).
    Uses hysteresis and caches computation every `compute_every` bars.
    """

    name: str = "Hurst"

    def __init__(
        self,
        window: int = 200,
        trend_th: float = 0.55,
        range_th: float = 0.45,
        compute_every: int = 10,
    ) -> None:
        self._window = window
        self._trend_th = trend_th
        self._range_th = range_th
        self._compute_every = compute_every
        self.warmup_bars: int = window + 1

        self._closes: deque[float] = deque(maxlen=window + 1)
        self._trend_mode: bool = False
        self._cached_h: float = 0.5
        self._bars_since_compute: int = 0

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        self._closes.append(close)

        if len(self._closes) < self._window + 1:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.5)

        self._bars_since_compute += 1
        if self._bars_since_compute >= self._compute_every:
            prices = np.array(self._closes, dtype=np.float64)
            self._cached_h = _rs_hurst(prices, self._window)
            self._bars_since_compute = 0

        h = self._cached_h

        # Hysteresis
        if self._trend_mode:
            if h < self._range_th:
                self._trend_mode = False
        else:
            if h > self._trend_th:
                self._trend_mode = True

        regime = "TREND" if self._trend_mode else "RANGE"
        conf = abs(h - 0.5) / 0.5  # distance from random walk
        conf = min(1.0, conf)
        return RegimeLabel(regime=regime, confidence=conf, raw_value=h)


def _rs_hurst(prices: np.ndarray, window: int) -> float:
    """Simplified multi-scale R/S Hurst on the last `window` log-returns.

    Uses 4 sub-divisions: n, n/2, n/4, n/8 to get a regression estimate.
    """
    log_returns = np.diff(np.log(prices[-window - 1:]))
    n = len(log_returns)

    sizes = []
    rs_values = []

    for div in [1, 2, 4, 8]:
        chunk_size = n // div
        if chunk_size < 8:
            break
        rs_list = []
        for i in range(div):
            chunk = log_returns[i * chunk_size: (i + 1) * chunk_size]
            mean = chunk.mean()
            dev = chunk - mean
            cumdev = np.cumsum(dev)
            r = cumdev.max() - cumdev.min()
            s = chunk.std(ddof=1)
            if s > 0:
                rs_list.append(r / s)
        if rs_list:
            sizes.append(chunk_size)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 2:
        return 0.5

    log_n = np.log(sizes)
    log_rs = np.log(rs_values)

    # Linear regression slope = Hurst exponent
    slope = np.polyfit(log_n, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))
