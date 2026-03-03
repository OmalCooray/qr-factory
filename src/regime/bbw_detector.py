"""Bollinger Band Width percentile-rank regime detector."""

from __future__ import annotations

from collections import deque
import math

from .protocol import RegimeLabel


class BBWidthDetector:
    """Bollinger Band Width as a regime proxy.

    High BBW percentile → TREND (expanding volatility),
    low BBW percentile → RANGE (compressed volatility).
    Uses hysteresis on the percentile rank in a rolling lookback.
    """

    name: str = "BBWidth"

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        rank_window: int = 200,
        trend_pct: float = 65.0,
        range_pct: float = 35.0,
    ) -> None:
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._rank_window = rank_window
        self._trend_pct = trend_pct
        self._range_pct = range_pct
        self.warmup_bars: int = bb_period + rank_window

        self._closes: deque[float] = deque(maxlen=bb_period)
        self._bbw_history: deque[float] = deque(maxlen=rank_window)
        self._trend_mode: bool = False

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        self._closes.append(close)

        if len(self._closes) < self._bb_period:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)

        # Compute BB width
        arr = list(self._closes)
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        std = math.sqrt(var)
        upper = mean + self._bb_std * std
        lower = mean - self._bb_std * std
        bbw = (upper - lower) / mean if mean > 0 else 0.0

        self._bbw_history.append(bbw)

        if len(self._bbw_history) < self._rank_window:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=bbw)

        # Percentile rank of current BBW
        current = bbw
        below = sum(1 for v in self._bbw_history if v < current)
        pct = 100.0 * below / len(self._bbw_history)

        # Hysteresis
        if self._trend_mode:
            if pct < self._range_pct:
                self._trend_mode = False
        else:
            if pct > self._trend_pct:
                self._trend_mode = True

        regime = "TREND" if self._trend_mode else "RANGE"
        conf = abs(pct - 50.0) / 50.0
        return RegimeLabel(regime=regime, confidence=conf, raw_value=pct)
