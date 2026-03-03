"""Choppiness Index regime detector."""

from __future__ import annotations

import math
from collections import deque

from .protocol import RegimeLabel


class ChoppinessDetector:
    """Choppiness Index: 100 * log10(sum(TR, n) / (HH - LL)) / log10(n).

    High CHOP (>61.8) → RANGE, low CHOP (<38.2) → TREND.
    Uses hysteresis to avoid flipping.
    """

    name: str = "Choppiness"

    def __init__(
        self,
        period: int = 14,
        trend_th: float = 38.2,
        range_th: float = 61.8,
    ) -> None:
        self._period = period
        self._trend_th = trend_th
        self._range_th = range_th
        self.warmup_bars: int = period + 1

        self._highs: deque[float] = deque(maxlen=period)
        self._lows: deque[float] = deque(maxlen=period)
        self._trs: deque[float] = deque(maxlen=period)
        self._prev_close: float | None = None
        self._trend_mode: bool = False

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        # True Range
        if self._prev_close is not None:
            tr = max(high - low, abs(high - self._prev_close), abs(low - self._prev_close))
        else:
            tr = high - low
        self._prev_close = close

        self._highs.append(high)
        self._lows.append(low)
        self._trs.append(tr)

        if len(self._trs) < self._period:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=50.0)

        hh = max(self._highs)
        ll = min(self._lows)
        atr_sum = sum(self._trs)

        denom = hh - ll
        if denom <= 0:
            chop = 50.0
        else:
            chop = 100.0 * math.log10(atr_sum / denom) / math.log10(self._period)
            chop = max(0.0, min(100.0, chop))

        # Hysteresis
        if self._trend_mode:
            if chop > self._range_th:
                self._trend_mode = False
        else:
            if chop < self._trend_th:
                self._trend_mode = True

        regime = "TREND" if self._trend_mode else "RANGE"
        conf = abs(chop - 50.0) / 50.0  # further from 50 = more confident
        return RegimeLabel(regime=regime, confidence=conf, raw_value=chop)
