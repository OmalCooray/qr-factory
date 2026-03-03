"""Adapter wrapping the existing TrendRangeFilter into RegimeDetector."""

from __future__ import annotations

from .filter import TrendRangeFilter, RegimeInfo
from .protocol import RegimeLabel


class HMMRegimeDetector:
    """Wraps TrendRangeFilter to satisfy the RegimeDetector protocol."""

    name: str = "HMM"

    def __init__(
        self,
        k: int = 20,
        window_W: int = 5000,
        refit_every: int = 500,
        predict_every: int = 10,
        enter_th: float = 0.70,
        exit_th: float = 0.50,
    ) -> None:
        self._filter = TrendRangeFilter(
            k=k,
            window_W=window_W,
            refit_every=refit_every,
            predict_every=predict_every,
            enter_th=enter_th,
            exit_th=exit_th,
        )
        self.warmup_bars: int = k + window_W

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        info: RegimeInfo = self._filter.update(close)
        return RegimeLabel(
            regime=info.regime,
            confidence=info.p_trend if info.regime == "TREND" else 1.0 - info.p_trend,
            raw_value=info.p_trend,
        )
