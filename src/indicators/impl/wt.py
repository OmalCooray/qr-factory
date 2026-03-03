"""WaveTrend — WaveTrend Classic oscillator (WT1 - WT2)."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WaveTrend:
    n1: int = 10
    n2: int = 11

    @property
    def name(self) -> str:
        return f"wt_{self.n1}_{self.n2}"

    @property
    def lookback(self) -> int:
        return self.n1 + self.n2 + 4

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in ohlcv.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")

        # hlc3 as source
        src = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0

        ema1 = src.ewm(span=self.n1, min_periods=self.n1, adjust=False).mean()
        d = (src - ema1).abs()
        ema_d = d.ewm(span=self.n1, min_periods=self.n1, adjust=False).mean()

        ci = (src - ema1) / (0.015 * ema_d)
        # Replace inf/nan from division
        ci = ci.fillna(0.0).replace([float("inf"), float("-inf")], 0.0)

        wt1 = ci.ewm(span=self.n2, min_periods=self.n2, adjust=False).mean()
        wt2 = wt1.rolling(window=4, min_periods=4).mean()

        wt = wt1 - wt2  # The oscillator value

        return pd.DataFrame({self.name: wt.values}, index=ohlcv.index)
