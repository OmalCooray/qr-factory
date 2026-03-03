"""RSI — Relative Strength Index (Wilder's smoothing)."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RSI:
    period: int = 14
    src: str = "close"

    @property
    def name(self) -> str:
        return f"rsi_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period + 1

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.src not in ohlcv.columns:
            raise ValueError(f"Source column '{self.src}' not found in input DataFrame.")

        src = ohlcv[self.src].values.astype(np.float64)
        n = len(src)
        p = self.period
        rsi = np.full(n, np.nan, dtype=np.float64)

        if n < p + 1:
            return pd.DataFrame({self.name: rsi}, index=ohlcv.index)

        # Price changes
        delta = np.diff(src)
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # First average: simple mean of first `period` changes
        avg_gain = np.mean(gains[:p])
        avg_loss = np.mean(losses[:p])

        if avg_loss == 0:
            rsi[p] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[p] = 100.0 - 100.0 / (1.0 + rs)

        # Wilder's smoothing for subsequent values
        for i in range(p, n - 1):
            avg_gain = (avg_gain * (p - 1) + gains[i]) / p
            avg_loss = (avg_loss * (p - 1) + losses[i]) / p
            if avg_loss == 0:
                rsi[i + 1] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

        return pd.DataFrame({self.name: rsi}, index=ohlcv.index)
