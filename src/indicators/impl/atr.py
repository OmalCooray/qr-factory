"""Average True Range (ATR) indicator."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ATR:
    period: int = 14

    @property
    def name(self) -> str:
        return f"atr_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period + 1  # one extra bar for prev_close

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in ohlcv.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")

        high = ohlcv["high"].values.astype(np.float64)
        low = ohlcv["low"].values.astype(np.float64)
        close = ohlcv["close"].values.astype(np.float64)
        n = len(ohlcv)

        tr = np.full(n, np.nan, dtype=np.float64)
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        tr_series = pd.Series(tr, index=ohlcv.index)
        atr = tr_series.rolling(window=self.period, min_periods=self.period).mean()
        return atr.to_frame(name=self.name)
