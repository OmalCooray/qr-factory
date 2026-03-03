"""CCI — Commodity Channel Index."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CCI:
    period: int = 20

    @property
    def name(self) -> str:
        return f"cci_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        for col in ("high", "low", "close"):
            if col not in ohlcv.columns:
                raise ValueError(f"Column '{col}' not found in input DataFrame.")

        tp = (ohlcv["high"] + ohlcv["low"] + ohlcv["close"]) / 3.0
        tp_sma = tp.rolling(window=self.period, min_periods=self.period).mean()
        # Mean absolute deviation
        mad = tp.rolling(window=self.period, min_periods=self.period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        cci = (tp - tp_sma) / (0.015 * mad)

        return pd.DataFrame({self.name: cci.values}, index=ohlcv.index)
