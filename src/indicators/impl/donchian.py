"""Donchian Channel indicators — highest high / lowest low over a rolling window."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DonchianHigh:
    period: int
    src: str = "high"

    @property
    def name(self) -> str:
        return f"donchian_high_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.src not in ohlcv.columns:
            raise ValueError(f"Source column '{self.src}' not found in input DataFrame.")
        feature = ohlcv[self.src].rolling(window=self.period, min_periods=self.period).max()
        return feature.to_frame(name=self.name)


@dataclass(frozen=True)
class DonchianLow:
    period: int
    src: str = "low"

    @property
    def name(self) -> str:
        return f"donchian_low_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.src not in ohlcv.columns:
            raise ValueError(f"Source column '{self.src}' not found in input DataFrame.")
        feature = ohlcv[self.src].rolling(window=self.period, min_periods=self.period).min()
        return feature.to_frame(name=self.name)
