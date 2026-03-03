"""Bollinger Bands — Upper and Lower band indicators."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BollingerUpper:
    """Upper Bollinger Band: SMA + mult * rolling std."""

    period: int = 20
    mult: float = 2.0
    src: str = "close"

    @property
    def name(self) -> str:
        return f"bb_upper_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.src not in ohlcv.columns:
            raise ValueError(f"Source column '{self.src}' not found in input DataFrame.")
        series = ohlcv[self.src]
        basis = series.rolling(window=self.period, min_periods=self.period).mean()
        std = series.rolling(window=self.period, min_periods=self.period).std()
        upper = basis + std * self.mult
        return pd.DataFrame({self.name: upper}, index=ohlcv.index)


@dataclass(frozen=True)
class BollingerLower:
    """Lower Bollinger Band: SMA - mult * rolling std."""

    period: int = 20
    mult: float = 2.0
    src: str = "close"

    @property
    def name(self) -> str:
        return f"bb_lower_{self.period}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.src not in ohlcv.columns:
            raise ValueError(f"Source column '{self.src}' not found in input DataFrame.")
        series = ohlcv[self.src]
        basis = series.rolling(window=self.period, min_periods=self.period).mean()
        std = series.rolling(window=self.period, min_periods=self.period).std()
        lower = basis - std * self.mult
        return pd.DataFrame({self.name: lower}, index=ohlcv.index)
