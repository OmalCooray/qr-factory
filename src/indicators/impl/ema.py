from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class EMA:
    period: int
    src: str = "close"

    @property
    def name(self) -> str:
        return f"ema_{self.period}_{self.src}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if self.src not in ohlcv.columns:
            raise ValueError(f"Source column '{self.src}' not found in input DataFrame.")

        feature = ohlcv[self.src].ewm(
            span=self.period, min_periods=self.period, adjust=False
        ).mean()

        return feature.to_frame(name=self.name)
