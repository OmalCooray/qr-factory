from dataclasses import dataclass
import pandas as pd
from ..core.interfaces import Indicator

@dataclass(frozen=True)
class SMA:
    period: int
    src: str = "close"

    @property
    def name(self) -> str:
        return f"sma_{self.period}_{self.src}"

    @property
    def lookback(self) -> int:
        return self.period

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the Simple Moving Average (SMA).
        
        Args:
            ohlcv: A DataFrame containing OHLCV data.
            
        Returns:
            A DataFrame with the computed SMA values.
        """
        # Validate existence of source column
        if self.src not in ohlcv.columns:
             raise ValueError(f"Source column '{self.src}' not found in input DataFrame.")

        # Compute SMA
        feature = ohlcv[self.src].rolling(window=self.period, min_periods=self.period).mean()
        
        # Return as a DataFrame to match interface
        return feature.to_frame(name=self.name)
