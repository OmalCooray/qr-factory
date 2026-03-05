"""Trading session one-hot indicator for XAUUSD (UTC-based).

Sessions (UTC hours):
- Asian:              00:00 - 07:59  (hours 0-7)
- London pre-overlap: 08:00 - 12:59  (hours 8-12)
- London-NY overlap:  13:00 - 16:59  (hours 13-16)
- New York late:      17:00 - 23:59  (hours 17-23)
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TradingSession:
    """One-hot encode the trading session from the bar timestamp.

    Returns a DataFrame with 4 binary columns:
    ``session_asian``, ``session_london``, ``session_overlap``, ``session_newyork``.
    """

    @property
    def name(self) -> str:
        return "session"

    @property
    def lookback(self) -> int:
        return 0

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        if "time" not in ohlcv.columns:
            raise ValueError("Column 'time' not found in input DataFrame.")

        hour = ohlcv["time"].dt.hour

        return pd.DataFrame(
            {
                "session_asian": (hour < 8).astype(float),
                "session_london": ((hour >= 8) & (hour < 13)).astype(float),
                "session_overlap": ((hour >= 13) & (hour < 17)).astype(float),
                "session_newyork": ((hour >= 17)).astype(float),
            },
            index=ohlcv.index,
        )
