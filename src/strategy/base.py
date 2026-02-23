from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable
import pandas as pd

from src.strategy.signal import Signal

@dataclass(frozen=True)
class StrategyContext:
    """
    Represents information available throughout the strategy at each bar.
    """
    ts: pd.Timestamp
    bar: pd.Series            # OHLCV row with open/high/low/close/volume
    features: pd.Series       # feature row at ts (computed elsewhere)
    position: float           # current signed position
    equity: float             # current equity (float)
    bar_index: int            # sequential bar counter

@runtime_checkable
class Strategy(Protocol):
    """
    Protocol for a trading strategy.
    """
    @property
    def name(self) -> str: ...

    @property
    def required_features(self) -> Sequence[str]: ...

    @property
    def warmup_bars(self) -> int: ...

    def can_trade(self, ctx: StrategyContext) -> bool:
        """
        Gating logic for warmup / missing features / NaNs.
        """
        ...

    def on_bar(self, ctx: StrategyContext) -> Signal:
        """
        Produces a Signal for the current bar.

        The decision layer converts Signal → OrderIntent based on risk and sizing.
        """
        ...

def validate_features(features: pd.Series, required: Sequence[str]) -> bool:
    """
    Helper function to validate required features exist and are not NaN.
    """
    for feature in required:
        if feature not in features:
            return False
        if pd.isna(features[feature]):
            return False
    return True
