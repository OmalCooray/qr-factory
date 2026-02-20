from dataclasses import dataclass
import pandas as pd
import numpy as np
from ..core.interfaces import Transform

@dataclass(frozen=True)
class Diff:
    k: int = 1
    
    @property
    def name(self) -> str:
        return f"_diff_{self.k}"

    @property
    def lookback(self) -> int:
        return self.k

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.diff(self.k)

@dataclass(frozen=True)
class Lag:
    k: int
    
    @property
    def name(self) -> str:
        return f"_lag_{self.k}"

    @property
    def lookback(self) -> int:
        return self.k

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.shift(self.k)

@dataclass(frozen=True)
class Rescale:
    old_min: float
    old_max: float
    new_min: float = 0.0
    new_max: float = 1.0
    
    @property
    def name(self) -> str:
        return f"_rescale_{self.old_min}_{self.old_max}"

    @property
    def lookback(self) -> int:
        return 0

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        # Avoid division by zero
        denom = self.old_max - self.old_min
        if denom == 0:
            return X # Or handle as error? For now, return X to be safe regarding shape
        
        scaled = (X - self.old_min) / denom
        return scaled * (self.new_max - self.new_min) + self.new_min

@dataclass(frozen=True)
class ZScoreRolling:
    window: int
    
    @property
    def name(self) -> str:
        return f"_zscore_{self.window}"

    @property
    def lookback(self) -> int:
        return self.window

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        roll = X.rolling(window=self.window, min_periods=self.window)
        mean = roll.mean()
        std = roll.std()
        
        # Avoid division by zero if std is 0
        z_score = (X - mean) / std.replace(0, np.nan) 
        return z_score
