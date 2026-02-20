from typing import Protocol, runtime_checkable
import pandas as pd

@runtime_checkable
class Indicator(Protocol):
    name: str
    lookback: int

    def compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the indicator values.
        
        Args:
            ohlcv: A DataFrame containing OHLCV data.
            
        Returns:
            A DataFrame with the computed indicator values. The index must match the input index.
        """
        ...

@runtime_checkable
class Transform(Protocol):
    name: str
    lookback: int

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Applies a transformation to the input DataFrame.
        
        Args:
            X: Input DataFrame (feature matrix).
            
        Returns:
            Transformed DataFrame. Index and row count must be preserved.
        """
        ...

def validate_ohlcv(df: pd.DataFrame) -> None:
    """
    Validates that the input DataFrame contains the required OHLCV columns.
    
    Args:
        df: Input DataFrame.
        
    Raises:
        ValueError: If required columns are missing.
    """
    required_columns = {"open", "high", "low", "close", "volume"}
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Input DataFrame missing required columns: {missing}")
