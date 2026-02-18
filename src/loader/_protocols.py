"""Protocol definitions for data sources and writers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

from ._config import LoaderConfig

EXPECTED_COLS: list[str] = [
    "time", "open", "high", "low", "close",
    "tick_volume", "spread", "real_volume",
]


@runtime_checkable
class DataSource(Protocol):
    """Abstraction over any bar-data provider (MT5, CSV replay, mock)."""

    def connect(self, cfg: LoaderConfig) -> None: ...
    def disconnect(self) -> None: ...
    def broker_name(self) -> str: ...
    def find_first_bar(
        self, symbol: str, start: datetime, timeframe: int,
        max_advances: int, pause: float,
    ) -> datetime | None: ...
    def fetch_range(
        self, symbol: str, timeframe: int,
        start: datetime, end: datetime,
    ) -> pd.DataFrame: ...


@runtime_checkable
class DataWriter(Protocol):
    """Abstraction over bar-data persistence (CSV, Parquet, etc.)."""

    def write(self, df: pd.DataFrame, symbol: str, out_dir: Path) -> None: ...
