"""
src.loader — Download M5 bars from MetaTrader 5 into versioned snapshots.

Each run creates a timestamped snapshot folder under ``data/``::

    data/20260217_073600/
        XAUUSD_M5_2024.csv
        META.json

Usage::

    uv run loader
    uv run loader --symbol EURUSD --start 2010-01-01
"""

from ._config import LoaderConfig
from ._csv_writer import CsvYearlyWriter
from ._loader import Loader, main
from ._mt5_source import MT5Source
from ._protocols import DataSource, DataWriter
from ._version_log import VersionLog

__all__ = [
    "LoaderConfig",
    "DataSource",
    "DataWriter",
    "MT5Source",
    "CsvYearlyWriter",
    "VersionLog",
    "Loader",
    "main",
]
