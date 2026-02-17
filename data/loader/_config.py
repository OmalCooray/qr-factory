"""Immutable configuration for a single download run."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class LoaderConfig:
    """Immutable bag of settings for a single download run."""

    symbol: str = "XAUUSD"
    timeframe: int = 5  # MT5 M5 numeric value
    start: str = "2005-01-01"
    end: str = ""
    days_per_chunk: int = 180
    pause_sec: float = 0.2
    max_empty_advances: int = 365
    init_retries: int = 15
    init_retry_delay: float = 2.0
    mt5_path: str | None = None
    mt5_login: int | None = None
    mt5_server: str | None = None
    mt5_password: str | None = None

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> LoaderConfig:
        """Build a config from command-line arguments."""
        yesterday = (
            datetime.now(timezone.utc) - timedelta(days=1)
        ).strftime("%Y-%m-%d")

        p = argparse.ArgumentParser(
            description="Download M5 bars from MetaTrader 5 into yearly CSVs.",
        )
        p.add_argument("--symbol",   default="XAUUSD",    help="MT5 symbol (default: %(default)s)")
        p.add_argument("--start",    default="2005-01-01", help="Start date YYYY-MM-DD (default: %(default)s)")
        p.add_argument("--end",      default=yesterday,    help="End date YYYY-MM-DD (default: %(default)s)")
        p.add_argument("--path",     default=None,         help="Path to terminal64.exe")
        p.add_argument("--login",    default=None, type=int, help="MT5 login id")
        p.add_argument("--server",   default=None,         help="MT5 server name")
        p.add_argument("--password", default=None,         help="MT5 password")

        args = p.parse_args(argv)
        return cls(
            symbol=args.symbol,
            start=args.start,
            end=args.end,
            mt5_path=args.path,
            mt5_login=args.login,
            mt5_server=args.server,
            mt5_password=args.password,
        )

    @property
    def start_dt(self) -> datetime:
        return datetime.strptime(self.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    @property
    def end_dt(self) -> datetime:
        return datetime.strptime(self.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
