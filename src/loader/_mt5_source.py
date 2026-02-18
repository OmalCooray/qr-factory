"""Concrete DataSource backed by the MetaTrader5 Python package."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from ._config import LoaderConfig
from ._protocols import EXPECTED_COLS

log = logging.getLogger(__name__)


class MT5Source:
    """Connect to a running MT5 terminal and fetch bar data."""

    def __init__(self) -> None:
        self._mt5 = None  # lazy import

    # -- lazy import so the package loads on any OS ------------------------

    def _lib(self):
        if self._mt5 is None:
            import MetaTrader5 as _mt5
            self._mt5 = _mt5
        return self._mt5

    # -- DataSource interface ----------------------------------------------

    def connect(self, cfg: LoaderConfig) -> None:
        mt5 = self._lib()
        kwargs: dict = {}
        if cfg.mt5_path:
            kwargs["path"] = cfg.mt5_path
        if cfg.mt5_login:
            kwargs["login"] = cfg.mt5_login
        if cfg.mt5_server:
            kwargs["server"] = cfg.mt5_server
        if cfg.mt5_password:
            kwargs["password"] = cfg.mt5_password

        last_err = None
        for attempt in range(1, cfg.init_retries + 1):
            if mt5.initialize(**kwargs):
                break
            last_err = mt5.last_error()
            code = last_err[0] if last_err else None
            if code == -6 and attempt < cfg.init_retries:
                log.warning(
                    "MT5 not yet authorised (attempt %d/%d), retrying in %.0fs ...",
                    attempt, cfg.init_retries, cfg.init_retry_delay,
                )
                time.sleep(cfg.init_retry_delay)
            else:
                raise RuntimeError(
                    f"MT5 initialize() failed after {attempt} attempt(s): {last_err}\n"
                    "Make sure the MetaTrader 5 terminal is running and logged in."
                )

        if not mt5.symbol_select(cfg.symbol, True):
            mt5.shutdown()
            raise RuntimeError(f"Failed to select symbol: {cfg.symbol}")

        log.info("Connected to MT5")

    def disconnect(self) -> None:
        self._lib().shutdown()
        log.info("Disconnected from MT5")

    def broker_name(self) -> str:
        info = self._lib().account_info()
        if info is None:
            return "unknown"
        return info.server or info.company or "unknown"

    def find_first_bar(
        self,
        symbol: str,
        start: datetime,
        timeframe: int,
        max_advances: int = 365,
        pause: float = 0.2,
    ) -> datetime | None:
        mt5 = self._lib()
        probe = start
        step = timedelta(days=7)

        for _ in range(max_advances):
            rates = mt5.copy_rates_from(symbol, timeframe, probe, 1)
            if rates is None:
                raise RuntimeError(f"copy_rates_from failed: {mt5.last_error()}")
            if len(rates) > 0:
                return pd.to_datetime(
                    rates[0]["time"], unit="s", utc=True,
                ).to_pydatetime()
            probe += step
            time.sleep(pause)

        return None

    def fetch_range(
        self,
        symbol: str,
        timeframe: int,
        start: datetime,
        end: datetime,
    ) -> pd.DataFrame:
        mt5 = self._lib()
        rates = mt5.copy_rates_range(symbol, timeframe, start, end)
        if rates is None:
            raise RuntimeError(f"copy_rates_range failed: {mt5.last_error()}")
        if len(rates) == 0:
            return pd.DataFrame(columns=EXPECTED_COLS)

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df
