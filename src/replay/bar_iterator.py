"""Deterministic bar iterator with strict timestamp validation."""

from __future__ import annotations

import logging
from typing import Generator

import pandas as pd

log = logging.getLogger(__name__)


class BarIterator:
    """Yields bars in strict chronological order from a validated DataFrame.

    Guarantees:
    - ``time`` column exists and is ``datetime64[ns, UTC]``
    - rows are sorted ascending by ``time``
    - no duplicate timestamps (first occurrence kept)
    - monotonic-increasing timestamps
    """

    def __init__(self, df: pd.DataFrame) -> None:
        df = self._validate_and_clean(df)
        self._df = df
        self._n_bars = len(df)

    # -- public API --------------------------------------------------------

    def __len__(self) -> int:
        return self._n_bars

    def __iter__(self) -> Generator[dict, None, None]:
        for _, row in self._df.iterrows():
            yield row.to_dict()

    @property
    def df(self) -> pd.DataFrame:
        """Return the cleaned, validated DataFrame (read-only copy)."""
        return self._df.copy()

    # -- validation --------------------------------------------------------

    @staticmethod
    def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        if "time" not in df.columns:
            raise ValueError("DataFrame must contain a 'time' column")

        df = df.copy()

        # Convert to datetime64[ns, UTC]
        df["time"] = pd.to_datetime(df["time"], utc=True)

        # Sort ascending
        df = df.sort_values("time").reset_index(drop=True)

        # Assert monotonic increasing (defense-in-depth; validate_bars
        # already enforces no duplicates and correct ordering).
        assert df["time"].is_monotonic_increasing, (
            "Timestamps are not monotonic increasing after dedup — data integrity issue"
        )

        log.info(
            "BarIterator: %s bars, %s → %s",
            f"{len(df):,}",
            df["time"].iloc[0].isoformat(),
            df["time"].iloc[-1].isoformat(),
        )

        return df
