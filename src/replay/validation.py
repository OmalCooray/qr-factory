"""Fail-fast data-integrity checks for bar DataFrames."""

from __future__ import annotations

import pandas as pd


def validate_bars(df: pd.DataFrame) -> None:
    """Validate a raw bar DataFrame *before* any processing.

    Raises ``ValueError`` immediately on the first problem found so that
    corrupt / malformed data never silently enters the replay loop.
    """

    # 1. Timestamp column exists with no nulls ──────────────────────────
    if "time" not in df.columns:
        raise ValueError("Missing 'time' column")
    if df["time"].isna().any():
        n = int(df["time"].isna().sum())
        raise ValueError(f"Null timestamps found: {n} rows")

    # 2. Strictly increasing time ───────────────────────────────────────
    times = pd.to_datetime(df["time"], utc=True)

    n_dupes = int(times.duplicated().sum())
    if n_dupes > 0:
        raise ValueError(f"Duplicate timestamps found: {n_dupes}")

    if not times.is_monotonic_increasing:
        raise ValueError("Timestamps not monotonic increasing")

    # 3. No NaNs in OHLC fields ────────────────────────────────────────
    ohlc = ["open", "high", "low", "close"]
    present = [c for c in ohlc if c in df.columns]
    if present:
        na_cols = [c for c in present if df[c].isna().any()]
        if na_cols:
            raise ValueError(f"NaN values in {na_cols}")

    # 4. Spread sanity ─────────────────────────────────────────────────
    if "spread" in df.columns:
        neg = int((df["spread"] < 0).sum())
        if neg > 0:
            raise ValueError(f"Negative spread found: {neg} rows")
