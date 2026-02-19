"""Tests for src.replay.validation.validate_bars."""

from __future__ import annotations

import pandas as pd
import pytest

from src.replay.validation import validate_bars


# ── helpers ──────────────────────────────────────────────────────────────

def _good_df() -> pd.DataFrame:
    """Return a minimal valid bar DataFrame."""
    return pd.DataFrame({
        "time": pd.to_datetime([
            "2024-01-01 00:00:00+00:00",
            "2024-01-01 00:05:00+00:00",
            "2024-01-01 00:10:00+00:00",
        ]),
        "open":  [100.0, 101.0, 102.0],
        "high":  [101.0, 102.0, 103.0],
        "low":   [99.0,  100.0, 101.0],
        "close": [100.5, 101.5, 102.5],
        "spread": [2, 3, 2],
    })


# ── happy path ───────────────────────────────────────────────────────────

def test_valid_df_passes():
    validate_bars(_good_df())   # should not raise


def test_no_spread_column_passes():
    df = _good_df().drop(columns=["spread"])
    validate_bars(df)           # spread check is optional


# ── timestamp checks ─────────────────────────────────────────────────────

def test_missing_time_column():
    df = _good_df().drop(columns=["time"])
    with pytest.raises(ValueError, match="Missing 'time' column"):
        validate_bars(df)


def test_null_timestamps():
    df = _good_df()
    df.loc[1, "time"] = pd.NaT
    with pytest.raises(ValueError, match="Null timestamps found: 1"):
        validate_bars(df)


def test_duplicate_timestamps():
    df = _good_df()
    df.loc[2, "time"] = df.loc[1, "time"]
    with pytest.raises(ValueError, match="Duplicate timestamps found: 1"):
        validate_bars(df)


def test_non_monotonic_timestamps():
    df = _good_df()
    # Swap rows 0 and 2 so timestamps go 10:00, 05:00, 00:00
    df.loc[0, "time"], df.loc[2, "time"] = df.loc[2, "time"], df.loc[0, "time"]
    with pytest.raises(ValueError, match="Timestamps not monotonic increasing"):
        validate_bars(df)


# ── OHLC NaN checks ─────────────────────────────────────────────────────

def test_nan_in_close():
    df = _good_df()
    df.loc[0, "close"] = float("nan")
    with pytest.raises(ValueError, match="NaN values in"):
        validate_bars(df)


def test_nan_in_open():
    df = _good_df()
    df.loc[1, "open"] = float("nan")
    with pytest.raises(ValueError, match="NaN values in"):
        validate_bars(df)


# ── spread checks ────────────────────────────────────────────────────────

def test_negative_spread():
    df = _good_df()
    df.loc[0, "spread"] = -1
    with pytest.raises(ValueError, match="Negative spread found: 1"):
        validate_bars(df)
