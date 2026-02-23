import numpy as np
import pandas as pd
import pytest

from src.indicators.impl.adx import ADX


def _make_trending_data(n=200):
    """Create data with a clear uptrend — ADX should be high."""
    close = np.cumsum(np.ones(n) * 0.5) + 100
    high = close + np.random.default_rng(42).uniform(0.1, 1.0, n)
    low = close - np.random.default_rng(43).uniform(0.1, 1.0, n)
    return pd.DataFrame({
        "open": close - 0.1,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n) * 1000,
    })


def _make_flat_data(n=200):
    """Create sideways/ranging data — ADX should be low."""
    rng = np.random.default_rng(44)
    close = 100 + rng.normal(0, 0.1, n).cumsum() * 0.01 + 100
    # Keep it very flat
    close = np.full(n, 100.0) + rng.uniform(-0.5, 0.5, n)
    high = close + rng.uniform(0.01, 0.3, n)
    low = close - rng.uniform(0.01, 0.3, n)
    return pd.DataFrame({
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.ones(n) * 1000,
    })


def test_adx_trending_data():
    df = _make_trending_data()
    adx = ADX(period=14)
    result = adx.compute(df)

    assert result.columns[0] == "adx_14"
    # After warmup, ADX should be high for trending data
    valid = result["adx_14"].dropna()
    assert len(valid) > 0
    assert valid.iloc[-1] > 40, f"ADX should be high for trending data, got {valid.iloc[-1]:.1f}"


def test_adx_flat_data():
    df = _make_flat_data()
    adx = ADX(period=14)
    result = adx.compute(df)

    valid = result["adx_14"].dropna()
    assert len(valid) > 0
    assert valid.iloc[-1] < 25, f"ADX should be low for flat data, got {valid.iloc[-1]:.1f}"


def test_adx_nan_warmup():
    df = _make_trending_data(100)
    adx = ADX(period=14)
    result = adx.compute(df)

    # First 2*period rows should be NaN
    assert result.iloc[:28].isna().all().all()
    # After warmup should have values
    assert result.iloc[28:].notna().all().all()


def test_adx_determinism():
    df = _make_trending_data()
    adx = ADX(period=14)
    r1 = adx.compute(df)
    r2 = adx.compute(df)
    pd.testing.assert_frame_equal(r1, r2)


def test_adx_name():
    assert ADX(period=14).name == "adx_14"
    assert ADX(period=20).name == "adx_20"


def test_adx_lookback():
    assert ADX(period=14).lookback == 28
    assert ADX(period=20).lookback == 40


def test_adx_missing_column():
    df = pd.DataFrame({"open": [1, 2, 3], "high": [2, 3, 4]})
    adx = ADX(period=2)
    with pytest.raises(ValueError, match="Column"):
        adx.compute(df)
