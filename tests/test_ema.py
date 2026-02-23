import numpy as np
import pandas as pd
import pytest

from src.indicators.impl.ema import EMA


@pytest.fixture
def ohlcv():
    n = 50
    close = np.linspace(100, 150, n)
    return pd.DataFrame({
        "open": close - 0.5,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.ones(n) * 1000,
    })


def test_ema_basic_computation(ohlcv):
    ema = EMA(period=10)
    result = ema.compute(ohlcv)

    assert result.shape == (len(ohlcv), 1)
    assert result.columns[0] == "ema_10_close"

    # First 9 rows should be NaN (min_periods=10)
    assert result.iloc[:9].isna().all().all()
    # Row 10 onward should have values
    assert result.iloc[9:].notna().all().all()


def test_ema_warmup_nans(ohlcv):
    ema = EMA(period=20)
    result = ema.compute(ohlcv)

    assert result.iloc[:19].isna().all().all()
    assert result.iloc[19:].notna().all().all()


def test_ema_determinism(ohlcv):
    ema = EMA(period=10)
    r1 = ema.compute(ohlcv)
    r2 = ema.compute(ohlcv)
    pd.testing.assert_frame_equal(r1, r2)


def test_ema_name():
    assert EMA(period=10).name == "ema_10_close"
    assert EMA(period=50, src="high").name == "ema_50_high"


def test_ema_lookback():
    assert EMA(period=10).lookback == 10
    assert EMA(period=50).lookback == 50


def test_ema_missing_column():
    df = pd.DataFrame({"open": [1, 2, 3]})
    ema = EMA(period=2)
    with pytest.raises(ValueError, match="Source column"):
        ema.compute(df)
