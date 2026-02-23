import pytest
import pandas as pd
from src.strategy.base import StrategyContext
from src.strategy.signal import Signal
from src.strategy.ma_crossover import MACrossoverStrategy

@pytest.fixture
def strategy():
    # Construct strategy with 2 warmup bars
    return MACrossoverStrategy(
        fast_col="sma_fast",
        slow_col="sma_slow",
        _warmup_bars=2
    )

def make_context(ts_str, fast, slow, bar_idx=10) -> StrategyContext:
    ts = pd.Timestamp(ts_str)
    # Dummy bar data
    bar = pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000})
    # Features
    features = pd.Series({"sma_fast": fast, "sma_slow": slow})

    return StrategyContext(
        ts=ts,
        bar=bar,
        features=features,
        position=0.0,
        equity=10000.0,
        bar_index=bar_idx
    )

def test_ma_crossover_logic(strategy):
    # Test 1: Warmup period (bar_index < warmup_bars)
    # warmup_bars=2. bar_index=1 should fail.
    ctx_warmup = make_context("2023-01-01", 100, 100, bar_idx=1)
    signal = strategy.on_bar(ctx_warmup)
    assert signal.direction == 0.0
    assert "warmup" in signal.reason or "feature_nan" in signal.reason

    # Test 2: NaN features (even after warmup)
    ctx_nan = make_context("2023-01-02", float("nan"), 100, bar_idx=10)
    signal = strategy.on_bar(ctx_nan)
    assert signal.direction == 0.0
    assert "feature_nan" in signal.reason or "warmup" in signal.reason

    # Test 3: Fast > Slow (Long)
    ctx_long = make_context("2023-01-03", 105.0, 100.0, bar_idx=10)
    signal_long = strategy.on_bar(ctx_long)
    assert signal_long.direction == 1.0
    assert signal_long.strength == 1.0
    assert signal_long.reason == "cross_above"

    # Test 4: Fast < Slow (Short)
    ctx_short = make_context("2023-01-04", 95.0, 100.0, bar_idx=11)
    signal_short = strategy.on_bar(ctx_short)
    assert signal_short.direction == -1.0
    assert signal_short.strength == 1.0
    assert signal_short.reason == "cross_below"

    # Test 5: Equal (Flat)
    ctx_equal = make_context("2023-01-05", 100.0, 100.0, bar_idx=12)
    signal_equal = strategy.on_bar(ctx_equal)
    assert signal_equal.direction == 0.0
    assert signal_equal.reason == "equal"

def test_missing_features(strategy):
    ctx_missing = make_context("2023-01-06", 100, 100, bar_idx=10)
    # Remove a required feature simulates missing column
    # Need to create a new series because the helper just returns a series
    features = ctx_missing.features.drop("sma_fast")
    # Dataclasses are frozen, so replace
    from dataclasses import replace
    ctx_missing = replace(ctx_missing, features=features)

    signal = strategy.on_bar(ctx_missing)
    assert signal.direction == 0.0
    # Reason should indicate validation failure
    assert "feature_nan" in signal.reason or "warmup" in signal.reason

def test_determinism(strategy):
    ctx = make_context("2023-01-07", 105.0, 100.0, bar_idx=20)
    d1 = strategy.on_bar(ctx)
    d2 = strategy.on_bar(ctx)
    assert d1 == d2
