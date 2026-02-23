import pandas as pd
import pytest

from src.strategy.adx_filtered_crossover import ADXFilteredCrossoverStrategy
from src.strategy.base import StrategyContext


@pytest.fixture
def strategy():
    return ADXFilteredCrossoverStrategy(
        fast_col="ema_fast",
        slow_col="ema_slow",
        adx_col="adx_14",
        adx_threshold=25.0,
        _warmup_bars=28,
    )


def _make_ctx(fast, slow, adx, bar_idx=50, position=0.0):
    return StrategyContext(
        ts=pd.Timestamp("2023-01-01"),
        bar=pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}),
        features=pd.Series({"ema_fast": fast, "ema_slow": slow, "adx_14": adx}),
        position=position,
        equity=100000.0,
        bar_index=bar_idx,
    )


def test_adx_below_threshold_returns_flat(strategy):
    ctx = _make_ctx(fast=105, slow=100, adx=20.0)
    signal = strategy.on_bar(ctx)
    assert signal.direction == 0.0
    assert signal.reason == "adx_below_threshold"


def test_adx_below_threshold_flattens_position(strategy):
    ctx = _make_ctx(fast=105, slow=100, adx=15.0, position=1.0)
    signal = strategy.on_bar(ctx)
    assert signal.direction == 0.0
    assert signal.reason == "adx_below_threshold"


def test_adx_above_threshold_long(strategy):
    ctx = _make_ctx(fast=105, slow=100, adx=30.0)
    signal = strategy.on_bar(ctx)
    assert signal.direction == 1.0
    assert signal.strength == 1.0
    assert signal.reason == "cross_above"


def test_adx_above_threshold_short(strategy):
    ctx = _make_ctx(fast=95, slow=100, adx=30.0)
    signal = strategy.on_bar(ctx)
    assert signal.direction == -1.0
    assert signal.strength == 1.0
    assert signal.reason == "cross_below"


def test_adx_above_threshold_equal(strategy):
    ctx = _make_ctx(fast=100, slow=100, adx=30.0)
    signal = strategy.on_bar(ctx)
    assert signal.direction == 0.0
    assert signal.reason == "equal"


def test_warmup_blocked(strategy):
    ctx = _make_ctx(fast=105, slow=100, adx=30.0, bar_idx=5)
    signal = strategy.on_bar(ctx)
    assert signal.direction == 0.0
    assert "warmup" in signal.reason


def test_nan_features_blocked(strategy):
    ctx = _make_ctx(fast=float("nan"), slow=100, adx=30.0)
    signal = strategy.on_bar(ctx)
    assert signal.direction == 0.0


def test_determinism(strategy):
    ctx = _make_ctx(fast=105, slow=100, adx=30.0)
    d1 = strategy.on_bar(ctx)
    d2 = strategy.on_bar(ctx)
    assert d1 == d2
