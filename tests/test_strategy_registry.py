"""Tests for the strategy registry / factory pattern."""

import pytest

from src.strategy.registry import build_strategy
from src.strategy.ma_crossover import MACrossoverStrategy
from src.strategy.adx_filtered_crossover import ADXFilteredCrossoverStrategy


def test_build_ma_crossover_sma():
    """SMA variant: correct strategy type, 2 specs, correct column names, warmup."""
    strategy, specs = build_strategy({
        "type": "ma_crossover",
        "fast_period": 10,
        "slow_period": 30,
        "indicator_type": "sma",
    })

    assert isinstance(strategy, MACrossoverStrategy)
    assert len(specs) == 2
    assert strategy.fast_col == "sma_10_close"
    assert strategy.slow_col == "sma_30_close"
    assert strategy.warmup_bars == 30


def test_build_ma_crossover_ema():
    """EMA variant: correct column names."""
    strategy, specs = build_strategy({
        "type": "ma_crossover",
        "fast_period": 50,
        "slow_period": 200,
        "indicator_type": "ema",
    })

    assert isinstance(strategy, MACrossoverStrategy)
    assert len(specs) == 2
    assert strategy.fast_col == "ema_50_close"
    assert strategy.slow_col == "ema_200_close"
    assert strategy.warmup_bars == 200


def test_build_adx_filtered():
    """ADX filtered: 3 specs, ADX column, warmup = max(slow, 2*adx)."""
    strategy, specs = build_strategy({
        "type": "adx_filtered",
        "fast_period": 10,
        "slow_period": 30,
        "indicator_type": "sma",
        "adx_period": 14,
        "adx_threshold": 25.0,
    })

    assert isinstance(strategy, ADXFilteredCrossoverStrategy)
    assert len(specs) == 3
    assert strategy.fast_col == "sma_10_close"
    assert strategy.slow_col == "sma_30_close"
    assert strategy.adx_col == "adx_14"
    assert strategy.adx_threshold == 25.0
    # warmup = max(30, 2*14) = 30
    assert strategy.warmup_bars == 30


def test_build_adx_filtered_large_adx_period():
    """When adx_period is large, warmup = 2 * adx_period."""
    strategy, specs = build_strategy({
        "type": "adx_filtered",
        "fast_period": 10,
        "slow_period": 20,
        "adx_period": 30,
    })

    # warmup = max(20, 2*30) = 60
    assert strategy.warmup_bars == 60


def test_build_unknown_raises():
    """Unknown strategy type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy type"):
        build_strategy({"type": "nonexistent"})


def test_build_missing_type_raises():
    """Missing 'type' key raises ValueError."""
    with pytest.raises(ValueError, match="must contain a 'type' key"):
        build_strategy({"fast_period": 10})
