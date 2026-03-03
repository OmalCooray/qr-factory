"""Tests for BBMeanReversionStrategy."""

import math

import pandas as pd
import pytest

from src.strategy.bb_mean_reversion import BBMeanReversionStrategy, build
from src.strategy.base import StrategyContext, Strategy
from src.indicators.core.pipeline import FeatureSpec


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_strategy(**overrides) -> BBMeanReversionStrategy:
    defaults = dict(
        cci_col="cci_20_lag_1",
        cci_prev_col="cci_20_lag_2",
        bb_upper_col="bb_upper_20_lag_1",
        bb_lower_col="bb_lower_20_lag_1",
        sma_col="sma_20_close_lag_1",
        atr_col="atr_14_lag_1",
        cci_entry=100.0,
        sl_atr_mult=2.0,
        _warmup_bars=23,
    )
    defaults.update(overrides)
    return BBMeanReversionStrategy(**defaults)


def _make_ctx(
    cci=0.0,
    cci_prev=0.0,
    bb_upper=2010.0,
    bb_lower=1990.0,
    sma=2000.0,
    atr=5.0,
    close=2000.0,
    high=2001.0,
    low=1999.0,
    bar_idx=50,
    position=0.0,
    adx=None,
) -> StrategyContext:
    feats = {
        "cci_20_lag_1": cci,
        "cci_20_lag_2": cci_prev,
        "bb_upper_20_lag_1": bb_upper,
        "bb_lower_20_lag_1": bb_lower,
        "sma_20_close_lag_1": sma,
        "atr_14_lag_1": atr,
    }
    if adx is not None:
        feats["adx_14_lag_1"] = adx
    return StrategyContext(
        ts=pd.Timestamp("2023-01-01"),
        bar=pd.Series({"open": close, "high": high, "low": low, "close": close, "volume": 1000}),
        features=pd.Series(feats),
        position=position,
        equity=100_000.0,
        bar_index=bar_idx,
    )


# ── Warmup / NaN blocking ───────────────────────────────────────────────────

def test_warmup_blocking():
    strat = _make_strategy()
    ctx = _make_ctx(bar_idx=5)
    sig = strat.on_bar(ctx)
    assert sig.direction == 0.0
    assert "warmup" in sig.reason


def test_nan_feature_blocking():
    strat = _make_strategy()
    ctx = _make_ctx(cci=float("nan"))
    sig = strat.on_bar(ctx)
    assert sig.direction == 0.0
    assert "warmup" in sig.reason or "nan" in sig.reason


# ── No signal when flat and no extremes ──────────────────────────────────────

def test_no_signal_when_flat_no_extremes():
    strat = _make_strategy()
    ctx = _make_ctx(cci=50.0, cci_prev=45.0)
    sig = strat.on_bar(ctx)
    assert sig.direction == 0.0
    assert sig.reason == "no_signal"


# ── Seed set but no cross-back yet ──────────────────────────────────────────

def test_seed_set_no_crossback_yet():
    strat = _make_strategy()
    # Set long seed: cci <= -100 AND close <= bb_lower
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0)
    sig = strat.on_bar(ctx_seed)
    assert sig.direction == 0.0
    assert sig.reason == "no_signal"
    assert strat._long_seed is True

    # Next bar: cci still extreme, no cross-back
    ctx_still = _make_ctx(cci=-110.0, cci_prev=-120.0, close=1989.0, high=1990.0, low=1988.0, bb_lower=1990.0)
    sig = strat.on_bar(ctx_still)
    assert sig.direction == 0.0
    assert sig.reason == "no_signal"


# ── Long entry on cross-back ────────────────────────────────────────────────

def test_long_entry_crossback():
    strat = _make_strategy()
    # Step 1: Set long seed
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0)
    strat.on_bar(ctx_seed)
    assert strat._long_seed is True

    # Step 2: Cross-back — cci_prev <= -100 AND cci > -100
    ctx_confirm = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, sma=2000.0, atr=5.0)
    sig = strat.on_bar(ctx_confirm)
    assert sig.direction == 1.0
    assert sig.strength == 1.0
    assert "long_entry" in sig.reason
    assert strat._in_position_side == 1.0
    assert strat._stop_level == 1995.0 - 2.0 * 5.0  # 1985.0
    assert strat._tp_target == 2000.0  # SMA = the mean


def test_long_full_sequence_seed_confirm_hold_tp():
    strat = _make_strategy()
    # Seed
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0)
    strat.on_bar(ctx_seed)

    # Confirm
    ctx_confirm = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, sma=2000.0, atr=5.0)
    sig = strat.on_bar(ctx_confirm)
    assert sig.direction == 1.0

    # Hold
    ctx_hold = _make_ctx(cci=-50.0, cci_prev=-90.0, close=1997.0, high=1998.0, low=1996.0, sma=2000.0, position=1.0)
    sig = strat.on_bar(ctx_hold)
    assert sig.direction == 1.0
    assert sig.reason == "hold_long"

    # TP at mean (high >= sma)
    ctx_tp = _make_ctx(cci=-20.0, cci_prev=-50.0, close=2001.0, high=2002.0, low=1999.0, sma=2000.0, position=1.0)
    sig = strat.on_bar(ctx_tp)
    assert sig.direction == 0.0
    assert sig.reason == "take_profit"


# ── Short entry on cross-back ───────────────────────────────────────────────

def test_short_entry_crossback():
    strat = _make_strategy()
    # Step 1: Set short seed — cci >= 100 AND close >= bb_upper
    ctx_seed = _make_ctx(cci=120.0, cci_prev=80.0, close=2012.0, high=2013.0, low=2011.0, bb_upper=2010.0)
    strat.on_bar(ctx_seed)
    assert strat._short_seed is True

    # Step 2: Cross-back — cci_prev >= 100 AND cci < 100
    ctx_confirm = _make_ctx(cci=90.0, cci_prev=105.0, close=2005.0, high=2006.0, low=2004.0, sma=2000.0, atr=5.0)
    sig = strat.on_bar(ctx_confirm)
    assert sig.direction == -1.0
    assert sig.strength == 1.0
    assert "short_entry" in sig.reason
    assert strat._in_position_side == -1.0
    assert strat._stop_level == 2005.0 + 2.0 * 5.0  # 2015.0
    assert strat._tp_target == 2000.0  # SMA = the mean


# ── Stop loss exit ──────────────────────────────────────────────────────────

def test_stop_loss_long():
    strat = _make_strategy()
    # Enter long
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0)
    strat.on_bar(ctx_seed)
    ctx_confirm = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, sma=2000.0, atr=5.0)
    strat.on_bar(ctx_confirm)
    # stop = 1995 - 10 = 1985

    # bar_low hits stop
    ctx_sl = _make_ctx(cci=-60.0, cci_prev=-90.0, close=1984.0, high=1986.0, low=1984.0, position=1.0)
    sig = strat.on_bar(ctx_sl)
    assert sig.direction == 0.0
    assert sig.reason == "stop_loss"


def test_stop_loss_short():
    strat = _make_strategy()
    # Enter short
    ctx_seed = _make_ctx(cci=120.0, cci_prev=80.0, close=2012.0, high=2013.0, low=2011.0, bb_upper=2010.0)
    strat.on_bar(ctx_seed)
    ctx_confirm = _make_ctx(cci=90.0, cci_prev=105.0, close=2005.0, high=2006.0, low=2004.0, sma=2000.0, atr=5.0)
    strat.on_bar(ctx_confirm)
    # stop = 2005 + 10 = 2015

    # bar_high hits stop
    ctx_sl = _make_ctx(cci=60.0, cci_prev=90.0, close=2016.0, high=2016.0, low=2014.0, position=-1.0)
    sig = strat.on_bar(ctx_sl)
    assert sig.direction == 0.0
    assert sig.reason == "stop_loss"


# ── Take profit at SMA midline ──────────────────────────────────────────────

def test_take_profit_long():
    strat = _make_strategy()
    # Enter long
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0)
    strat.on_bar(ctx_seed)
    ctx_confirm = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, sma=2000.0, atr=5.0)
    strat.on_bar(ctx_confirm)
    # tp = sma = 2000

    ctx_tp = _make_ctx(cci=-30.0, cci_prev=-60.0, close=2001.0, high=2001.0, low=1997.0, position=1.0)
    sig = strat.on_bar(ctx_tp)
    assert sig.direction == 0.0
    assert sig.reason == "take_profit"


def test_take_profit_short():
    strat = _make_strategy()
    # Enter short
    ctx_seed = _make_ctx(cci=120.0, cci_prev=80.0, close=2012.0, high=2013.0, low=2011.0, bb_upper=2010.0)
    strat.on_bar(ctx_seed)
    ctx_confirm = _make_ctx(cci=90.0, cci_prev=105.0, close=2005.0, high=2006.0, low=2004.0, sma=2000.0, atr=5.0)
    strat.on_bar(ctx_confirm)
    # tp = sma = 2000

    ctx_tp = _make_ctx(cci=30.0, cci_prev=60.0, close=1999.0, high=2002.0, low=1999.0, position=-1.0)
    sig = strat.on_bar(ctx_tp)
    assert sig.direction == 0.0
    assert sig.reason == "take_profit"


# ── Hold while in position ──────────────────────────────────────────────────

def test_hold_long():
    strat = _make_strategy()
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0)
    strat.on_bar(ctx_seed)
    ctx_confirm = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, sma=2000.0, atr=5.0)
    strat.on_bar(ctx_confirm)

    # No SL/TP hit
    ctx_hold = _make_ctx(cci=-50.0, cci_prev=-90.0, close=1997.0, high=1998.0, low=1996.0, position=1.0)
    sig = strat.on_bar(ctx_hold)
    assert sig.direction == 1.0
    assert sig.reason == "hold_long"


def test_hold_short():
    strat = _make_strategy()
    ctx_seed = _make_ctx(cci=120.0, cci_prev=80.0, close=2012.0, high=2013.0, low=2011.0, bb_upper=2010.0)
    strat.on_bar(ctx_seed)
    ctx_confirm = _make_ctx(cci=90.0, cci_prev=105.0, close=2005.0, high=2006.0, low=2004.0, sma=2000.0, atr=5.0)
    strat.on_bar(ctx_confirm)

    # No SL/TP hit
    ctx_hold = _make_ctx(cci=50.0, cci_prev=90.0, close=2003.0, high=2004.0, low=2002.0, position=-1.0)
    sig = strat.on_bar(ctx_hold)
    assert sig.direction == -1.0
    assert sig.reason == "hold_short"


# ── Position sync when engine flattens ───────────────────────────────────────

def test_position_sync_engine_flattens():
    strat = _make_strategy()
    # Enter long
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0)
    strat.on_bar(ctx_seed)
    ctx_confirm = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, sma=2000.0, atr=5.0)
    strat.on_bar(ctx_confirm)
    assert strat._in_position_side == 1.0

    # Engine flattened externally (position=0.0) — strategy syncs
    ctx_flat = _make_ctx(cci=-50.0, cci_prev=-90.0, close=1997.0, high=1998.0, low=1996.0, position=0.0)
    sig = strat.on_bar(ctx_flat)
    assert strat._in_position_side == 0.0
    assert sig.reason == "no_signal"


# ── ADX gate ────────────────────────────────────────────────────────────────

def test_adx_gate_blocks_entry_and_clears_seeds():
    strat = _make_strategy(adx_col="adx_14_lag_1", adx_max_threshold=25.0)

    # Set long seed first
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0, adx=20.0)
    strat.on_bar(ctx_seed)
    assert strat._long_seed is True

    # ADX spikes above threshold — blocks entry and clears seeds
    ctx_adx = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, adx=30.0)
    sig = strat.on_bar(ctx_adx)
    assert sig.direction == 0.0
    assert sig.reason == "adx_gate"
    assert strat._long_seed is False
    assert strat._short_seed is False


def test_adx_gate_allows_entry_when_below_threshold():
    strat = _make_strategy(adx_col="adx_14_lag_1", adx_max_threshold=25.0)

    # Set long seed with ADX below threshold
    ctx_seed = _make_ctx(cci=-120.0, cci_prev=-80.0, close=1988.0, high=1989.0, low=1987.0, bb_lower=1990.0, adx=20.0)
    strat.on_bar(ctx_seed)
    assert strat._long_seed is True

    # Cross-back with ADX still below threshold — entry allowed
    ctx_confirm = _make_ctx(cci=-90.0, cci_prev=-105.0, close=1995.0, high=1996.0, low=1994.0, sma=2000.0, atr=5.0, adx=22.0)
    sig = strat.on_bar(ctx_confirm)
    assert sig.direction == 1.0
    assert "long_entry" in sig.reason


# ── Build function ──────────────────────────────────────────────────────────

def test_build_returns_correct_types():
    params = {
        "cci_period": 20,
        "bb_period": 20,
        "bb_mult": 2.0,
        "atr_period": 14,
    }
    strategy, specs = build(params)
    assert isinstance(strategy, BBMeanReversionStrategy)
    assert isinstance(specs, list)
    assert all(isinstance(s, FeatureSpec) for s in specs)
    # 6 specs: cci lag1, cci lag2, bb_upper lag1, bb_lower lag1, sma lag1, atr lag1
    assert len(specs) == 6


def test_build_with_adx():
    params = {
        "cci_period": 20,
        "bb_period": 20,
        "bb_mult": 2.0,
        "atr_period": 14,
        "adx_period": 14,
        "adx_max_threshold": 25.0,
    }
    strategy, specs = build(params)
    assert len(specs) == 7  # + adx
    assert strategy.adx_col != ""
    assert strategy.adx_max_threshold == 25.0


def test_build_column_names():
    params = {
        "cci_period": 20,
        "bb_period": 20,
        "bb_mult": 2.0,
        "atr_period": 14,
    }
    strategy, _ = build(params)
    assert strategy.cci_col == "cci_20_lag_1"
    assert strategy.cci_prev_col == "cci_20_lag_2"
    assert strategy.bb_upper_col == "bb_upper_20_lag_1"
    assert strategy.bb_lower_col == "bb_lower_20_lag_1"
    assert strategy.sma_col == "sma_20_close_lag_1"
    assert strategy.atr_col == "atr_14_lag_1"


# ── Registry integration ───────────────────────────────────────────────────

def test_registry_integration():
    from src.strategy.registry import build_strategy

    cfg = {
        "type": "bb_mean_reversion",
        "cci_period": 20,
        "bb_period": 20,
        "bb_mult": 2.0,
        "atr_period": 14,
    }
    strategy, specs = build_strategy(cfg)
    assert isinstance(strategy, BBMeanReversionStrategy)
    assert isinstance(strategy, Strategy)


# ── Protocol compliance ─────────────────────────────────────────────────────

def test_strategy_protocol():
    strat = _make_strategy()
    assert isinstance(strat, Strategy)
