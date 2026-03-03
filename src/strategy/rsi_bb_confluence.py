"""RSI + Bollinger Band confluence strategy with confirmation.

Translated from a TradingView Pine Script indicator.

Entry logic:
  - Raw Long:  RSI in [buy_rsi_min, buy_rsi_max] AND low touches lower BB
  - Raw Short: RSI in [sell_rsi_min, sell_rsi_max] AND high touches upper BB

Confirmation (fires on seed bar + look_ahead bars later):
  - Long:  bullish candle (close > open AND close > prev_close)
           AND optionally RSI rising (rsi > prev_rsi)
  - Short: bearish candle (close < open AND close < prev_close)
           AND optionally RSI falling (rsi < prev_rsi)

Exit: ATR-based stop-loss and take-profit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.indicators.core.pipeline import FeatureSpec
from src.strategy.signal import Signal
from .base import StrategyContext, validate_features


@dataclass
class RSIBBConfluenceStrategy:
    """RSI + BB confluence with candle/RSI-slope confirmation."""

    rsi_col: str          # RSI lag 1 (previous bar)
    rsi_prev_col: str     # RSI lag 2 (two bars ago, for slope)
    bb_upper_col: str     # BB upper lag 1
    bb_lower_col: str     # BB lower lag 1
    atr_col: str          # ATR lag 1

    buy_rsi_min: float = 30.0
    buy_rsi_max: float = 35.0
    sell_rsi_min: float = 65.0
    sell_rsi_max: float = 70.0

    need_rsi_slope: bool = True
    look_ahead: int = 0

    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0

    _name: str = "rsi_bb_confluence"
    _warmup_bars: int = 0

    # ── Mutable internal state ──
    _in_position_side: float = field(default=0.0, init=False, repr=False)
    _stop_level: float = field(default=0.0, init=False, repr=False)
    _tp_level: float = field(default=0.0, init=False, repr=False)
    _long_seed_bar: int = field(default=-1, init=False, repr=False)
    _short_seed_bar: int = field(default=-1, init=False, repr=False)
    _prev_close: float = field(default=0.0, init=False, repr=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> Sequence[str]:
        return [self.rsi_col, self.rsi_prev_col, self.bb_upper_col, self.bb_lower_col, self.atr_col]

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        if ctx.bar_index < self.warmup_bars:
            return False
        return validate_features(ctx.features, self.required_features)

    def on_bar(self, ctx: StrategyContext) -> Signal:
        close = float(ctx.bar["close"])
        bar_open = float(ctx.bar["open"])
        bar_high = float(ctx.bar["high"])
        bar_low = float(ctx.bar["low"])

        # Update prev_close tracking (need it for confirmation candle)
        prev_close = self._prev_close
        self._prev_close = close

        if not self.can_trade(ctx):
            return Signal(direction=0.0, strength=0.0, reason="warmup/feature_nan")

        # Sync internal side tracker with engine position
        if ctx.position == 0.0:
            self._in_position_side = 0.0
            self._stop_level = 0.0
            self._tp_level = 0.0

        rsi = float(ctx.features[self.rsi_col])
        rsi_prev = float(ctx.features[self.rsi_prev_col])
        bb_upper = float(ctx.features[self.bb_upper_col])
        bb_lower = float(ctx.features[self.bb_lower_col])
        atr = float(ctx.features[self.atr_col])

        # ── If in position: check SL / TP ──
        if self._in_position_side == 1.0:
            if bar_low <= self._stop_level:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="stop_loss")
            if bar_high >= self._tp_level:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="take_profit")
            return Signal(direction=1.0, strength=1.0, reason="hold_long")

        if self._in_position_side == -1.0:
            if bar_high >= self._stop_level:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="stop_loss")
            if bar_low <= self._tp_level:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="take_profit")
            return Signal(direction=-1.0, strength=1.0, reason="hold_short")

        # ── Flat: check for raw triggers (seed bars) ──
        # RSI zone + BB touch → set seed
        touch_lower = bar_low <= bb_lower
        touch_upper = bar_high >= bb_upper
        rsi_in_buy_zone = rsi >= self.buy_rsi_min and rsi <= self.buy_rsi_max
        rsi_in_sell_zone = rsi >= self.sell_rsi_min and rsi <= self.sell_rsi_max

        if rsi_in_buy_zone and touch_lower:
            self._long_seed_bar = ctx.bar_index
        if rsi_in_sell_zone and touch_upper:
            self._short_seed_bar = ctx.bar_index

        # ── Check confirmation on seed bars ──
        # Bullish candle: close > open AND close > prev_close
        bull_candle = close > bar_open and prev_close > 0 and close > prev_close
        # Bearish candle: close < open AND close < prev_close
        bear_candle = close < bar_open and prev_close > 0 and close < prev_close
        # RSI momentum
        momentum_up = rsi > rsi_prev
        momentum_down = rsi < rsi_prev

        # Long confirmation
        if self._long_seed_bar >= 0:
            bars_since_seed = ctx.bar_index - self._long_seed_bar
            if bars_since_seed >= self.look_ahead:
                if bull_candle and (not self.need_rsi_slope or momentum_up):
                    self._long_seed_bar = -1
                    self._in_position_side = 1.0
                    self._stop_level = close - self.sl_atr_mult * atr
                    self._tp_level = close + self.tp_atr_mult * atr
                    return Signal(
                        direction=1.0,
                        strength=1.0,
                        reason=f"long_confirmed|rsi={rsi:.1f}|atr_sl={self._stop_level:.2f}|atr_tp={self._tp_level:.2f}",
                    )

        # Short confirmation
        if self._short_seed_bar >= 0:
            bars_since_seed = ctx.bar_index - self._short_seed_bar
            if bars_since_seed >= self.look_ahead:
                if bear_candle and (not self.need_rsi_slope or momentum_down):
                    self._short_seed_bar = -1
                    self._in_position_side = -1.0
                    self._stop_level = close + self.sl_atr_mult * atr
                    self._tp_level = close - self.tp_atr_mult * atr
                    return Signal(
                        direction=-1.0,
                        strength=1.0,
                        reason=f"short_confirmed|rsi={rsi:.1f}|atr_sl={self._stop_level:.2f}|atr_tp={self._tp_level:.2f}",
                    )

        return Signal(direction=0.0, strength=0.0, reason="no_signal")


def build(params: dict) -> tuple[RSIBBConfluenceStrategy, list[FeatureSpec]]:
    """Build RSIBBConfluenceStrategy and its FeatureSpecs from config."""
    from src.indicators.impl.rsi import RSI
    from src.indicators.impl.bb import BollingerUpper, BollingerLower
    from src.indicators.impl.atr import ATR
    from src.indicators.impl.transforms import Lag

    rsi_period: int = params.get("rsi_period", 14)
    bb_period: int = params.get("bb_period", 20)
    bb_mult: float = params.get("bb_mult", 2.0)
    atr_period: int = params.get("atr_period", 14)

    # Indicators
    rsi_ind = RSI(period=rsi_period)
    bb_upper = BollingerUpper(period=bb_period, mult=bb_mult)
    bb_lower = BollingerLower(period=bb_period, mult=bb_mult)
    atr_ind = ATR(period=atr_period)
    lag1 = Lag(1)
    lag2 = Lag(2)

    specs = [
        FeatureSpec(rsi_ind, [lag1]),       # previous bar's RSI
        FeatureSpec(rsi_ind, [lag2]),       # 2 bars ago RSI (for slope)
        FeatureSpec(bb_upper, [lag1]),      # previous bar's upper BB
        FeatureSpec(bb_lower, [lag1]),      # previous bar's lower BB
        FeatureSpec(atr_ind, [lag1]),       # previous bar's ATR
    ]

    # Column names
    rsi_col = f"{rsi_ind.name}{lag1.name}"          # rsi_14_lag_1
    rsi_prev_col = f"{rsi_ind.name}{lag2.name}"     # rsi_14_lag_2
    bb_upper_col = f"{bb_upper.name}{lag1.name}"    # bb_upper_20_lag_1
    bb_lower_col = f"{bb_lower.name}{lag1.name}"    # bb_lower_20_lag_1
    atr_col = f"{atr_ind.name}{lag1.name}"          # atr_14_lag_1

    warmup = max(rsi_period, bb_period, atr_period) + 3  # +3 for lags and safety

    strategy = RSIBBConfluenceStrategy(
        rsi_col=rsi_col,
        rsi_prev_col=rsi_prev_col,
        bb_upper_col=bb_upper_col,
        bb_lower_col=bb_lower_col,
        atr_col=atr_col,
        buy_rsi_min=params.get("buy_rsi_min", 30.0),
        buy_rsi_max=params.get("buy_rsi_max", 35.0),
        sell_rsi_min=params.get("sell_rsi_min", 65.0),
        sell_rsi_max=params.get("sell_rsi_max", 70.0),
        need_rsi_slope=params.get("need_rsi_slope", True),
        look_ahead=params.get("look_ahead", 0),
        sl_atr_mult=params.get("sl_atr_mult", 1.5),
        tp_atr_mult=params.get("tp_atr_mult", 3.0),
        _warmup_bars=warmup,
    )

    return strategy, specs
