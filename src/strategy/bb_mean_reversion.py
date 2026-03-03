"""BB Mean Reversion with CCI Confirmation strategy.

Mean reversion strategy using CCI as the primary oscillator,
Bollinger Bands as the volatility envelope, and SMA midline as
the take-profit target (reversion to the mean).

Entry logic (two-step: seed -> cross-back):
  1. Seed: CCI at extreme AND close at BB band
     - Long seed:  cci <= -cci_entry AND close <= bb_lower
     - Short seed: cci >= cci_entry AND close >= bb_upper
  2. Cross-back confirmation:
     - Long entry:  _long_seed AND cci_prev <= -cci_entry AND cci > -cci_entry
     - Short entry: _short_seed AND cci_prev >= cci_entry AND cci < cci_entry
  3. On entry: stop = close -/+ sl_atr_mult * atr, TP = sma (the mean)

Exit: ATR-based stop-loss or take-profit at SMA midline.
Optional ADX gate filters out trending markets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.indicators.core.pipeline import FeatureSpec
from src.strategy.signal import Signal
from .base import StrategyContext, validate_features


@dataclass
class BBMeanReversionStrategy:
    """BB mean reversion with CCI cross-back confirmation."""

    cci_col: str
    cci_prev_col: str
    bb_upper_col: str
    bb_lower_col: str
    sma_col: str
    atr_col: str

    cci_entry: float = 100.0
    sl_atr_mult: float = 2.0
    adx_col: str = ""
    adx_max_threshold: float = 25.0

    _name: str = "bb_mean_reversion"
    _warmup_bars: int = 0

    # ── Mutable internal state ──
    _in_position_side: float = field(default=0.0, init=False, repr=False)
    _stop_level: float = field(default=0.0, init=False, repr=False)
    _tp_target: float = field(default=0.0, init=False, repr=False)
    _long_seed: bool = field(default=False, init=False, repr=False)
    _short_seed: bool = field(default=False, init=False, repr=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> Sequence[str]:
        cols = [self.cci_col, self.cci_prev_col, self.bb_upper_col,
                self.bb_lower_col, self.sma_col, self.atr_col]
        if self.adx_col:
            cols.append(self.adx_col)
        return cols

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        if ctx.bar_index < self.warmup_bars:
            return False
        return validate_features(ctx.features, self.required_features)

    def on_bar(self, ctx: StrategyContext) -> Signal:
        if not self.can_trade(ctx):
            return Signal(direction=0.0, strength=0.0, reason="warmup/feature_nan")

        close = float(ctx.bar["close"])
        bar_high = float(ctx.bar["high"])
        bar_low = float(ctx.bar["low"])

        # Sync internal side tracker with engine position
        if ctx.position == 0.0:
            self._in_position_side = 0.0
            self._stop_level = 0.0
            self._tp_target = 0.0

        cci = float(ctx.features[self.cci_col])
        cci_prev = float(ctx.features[self.cci_prev_col])
        bb_upper = float(ctx.features[self.bb_upper_col])
        bb_lower = float(ctx.features[self.bb_lower_col])
        sma = float(ctx.features[self.sma_col])
        atr = float(ctx.features[self.atr_col])

        # ── If in position: check SL / TP ──
        if self._in_position_side == 1.0:
            if bar_low <= self._stop_level:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="stop_loss")
            if bar_high >= self._tp_target:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="take_profit")
            return Signal(direction=1.0, strength=1.0, reason="hold_long")

        if self._in_position_side == -1.0:
            if bar_high >= self._stop_level:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="stop_loss")
            if bar_low <= self._tp_target:
                self._in_position_side = 0.0
                return Signal(direction=0.0, strength=1.0, reason="take_profit")
            return Signal(direction=-1.0, strength=1.0, reason="hold_short")

        # ── ADX gate (optional) ──
        if self.adx_col:
            adx = float(ctx.features[self.adx_col])
            if adx >= self.adx_max_threshold:
                self._long_seed = False
                self._short_seed = False
                return Signal(direction=0.0, strength=0.0, reason="adx_gate")

        # ── Cross-back confirmation (checked BEFORE setting new seeds) ──
        if self._long_seed and cci_prev <= -self.cci_entry and cci > -self.cci_entry:
            self._long_seed = False
            self._short_seed = False
            self._in_position_side = 1.0
            self._stop_level = close - self.sl_atr_mult * atr
            self._tp_target = sma
            return Signal(
                direction=1.0,
                strength=1.0,
                reason=f"long_entry|cci={cci:.1f}|sl={self._stop_level:.2f}|tp={self._tp_target:.2f}",
            )

        if self._short_seed and cci_prev >= self.cci_entry and cci < self.cci_entry:
            self._short_seed = False
            self._long_seed = False
            self._in_position_side = -1.0
            self._stop_level = close + self.sl_atr_mult * atr
            self._tp_target = sma
            return Signal(
                direction=-1.0,
                strength=1.0,
                reason=f"short_entry|cci={cci:.1f}|sl={self._stop_level:.2f}|tp={self._tp_target:.2f}",
            )

        # ── Set seeds ──
        if cci <= -self.cci_entry and close <= bb_lower:
            self._long_seed = True
        if cci >= self.cci_entry and close >= bb_upper:
            self._short_seed = True

        return Signal(direction=0.0, strength=0.0, reason="no_signal")


def build(params: dict) -> tuple[BBMeanReversionStrategy, list[FeatureSpec]]:
    """Build BBMeanReversionStrategy and its FeatureSpecs from config."""
    from src.indicators.impl.cci import CCI
    from src.indicators.impl.bb import BollingerUpper, BollingerLower
    from src.indicators.impl.ma import SMA
    from src.indicators.impl.atr import ATR
    from src.indicators.impl.adx import ADX
    from src.indicators.impl.transforms import Lag

    cci_period: int = params.get("cci_period", 20)
    bb_period: int = params.get("bb_period", 20)
    bb_mult: float = params.get("bb_mult", 2.0)
    atr_period: int = params.get("atr_period", 14)

    cci_ind = CCI(period=cci_period)
    bb_upper = BollingerUpper(period=bb_period, mult=bb_mult)
    bb_lower = BollingerLower(period=bb_period, mult=bb_mult)
    sma_ind = SMA(period=bb_period, src="close")
    atr_ind = ATR(period=atr_period)
    lag1 = Lag(1)
    lag2 = Lag(2)

    specs = [
        FeatureSpec(cci_ind, [lag1]),
        FeatureSpec(cci_ind, [lag2]),
        FeatureSpec(bb_upper, [lag1]),
        FeatureSpec(bb_lower, [lag1]),
        FeatureSpec(sma_ind, [lag1]),
        FeatureSpec(atr_ind, [lag1]),
    ]

    cci_col = f"{cci_ind.name}{lag1.name}"
    cci_prev_col = f"{cci_ind.name}{lag2.name}"
    bb_upper_col = f"{bb_upper.name}{lag1.name}"
    bb_lower_col = f"{bb_lower.name}{lag1.name}"
    sma_col = f"{sma_ind.name}{lag1.name}"
    atr_col = f"{atr_ind.name}{lag1.name}"

    adx_col = ""
    adx_period: int = params.get("adx_period", 0)
    adx_max_threshold: float = params.get("adx_max_threshold", 25.0)
    if adx_period > 0:
        adx_ind = ADX(period=adx_period)
        specs.append(FeatureSpec(adx_ind, [lag1]))
        adx_col = f"{adx_ind.name}{lag1.name}"

    warmup = max(cci_period, bb_period, atr_period, adx_period) + 3

    strategy = BBMeanReversionStrategy(
        cci_col=cci_col,
        cci_prev_col=cci_prev_col,
        bb_upper_col=bb_upper_col,
        bb_lower_col=bb_lower_col,
        sma_col=sma_col,
        atr_col=atr_col,
        cci_entry=params.get("cci_entry", 100.0),
        sl_atr_mult=params.get("sl_atr_mult", 2.0),
        adx_col=adx_col,
        adx_max_threshold=adx_max_threshold,
        _warmup_bars=warmup,
    )

    return strategy, specs
