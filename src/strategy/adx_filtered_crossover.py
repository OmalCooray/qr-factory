from typing import Sequence
from dataclasses import dataclass

from src.indicators.core.pipeline import FeatureSpec
from src.strategy.signal import Signal
from .base import StrategyContext, validate_features


@dataclass
class ADXFilteredCrossoverStrategy:
    """MA Crossover gated by ADX — flattens when ADX < threshold."""

    fast_col: str
    slow_col: str
    adx_col: str
    adx_threshold: float = 25.0
    _name: str = "adx_filtered_crossover"
    _warmup_bars: int = 0

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> Sequence[str]:
        return [self.fast_col, self.slow_col, self.adx_col]

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        if ctx.bar_index < self.warmup_bars:
            return False
        if not validate_features(ctx.features, self.required_features):
            return False
        return True

    def on_bar(self, ctx: StrategyContext) -> Signal:
        if not self.can_trade(ctx):
            return Signal(direction=0.0, strength=0.0, reason="warmup/feature_nan")

        adx_val = ctx.features[self.adx_col]

        # ADX below threshold → go flat (including flattening open positions)
        if adx_val < self.adx_threshold:
            return Signal(direction=0.0, strength=0.0, reason="adx_below_threshold")

        fast_val = ctx.features[self.fast_col]
        slow_val = ctx.features[self.slow_col]

        if fast_val > slow_val:
            return Signal(direction=1.0, strength=1.0, reason="cross_above")
        elif fast_val < slow_val:
            return Signal(direction=-1.0, strength=1.0, reason="cross_below")
        else:
            return Signal(direction=0.0, strength=0.0, reason="equal")


def build(params: dict) -> tuple[ADXFilteredCrossoverStrategy, list[FeatureSpec]]:
    """Build an ADXFilteredCrossoverStrategy and its FeatureSpecs from config params."""
    from src.indicators.impl.ma import SMA
    from src.indicators.impl.ema import EMA
    from src.indicators.impl.adx import ADX

    fast_period: int = params["fast_period"]
    slow_period: int = params["slow_period"]
    indicator_type: str = params.get("indicator_type", "sma")
    adx_period: int = params.get("adx_period", 14)
    adx_threshold: float = params.get("adx_threshold", 25.0)

    if indicator_type == "ema":
        fast_indicator = EMA(fast_period)
        slow_indicator = EMA(slow_period)
    else:
        fast_indicator = SMA(fast_period)
        slow_indicator = SMA(slow_period)

    adx_indicator = ADX(adx_period)

    specs = [
        FeatureSpec(fast_indicator, []),
        FeatureSpec(slow_indicator, []),
        FeatureSpec(adx_indicator, []),
    ]

    strategy = ADXFilteredCrossoverStrategy(
        fast_col=fast_indicator.name,
        slow_col=slow_indicator.name,
        adx_col=adx_indicator.name,
        adx_threshold=adx_threshold,
        _warmup_bars=max(slow_period, 2 * adx_period),
    )

    return strategy, specs
