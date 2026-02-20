from typing import Sequence
from dataclasses import dataclass
from .base import Strategy, StrategyContext, StrategyDecision, validate_features

@dataclass
class MACrossoverStrategy:
    """
    Simple Moving Average Crossover Strategy.
    """
    fast_col: str
    slow_col: str
    size: float = 1.0
    _name: str = "ma_crossover"
    _warmup_bars: int = 0

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def required_features(self) -> Sequence[str]:
        return [self.fast_col, self.slow_col]

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        """
        Check if we can trade:
        1. Passed warmup period.
        2. Required features exist and are not NaN.
        """
        if ctx.bar_index < self.warmup_bars:
            return False
            
        if not validate_features(ctx.features, self.required_features):
            return False
            
        return True

    def on_bar(self, ctx: StrategyContext) -> StrategyDecision:
        """
        Generate trading decision based on MA crossover.
        """
        if not self.can_trade(ctx):
            return StrategyDecision(
                target_position=0.0,
                reason="warmup/feature_nan"
            )

        fast_val = ctx.features[self.fast_col]
        slow_val = ctx.features[self.slow_col]

        target_pos = 0.0
        reason = "wait"

        if fast_val > slow_val:
            target_pos = self.size
            reason = "cross_above"
        elif fast_val < slow_val:
            target_pos = -self.size
            reason = "cross_below"
        else:
            target_pos = 0.0
            reason = "equal"
        
        return StrategyDecision(
            target_position=target_pos,
            reason=reason
        )
