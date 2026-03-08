"""Probability-sized policy — linear scaling from prediction confidence.

Entry size scales linearly between min_size and max_size based on how far
yhat exceeds p_enter (up to p_full).  Includes hysteresis dead-band and
minimum hold period, same as FixedThresholdPolicy.

Important: the engine does NOT resize mid-trade.  The entry-bar sizing
(signal.strength at the bar the position opens) determines the position
size for the entire trade.  Hold signals re-compute strength for audit /
logging purposes, but the engine maintains the original position size
until exit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.strategy.base import StrategyContext
from src.strategy.signal import Signal


@dataclass
class ProbabilitySizedPolicy:
    """Enter/exit with position size proportional to prediction confidence.

    Parameters
    ----------
    p_enter : float
        Minimum yhat to open a new position.
    p_exit : float | None
        Exit threshold (default: falls back to p_enter).
    p_full : float
        yhat at which position reaches max_size.  Must be > p_enter.
    min_size : float
        Position fraction at exactly p_enter.
    max_size : float
        Position fraction at p_full and above.
    min_hold_bars : int
        Minimum bars to hold before allowing exit.
    """

    p_enter: float = 0.55
    p_exit: float | None = None
    p_full: float = 0.65
    min_size: float = 0.25
    max_size: float = 1.0
    min_hold_bars: int = 0
    _bars_held: int = field(default=0, init=False, repr=False)

    def _scale(self, yhat: float) -> float:
        """Linear interpolation of size between min_size and max_size."""
        confidence = (yhat - self.p_enter) / (self.p_full - self.p_enter)
        confidence = max(0.0, min(1.0, confidence))
        return self.min_size + confidence * (self.max_size - self.min_size)

    def decide(self, ctx: StrategyContext, yhat: float) -> Signal:
        effective_p_exit = self.p_exit if self.p_exit is not None else self.p_enter
        is_long = ctx.position > 0

        if is_long:
            self._bars_held += 1
            bars_held = self._bars_held

            if yhat >= effective_p_exit or bars_held < self.min_hold_bars:
                strength = self._scale(yhat) if yhat >= effective_p_exit else self.min_size
                return Signal(
                    direction=1.0,
                    strength=strength,
                    reason=f"hold yhat={yhat:.3f} bars={bars_held}",
                )
            self._bars_held = 0
            return Signal(
                direction=0.0,
                strength=1.0,
                reason=f"exit yhat={yhat:.3f}<p_exit={effective_p_exit:.3f}",
            )
        else:
            if yhat >= self.p_enter:
                self._bars_held = 1
                strength = self._scale(yhat)
                return Signal(
                    direction=1.0,
                    strength=strength,
                    reason=f"enter yhat={yhat:.3f}>=p_enter",
                )
            self._bars_held = 0
            return Signal(
                direction=0.0,
                strength=0.0,
                reason=f"flat yhat={yhat:.3f}<p_enter",
            )

    def reset(self) -> None:
        self._bars_held = 0
