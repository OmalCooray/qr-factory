"""Fixed-threshold policy with hysteresis and minimum hold."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.strategy.base import StrategyContext
from src.strategy.signal import Signal


@dataclass
class FixedThresholdPolicy:
    """Enter/exit based on fixed probability thresholds.

    Hysteresis logic:
    - **Flat**: enter long if ``yhat >= p_enter``
    - **Long**: hold if ``yhat >= p_exit`` OR ``bars_held < min_hold_bars``
    - **Long**: exit if ``yhat < p_exit`` AND ``bars_held >= min_hold_bars``

    If ``p_exit is None``, falls back to ``p_enter`` (no dead-band).
    """

    p_enter: float = 0.55
    p_exit: float | None = None
    min_hold_bars: int = 0
    _bars_held: int = field(default=0, init=False, repr=False)

    def decide(self, ctx: StrategyContext, yhat: float) -> Signal:
        effective_p_exit = self.p_exit if self.p_exit is not None else self.p_enter
        is_long = ctx.position > 0

        if is_long:
            self._bars_held += 1
            bars_held = self._bars_held

            if yhat >= effective_p_exit or bars_held < self.min_hold_bars:
                return Signal(
                    direction=1.0,
                    strength=1.0,
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
                return Signal(
                    direction=1.0,
                    strength=1.0,
                    reason=f"enter yhat={yhat:.3f}>=p_enter",
                )
            self._bars_held = 0
            return Signal(
                direction=0.0,
                strength=1.0,
                reason=f"flat yhat={yhat:.3f}<p_enter",
            )

    def reset(self) -> None:
        self._bars_held = 0
