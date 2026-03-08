"""Policy protocol — converts a probability into a Signal."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.strategy.base import StrategyContext
from src.strategy.signal import Signal


@runtime_checkable
class Policy(Protocol):
    """Decides whether to trade given a predicted probability.

    Implementations maintain internal state (e.g. bars-held counter)
    and produce a Signal each bar.
    """

    def decide(self, ctx: StrategyContext, yhat: float) -> Signal: ...

    def reset(self) -> None: ...
