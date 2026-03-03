"""Execution-layer value objects: OrderIntent and Fill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OrderIntent:
    """What the strategy wants the engine to do on the next bar.

    Replaces the former ``StrategyDecision``.
    """

    target_position: float  # -size short, +size long, 0 flat
    reason: str = ""
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass(frozen=True)
class Fill:
    """One completed round-trip trade — maps 1:1 to a trades.csv row."""

    entry_ts: str
    exit_ts: str
    side: str  # "long" or "short"
    qty: float
    entry_price: float
    exit_price: float
    pnl: float
    gross_pnl: float = 0.0

    @property
    def cost_pnl(self) -> float:
        """Transaction cost = gross_pnl - net_pnl (always >= 0)."""
        return self.gross_pnl - self.pnl

    def to_dict(self) -> dict:
        return {
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "side": self.side,
            "qty": self.qty,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl": self.pnl,
            "gross_pnl": self.gross_pnl,
            "net_pnl": self.pnl,
            "cost_pnl": self.cost_pnl,
        }
