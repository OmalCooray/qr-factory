"""Execution cost model — fixed spread + slippage.

Converts point-based config to price-unit costs and applies them to fill prices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostConfig:
    """Execution cost parameters (all in points)."""

    spread_points: float = 0.0
    slippage_points: float = 0.0
    point_value: float = 1.0

    @property
    def spread(self) -> float:
        """Full spread in price units."""
        return self.spread_points * self.point_value

    @property
    def slippage(self) -> float:
        """Slippage in price units."""
        return self.slippage_points * self.point_value

    @property
    def half_spread(self) -> float:
        return self.spread / 2.0


def cost_adjusted_price(
    mid: float, is_buy: bool, cost: CostConfig
) -> float:
    """Apply half-spread + slippage to a mid price.

    BUY  → mid + half_spread + slippage  (pay more)
    SELL → mid - half_spread - slippage  (receive less)
    """
    adjustment = cost.half_spread + cost.slippage
    return mid + adjustment if is_buy else mid - adjustment
