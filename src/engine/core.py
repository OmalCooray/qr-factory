"""TradingEngine — core bar-by-bar trading logic, mode-agnostic.

Orchestrates the per-bar pipeline:
  execution → MTM → risk → model/signal → decision → order intent

Both backtest and future live mode use this same core.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.execution.models import Fill, OrderIntent
from src.risk import RiskAction, RiskManager
from src.strategy.base import Strategy, StrategyContext
from src.strategy.signal import Signal

log = logging.getLogger(__name__)


class TradingEngine:
    """Core bar-by-bar trading logic — mode-agnostic.

    Parameters
    ----------
    strategy : Strategy
        The model that produces Signals.
    risk_manager : RiskManager
        Stateful risk orchestrator.
    starting_capital : float
        Initial equity.
    position_size : float
        Size multiplier applied when converting Signal → OrderIntent.
    """

    def __init__(
        self,
        strategy: Strategy,
        risk_manager: RiskManager,
        starting_capital: float,
        position_size: float = 1.0,
    ) -> None:
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.starting_capital = starting_capital
        self.position_size = position_size

        # Position state
        self._equity = starting_capital
        self._position = 0.0
        self._entry_price = 0.0
        self._entry_ts = ""
        self._pending_intent: Optional[OrderIntent] = None

        # Collected outputs
        self.all_fills: list[Fill] = []
        self.equity_records: list[dict] = []

    @property
    def equity(self) -> float:
        return self._equity

    @property
    def position(self) -> float:
        return self._position

    def process_bar(
        self,
        bar_row: pd.Series,
        features_row: pd.Series,
        bar_index: int,
        is_last: bool = False,
    ) -> None:
        """Process one bar through the full pipeline."""
        bar_ts = bar_row["time"]
        bar_ts_str = bar_ts.isoformat()

        # ── EXECUTION ENGINE: fill previous bar's intent at Open ──
        target_position = (
            self._pending_intent.target_position
            if self._pending_intent is not None
            else self._position
        )
        self._position, self._entry_price, self._entry_ts, fills = _execute_order(
            target_position,
            self._position,
            self._entry_price,
            self._entry_ts,
            bar_row["open"],
            bar_ts_str,
        )
        for f in fills:
            self._equity += f.pnl
        self.all_fills.extend(fills)

        # ── MARK-TO-MARKET at Close ──
        current_close = bar_row["close"]
        unrealized_pnl = 0.0
        if self._position != 0:
            if self._position > 0:
                unrealized_pnl = (current_close - self._entry_price) * abs(
                    self._position
                )
            else:
                unrealized_pnl = (self._entry_price - current_close) * abs(
                    self._position
                )

        current_equity = self._equity + unrealized_pnl

        # ── RISK LAYER: check drawdown limits ──
        risk_action = self.risk_manager.update(bar_ts, current_equity)
        if risk_action.reason:
            log.warning("Risk: %s at %s", risk_action.reason, bar_ts_str)

        # ── RECORD equity snapshot ──
        self.equity_records.append(
            {
                "timestamp": bar_ts_str,
                "equity": current_equity,
                "position": self._position,
                "unrealized_pnl": unrealized_pnl,
                "realized_pnl": self._equity - self.starting_capital,
            }
        )

        if is_last:
            self._pending_intent = None
            return

        # ── MODEL → SIGNAL: strategy reads features, produces signal ──
        ctx = StrategyContext(
            ts=bar_ts,
            bar=bar_row,
            features=features_row,
            position=self._position,
            equity=current_equity,
            bar_index=bar_index,
        )
        signal = self.strategy.on_bar(ctx)

        # ── DECISION LOGIC: signal + risk → order intent ──
        intent = self._decide(signal, risk_action)

        # ── ORDER INTENT: stored for next bar's execution ──
        self._pending_intent = intent

    def _decide(self, signal: Signal, risk_action: RiskAction) -> OrderIntent:
        """Convert Signal + RiskAction → OrderIntent (sizing + risk gate)."""
        if risk_action.flatten:
            return OrderIntent(target_position=0.0, reason=f"risk:{risk_action.reason}")

        target = signal.direction * self.position_size
        return OrderIntent(target_position=target, reason=signal.reason)


def _execute_order(
    target_position: float,
    position: float,
    entry_price: float,
    entry_ts: str,
    bar_open: float,
    bar_ts: str,
) -> tuple[float, float, str, list[Fill]]:
    """Execute a position change at the current bar's open price.

    Returns
    -------
    tuple of (new_position, new_entry_price, new_entry_ts, fills)
    """
    fills: list[Fill] = []

    if position == target_position:
        return position, entry_price, entry_ts, fills

    # Close existing position if flattening or flipping
    if position != 0:
        if (position > 0 and target_position <= 0) or (
            position < 0 and target_position >= 0
        ):
            exit_price = bar_open
            if position > 0:
                pnl = (exit_price - entry_price) * abs(position)
                side = "long"
            else:
                pnl = (entry_price - exit_price) * abs(position)
                side = "short"

            fills.append(
                Fill(
                    entry_ts=entry_ts,
                    exit_ts=bar_ts,
                    side=side,
                    qty=abs(position),
                    entry_price=entry_price,
                    exit_price=exit_price,
                    pnl=pnl,
                )
            )
            position = 0.0
            entry_price = 0.0

    # Open new position (or flip into new)
    if target_position != 0 and position == 0:
        position = target_position
        entry_price = bar_open
        entry_ts = bar_ts

    return position, entry_price, entry_ts, fills
