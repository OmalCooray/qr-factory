"""Unified risk-management orchestrator.

Owns global drawdown halt, daily pause, and monthly pause logic.
The engine calls :meth:`RiskManager.update` once per bar (after MTM
computation) and inspects the returned :class:`RiskAction`.

**Flatten latency**: when a :class:`RiskAction` requests ``flatten=True``,
the engine sets ``target_position = 0.0``.  The actual position close
executes at the *next* bar's open price (by design — the backtest uses
next-bar-open execution).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from src.risk.drawdown import DrawdownTracker, compute_drawdown_pct


@dataclass(frozen=True)
class RiskConfig:
    """Risk limits read from the YAML backtest config."""

    max_drawdown_pct: Optional[float] = None
    daily_dd_limit: Optional[float] = None
    monthly_dd_limit: Optional[float] = None


@dataclass(frozen=True)
class RiskAction:
    """Instruction returned to the engine on every bar.

    * ``flatten`` — ``True`` means the engine must set
      ``target_position = 0.0``.
    * ``halted`` — ``True`` means the global (permanent) halt fired.
    * ``reason`` — human-readable explanation; empty string when no action.
    """

    flatten: bool = False
    halted: bool = False
    reason: str = ""


class RiskManager:
    """Stateful per-backtest risk orchestrator.

    Parameters
    ----------
    config : RiskConfig
        Thresholds.  ``None`` values disable the corresponding check.
    starting_capital : float
        Initial equity (used as the first period baseline).
    """

    def __init__(self, config: RiskConfig, starting_capital: float) -> None:
        self._cfg = config
        self._starting_capital = starting_capital

        # Global drawdown tracker (peak / max DD)
        self._global_tracker = DrawdownTracker(starting_capital)
        self._risk_halted = False
        self._risk_halted_at: Optional[str] = None

        # Daily state
        self._daily_paused = False
        self._current_day: Optional[object] = None
        self._day_start_equity = starting_capital
        self._daily_halts = 0

        # Monthly state
        self._monthly_paused = False
        self._current_month: Optional[tuple[int, int]] = None
        self._month_start_equity = starting_capital
        self._monthly_halts = 0

        # MTM baseline: tracks the *previous* bar's MTM equity so that
        # period-boundary baselines reflect unrealized P&L.  Initialised to
        # starting_capital (correct for the very first bar).
        self._last_equity_close = starting_capital

    # -- public API ----------------------------------------------------------

    def update(self, bar_ts: pd.Timestamp, current_equity: float) -> RiskAction:
        """Process one bar and return the risk action.

        Must be called *after* mark-to-market computation (i.e.
        ``current_equity`` includes unrealized P&L).
        """
        flatten = False
        reason_parts: list[str] = []

        bar_ts_str = bar_ts.isoformat()

        # 0. Period-boundary resets — use *previous bar's* MTM equity as the
        #    new period baseline (fixes the realised-only baseline bug).
        bar_date = bar_ts.date()
        bar_month = (bar_ts.year, bar_ts.month)

        if bar_date != self._current_day:
            self._daily_paused = False
            self._current_day = bar_date
            self._day_start_equity = self._last_equity_close

        if bar_month != self._current_month:
            self._monthly_paused = False
            self._current_month = bar_month
            self._month_start_equity = self._last_equity_close

        # 1. Global drawdown check
        state = self._global_tracker.update(current_equity)

        if (
            not self._risk_halted
            and self._cfg.max_drawdown_pct is not None
            and state.current_drawdown_pct >= self._cfg.max_drawdown_pct
        ):
            self._risk_halted = True
            self._risk_halted_at = bar_ts_str
            flatten = True
            reason_parts.append(
                f"Global DD {state.current_drawdown_pct:.2f}% >= "
                f"limit {self._cfg.max_drawdown_pct:.2f}%"
            )

        # 2. Daily drawdown check
        if (
            not self._daily_paused
            and self._cfg.daily_dd_limit is not None
            and self._day_start_equity > 0
        ):
            daily_dd = compute_drawdown_pct(self._day_start_equity, current_equity)
            if daily_dd >= self._cfg.daily_dd_limit:
                self._daily_paused = True
                self._daily_halts += 1
                flatten = True
                reason_parts.append(
                    f"Daily DD {daily_dd:.2f}% >= "
                    f"limit {self._cfg.daily_dd_limit:.2f}%"
                )

        # 3. Monthly drawdown check
        if (
            not self._monthly_paused
            and self._cfg.monthly_dd_limit is not None
            and self._month_start_equity > 0
        ):
            monthly_dd = compute_drawdown_pct(self._month_start_equity, current_equity)
            if monthly_dd >= self._cfg.monthly_dd_limit:
                self._monthly_paused = True
                self._monthly_halts += 1
                flatten = True
                reason_parts.append(
                    f"Monthly DD {monthly_dd:.2f}% >= "
                    f"limit {self._cfg.monthly_dd_limit:.2f}%"
                )

        # If already halted/paused from a previous bar, keep flattening
        if self._risk_halted or self._daily_paused or self._monthly_paused:
            flatten = True

        # Update MTM baseline for next bar's period-boundary detection
        self._last_equity_close = current_equity

        return RiskAction(
            flatten=flatten,
            halted=self._risk_halted,
            reason="; ".join(reason_parts),
        )

    def metrics(self) -> dict:
        """Return the 7 risk-related keys expected by ``metrics.json``."""
        return {
            "max_drawdown_pct": float(self._global_tracker.max_drawdown_pct),
            "risk_halted": self._risk_halted,
            "risk_halted_at": self._risk_halted_at,
            "daily_drawdown_pct_limit": self._cfg.daily_dd_limit,
            "monthly_drawdown_pct_limit": self._cfg.monthly_dd_limit,
            "daily_halts": self._daily_halts,
            "monthly_halts": self._monthly_halts,
        }
