"""Unit tests for src.risk.risk_manager — no backtest engine required."""

from __future__ import annotations

import pandas as pd
import pytest

from src.risk import RiskAction, RiskConfig, RiskManager


def _ts(iso: str) -> pd.Timestamp:
    """Shorthand: parse an ISO string into a tz-aware Timestamp."""
    return pd.Timestamp(iso, tz="UTC")


# ---------------------------------------------------------------------------
# Global halt
# ---------------------------------------------------------------------------


class TestGlobalHalt:
    def test_triggers_at_threshold(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=10.0), starting_capital=1000.0)
        # Equity rises to 1100 (new peak), then drops to 990 → DD = 10%
        rm.update(_ts("2024-01-01 00:00"), 1100.0)
        action = rm.update(_ts("2024-01-01 01:00"), 990.0)
        assert action.flatten is True
        assert action.halted is True
        assert "Global DD" in action.reason

    def test_permanent_after_trigger(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=5.0), starting_capital=1000.0)
        rm.update(_ts("2024-01-01 00:00"), 1000.0)
        rm.update(_ts("2024-01-01 01:00"), 949.0)  # DD = 5.1% → halt
        # Even if equity recovers, halt is permanent
        action = rm.update(_ts("2024-01-01 02:00"), 2000.0)
        assert action.flatten is True
        assert action.halted is True

    def test_no_trigger_below_threshold(self):
        rm = RiskManager(RiskConfig(max_drawdown_pct=10.0), starting_capital=1000.0)
        action = rm.update(_ts("2024-01-01 00:00"), 995.0)  # DD = 0.5%
        assert action.flatten is False
        assert action.halted is False


# ---------------------------------------------------------------------------
# Daily drawdown
# ---------------------------------------------------------------------------


class TestDailyDrawdown:
    def test_triggers_on_limit(self):
        rm = RiskManager(RiskConfig(daily_dd_limit=5.0), starting_capital=1000.0)
        # Bar 1 on day 1 — equity at 1000
        rm.update(_ts("2024-01-01 00:00"), 1000.0)
        # Bar 2 same day — equity drops to 950 → daily DD = 5%
        action = rm.update(_ts("2024-01-01 01:00"), 950.0)
        assert action.flatten is True
        assert action.halted is False  # daily pause, not permanent
        assert "Daily DD" in action.reason

    def test_resets_on_day_boundary(self):
        rm = RiskManager(RiskConfig(daily_dd_limit=5.0), starting_capital=1000.0)
        rm.update(_ts("2024-01-01 00:00"), 1000.0)
        rm.update(_ts("2024-01-01 01:00"), 950.0)  # triggers daily
        # New day → pause resets; baseline becomes last bar's MTM (950)
        action = rm.update(_ts("2024-01-02 00:00"), 950.0)
        # 950 vs baseline 950 → 0% DD → no flatten
        assert action.flatten is False

    def test_mtm_baseline(self):
        """Key bug-fix test: baseline uses previous bar's MTM equity, not
        realised equity.

        Scenario:
          - Day 1 ends with MTM equity 1050 (unrealised profit).
          - Day 2 baseline must be 1050.
          - Drop to 997.5 → DD = (1050 - 997.5) / 1050 = 5.0% → triggers.
        """
        rm = RiskManager(RiskConfig(daily_dd_limit=5.0), starting_capital=1000.0)
        # Day 1: equity rises to 1050 (e.g. unrealised P&L on open position)
        rm.update(_ts("2024-01-01 00:00"), 1000.0)
        rm.update(_ts("2024-01-01 01:00"), 1050.0)
        # Day 2: baseline should be 1050 (last bar's MTM), not starting_capital
        action = rm.update(_ts("2024-01-02 00:00"), 997.5)
        # DD = (1050 - 997.5) / 1050 * 100 = 5.0%
        assert action.flatten is True
        assert "Daily DD" in action.reason

        # Verify: if baseline were 1000 (realised), DD would only be 0.25% —
        # and should NOT trigger. The fact that it triggered proves MTM baseline.
        metrics = rm.metrics()
        assert metrics["daily_halts"] == 1


# ---------------------------------------------------------------------------
# Monthly drawdown
# ---------------------------------------------------------------------------


class TestMonthlyDrawdown:
    def test_triggers_on_limit(self):
        rm = RiskManager(RiskConfig(monthly_dd_limit=8.0), starting_capital=1000.0)
        rm.update(_ts("2024-01-15 00:00"), 1000.0)
        action = rm.update(_ts("2024-01-20 00:00"), 920.0)  # DD = 8%
        assert action.flatten is True
        assert "Monthly DD" in action.reason

    def test_resets_on_month_boundary(self):
        rm = RiskManager(RiskConfig(monthly_dd_limit=8.0), starting_capital=1000.0)
        rm.update(_ts("2024-01-15 00:00"), 1000.0)
        rm.update(_ts("2024-01-20 00:00"), 920.0)  # triggers monthly
        # February → pause resets; baseline = last bar's MTM (920)
        action = rm.update(_ts("2024-02-01 00:00"), 920.0)
        assert action.flatten is False


# ---------------------------------------------------------------------------
# No limits configured
# ---------------------------------------------------------------------------


class TestNoLimitsConfigured:
    def test_never_flattens(self):
        rm = RiskManager(RiskConfig(), starting_capital=1000.0)
        for eq in [900, 800, 500, 100]:
            action = rm.update(_ts("2024-01-01 00:00"), float(eq))
            assert action.flatten is False
            assert action.halted is False

    def test_still_tracks_max_dd(self):
        rm = RiskManager(RiskConfig(), starting_capital=1000.0)
        rm.update(_ts("2024-01-01 00:00"), 1000.0)
        rm.update(_ts("2024-01-01 01:00"), 800.0)
        m = rm.metrics()
        assert m["max_drawdown_pct"] == pytest.approx(20.0)

    def test_metrics_schema(self):
        rm = RiskManager(RiskConfig(), starting_capital=1000.0)
        rm.update(_ts("2024-01-01 00:00"), 1000.0)
        m = rm.metrics()
        expected_keys = {
            "max_drawdown_pct",
            "risk_halted",
            "risk_halted_at",
            "daily_drawdown_pct_limit",
            "monthly_drawdown_pct_limit",
            "daily_halts",
            "monthly_halts",
        }
        assert set(m.keys()) == expected_keys
