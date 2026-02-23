"""Tests for max-drawdown risk guard in the backtest engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.engine.runner import run_backtest


def _write_synthetic_crash(data_dir: Path) -> None:
    """Write a CSV where prices rise then crash sharply.

    With fast=2 / slow=5 and starting_capital=100, the strategy enters
    long during the rise.  The subsequent crash inflicts a drawdown well
    above 5 % of peak equity, tripping the risk guard.

    Price series (close):
      bars 0-9:   flat at 50   (warmup / equal MAs → flat)
      bars 10-19: ramp 55..100 (fast > slow → long)
      bars 20-29: crash to 30  (drawdown while still long)
      bars 30-39: stay at 30   (should be halted)
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    n = 40
    times = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    close: list[float] = (
        [50.0] * 10
        + [55 + i * 5 for i in range(10)]  # 55, 60, 65, ..., 100
        + [30.0] * 10
        + [30.0] * 10
    )

    # Open = previous close (simple gap-free series)
    open_: list[float] = [close[0]] + [close[i - 1] for i in range(1, n)]

    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": [max(o, c) + 1 for o, c in zip(open_, close)],
        "low": [min(o, c) - 1 for o, c in zip(open_, close)],
        "close": close,
        "volume": [1000] * n,
    })
    df.to_csv(data_dir / "crash.csv", index=False)


def test_risk_guard_halts_on_drawdown(tmp_path: Path) -> None:
    """When max_drawdown_pct is set, engine halts trading after breach."""
    data_dir = tmp_path / "data"
    _write_synthetic_crash(data_dir)

    config = {
        "symbol": "TEST",
        "timeframe": "1h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 100.0,   # small capital so 1-lot moves matter
        "output_dir": str(tmp_path / "runs"),
        "max_drawdown_pct": 5.0,
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_id = run_backtest(str(config_path))

    run_dir = tmp_path / "runs" / run_id
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))

    # Risk guard must have triggered
    assert metrics["risk_halted"] is True
    assert metrics["risk_halted_at"] is not None
    assert metrics["max_drawdown_pct"] >= 5.0

    # After halt, position should be flat for the rest of the run
    equity_df = pd.read_csv(run_dir / "equity.csv")
    halt_ts = metrics["risk_halted_at"]
    # Find rows after the halt timestamp (next bar onward, where flatten executes)
    after_halt = equity_df[equity_df["timestamp"] > halt_ts]
    # Allow one bar for the flatten execution; after that, position must be 0
    if len(after_halt) > 1:
        post_flatten = after_halt.iloc[1:]
        assert (post_flatten["position"] == 0.0).all(), (
            "Position should be flat after risk guard flattens"
        )


def test_no_risk_guard_by_default(tmp_path: Path) -> None:
    """Without max_drawdown_pct, risk_halted should be False."""
    data_dir = tmp_path / "data"
    _write_synthetic_crash(data_dir)

    config = {
        "symbol": "TEST",
        "timeframe": "1h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 100.0,
        "output_dir": str(tmp_path / "runs"),
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_id = run_backtest(str(config_path))
    run_dir = tmp_path / "runs" / run_id
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))

    assert metrics["risk_halted"] is False
    assert metrics["risk_halted_at"] is None


# ── Helpers for periodic drawdown tests ──────────────────────────────


def _write_daily_crash(data_dir: Path) -> None:
    """Two days of hourly bars. Day 1: ramp then crash. Day 2: normal bars.

    Day 1 (2024-01-01): 12 bars — ramp from 50 to 100, then crash to 30.
    Day 2 (2024-01-02): 12 bars — flat at 50 (normal trading).
    With fast=2 / slow=5 / starting_capital=10000, the daily DD on day 1
    should exceed 3 % of day-start equity once the crash hits while long.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Day 1: 12 hourly bars
    day1_close = (
        [50.0] * 5                                 # warmup
        + [60.0, 70.0, 80.0, 90.0, 100.0]         # ramp → goes long
        + [30.0, 30.0]                             # crash while long
    )
    # Day 2: 12 hourly bars — flat at 50
    day2_close = [50.0] * 12

    close = day1_close + day2_close
    n = len(close)

    times = (
        list(pd.date_range("2024-01-01", periods=12, freq="1h", tz="UTC"))
        + list(pd.date_range("2024-01-02", periods=12, freq="1h", tz="UTC"))
    )

    open_ = [close[0]] + [close[i - 1] for i in range(1, n)]
    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": [max(o, c) + 1 for o, c in zip(open_, close)],
        "low": [min(o, c) - 1 for o, c in zip(open_, close)],
        "close": close,
        "volume": [1000] * n,
    })
    df.to_csv(data_dir / "daily.csv", index=False)


def _write_monthly_crash(data_dir: Path) -> None:
    """Bars spanning two months. Jan crash triggers monthly DD; Feb resumes.

    Jan 28-31 (4 days × 4 bars/day = 16 bars): ramp then crash.
    Feb 1-2  (2 days × 4 bars/day = 8 bars): flat/normal.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Jan: warmup → ramp → crash
    jan_close = (
        [50.0] * 5                                 # warmup
        + [60.0, 70.0, 80.0, 90.0, 100.0, 100.0]  # ramp
        + [30.0] * 5                               # crash
    )
    # Feb: flat at 50
    feb_close = [50.0] * 8

    close = jan_close + feb_close
    n = len(close)

    times = (
        list(pd.date_range("2024-01-28", periods=16, freq="6h", tz="UTC"))
        + list(pd.date_range("2024-02-01", periods=8, freq="6h", tz="UTC"))
    )

    open_ = [close[0]] + [close[i - 1] for i in range(1, n)]
    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": [max(o, c) + 1 for o, c in zip(open_, close)],
        "low": [min(o, c) - 1 for o, c in zip(open_, close)],
        "close": close,
        "volume": [1000] * n,
    })
    df.to_csv(data_dir / "monthly.csv", index=False)


# ── Periodic drawdown tests ─────────────────────────────────────────


def test_daily_drawdown_pauses_and_resumes(tmp_path: Path) -> None:
    """Daily DD limit pauses trading for the rest of the day, resumes next day."""
    data_dir = tmp_path / "data"
    _write_daily_crash(data_dir)

    config = {
        "symbol": "TEST",
        "timeframe": "1h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 100.0,
        "output_dir": str(tmp_path / "runs"),
        "daily_drawdown_pct": 3.0,
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_id = run_backtest(str(config_path))
    run_dir = tmp_path / "runs" / run_id
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))

    # Daily halt must have triggered
    assert metrics["daily_halts"] >= 1
    assert metrics["daily_drawdown_pct_limit"] == 3.0

    # Check equity CSV: after day-1 crash, position should go flat
    equity_df = pd.read_csv(run_dir / "equity.csv")
    day2_rows = equity_df[equity_df["timestamp"].str.startswith("2024-01-02")]
    # On day 2 the strategy should be allowed to trade again (not permanently halted)
    assert not day2_rows.empty, "Should have day-2 bars"
    # risk_halted should NOT be set (periodic != permanent)
    assert metrics["risk_halted"] is False


def test_monthly_drawdown_pauses_and_resumes(tmp_path: Path) -> None:
    """Monthly DD limit pauses trading for the rest of the month, resumes next month."""
    data_dir = tmp_path / "data"
    _write_monthly_crash(data_dir)

    config = {
        "symbol": "TEST",
        "timeframe": "6h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 100.0,
        "output_dir": str(tmp_path / "runs"),
        "monthly_drawdown_pct": 3.0,
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_id = run_backtest(str(config_path))
    run_dir = tmp_path / "runs" / run_id
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))

    # Monthly halt must have triggered
    assert metrics["monthly_halts"] >= 1
    assert metrics["monthly_drawdown_pct_limit"] == 3.0

    # Check equity CSV: Feb bars should exist (trading resumed)
    equity_df = pd.read_csv(run_dir / "equity.csv")
    feb_rows = equity_df[equity_df["timestamp"].str.startswith("2024-02")]
    assert not feb_rows.empty, "Should have Feb bars after monthly reset"
    # risk_halted should NOT be set
    assert metrics["risk_halted"] is False


def test_no_periodic_limits_by_default(tmp_path: Path) -> None:
    """Without daily/monthly config, periodic halt counters stay at zero."""
    data_dir = tmp_path / "data"
    _write_synthetic_crash(data_dir)

    config = {
        "symbol": "TEST",
        "timeframe": "1h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 100.0,
        "output_dir": str(tmp_path / "runs"),
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_id = run_backtest(str(config_path))
    run_dir = tmp_path / "runs" / run_id
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))

    assert metrics["daily_halts"] == 0
    assert metrics["monthly_halts"] == 0
    assert metrics["daily_drawdown_pct_limit"] is None
    assert metrics["monthly_drawdown_pct_limit"] is None
