"""Tests for the execution cost model.

1. Zero-cost invariance: costs=0 ⇒ net == gross everywhere.
2. Cost monotonicity: higher costs ⇒ worse net equity.
"""

import json

import pandas as pd
import pytest
import yaml

from src.engine.runner import run_backtest


def _make_config(tmp_path, *, spread_points=0.0, slippage_points=0.0, point_value=1.0):
    """Create a minimal config + synthetic data for a cost test."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    # Synthetic OHLCV with a crossover at bar 10 and reversal at bar 20
    dates = pd.date_range("2024-01-01", periods=50, freq="1h")
    df = pd.DataFrame(
        {
            "time": dates,
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "volume": 1000.0,
        }
    )
    # Induce crossover: price jump at bar 10, drop at bar 20
    df.loc[10:15, ["open", "high", "close"]] = 110.0
    df.loc[20:25, ["open", "low", "close"]] = 90.0
    df.to_csv(data_dir / "test.csv", index=False)

    config = {
        "symbol": "TEST",
        "timeframe": "1h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 10000.0,
        "output_dir": str(tmp_path / "runs"),
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    if spread_points or slippage_points or point_value != 1.0:
        config["execution"] = {
            "spread_points": spread_points,
            "slippage_points": slippage_points,
            "point_value": point_value,
        }

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path


def test_zero_cost_invariance(tmp_path):
    """With zero costs, net and gross equity must be identical."""
    cfg_path = _make_config(tmp_path, spread_points=0, slippage_points=0)
    run_id = run_backtest(str(cfg_path))
    run_dir = tmp_path / "runs" / run_id

    # equity.csv (net) and equity_gross.csv must be identical
    eq_net = pd.read_csv(run_dir / "equity.csv")
    eq_gross = pd.read_csv(run_dir / "equity_gross.csv")
    pd.testing.assert_series_equal(
        eq_net["equity"], eq_gross["equity"], check_names=False
    )

    # trades.csv: gross_pnl == net_pnl, cost_pnl == 0 for every trade
    trades = pd.read_csv(run_dir / "trades.csv")
    if not trades.empty:
        pd.testing.assert_series_equal(
            trades["gross_pnl"], trades["net_pnl"], check_names=False
        )
        assert (trades["cost_pnl"].abs() < 1e-12).all(), "cost_pnl should be 0"

    # metrics.json
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert abs(metrics["total_cost_pnl"]) < 1e-12
    assert abs(metrics["total_gross_pnl"] - metrics["total_net_pnl"]) < 1e-12


def test_cost_monotonicity(tmp_path):
    """Increasing costs should degrade net equity relative to gross."""
    # Run with zero costs
    dir_zero = tmp_path / "zero"
    dir_zero.mkdir()
    cfg_zero = _make_config(dir_zero, spread_points=0, slippage_points=0)
    rid_zero = run_backtest(str(cfg_zero))
    m_zero = json.loads(
        (dir_zero / "runs" / rid_zero / "metrics.json").read_text()
    )

    # Run with moderate costs
    dir_cost = tmp_path / "cost"
    dir_cost.mkdir()
    cfg_cost = _make_config(
        dir_cost, spread_points=50, slippage_points=10, point_value=0.01
    )
    rid_cost = run_backtest(str(cfg_cost))
    run_dir_cost = dir_cost / "runs" / rid_cost
    m_cost = json.loads((run_dir_cost / "metrics.json").read_text())

    # Net equity should be worse (or equal) with costs
    assert m_cost["ending_equity"] <= m_zero["ending_equity"], (
        f"Net equity with costs ({m_cost['ending_equity']}) should be "
        f"<= zero-cost ({m_zero['ending_equity']})"
    )

    # Gross should equal the zero-cost net (same mid fills)
    assert abs(m_cost["total_gross_pnl"] - m_zero["total_pnl"]) < 1e-10, (
        "Gross PnL with costs should equal zero-cost PnL"
    )

    # Cost PnL should be positive
    assert m_cost["total_cost_pnl"] >= 0, "total_cost_pnl must be >= 0"

    # Artifact existence
    assert (run_dir_cost / "equity_gross.csv").exists()
    assert (run_dir_cost / "plots" / "gross_vs_net_equity.png").exists()

    trades = pd.read_csv(run_dir_cost / "trades.csv")
    if not trades.empty:
        assert (trades["cost_pnl"] >= -1e-12).all(), "Per-trade cost must be >= 0"
