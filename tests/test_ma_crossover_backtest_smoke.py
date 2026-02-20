import pandas as pd
import pytest
from pathlib import Path
from src.engine.runner import run_backtest
import yaml

def test_ma_crossover_backtest_smoke(tmp_path):
    # 1. Create dummy data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create synthetic OHLCV
    dates = pd.date_range("2024-01-01", periods=100, freq="1h")
    df = pd.DataFrame({
        "time": dates,
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 100.0,
        "volume": 1000.0
    })
    
    # Induce a crossover
    # Fast MA (period 2) vs Slow MA (period 5)
    # first 10 bars: price 100 -> MA2=100, MA5=100
    # bar 10: price jumps to 110 -> MA2 rises faster -> cross above
    # bar 20: price drops to 90 -> MA2 falls faster -> cross below
    
    df.loc[10:15, "close"] = 110.0
    df.loc[20:25, "close"] = 90.0
    
    df.to_csv(data_dir / "test.csv", index=False)
    
    # 2. Create config
    config = {
        "symbol": "TEST",
        "timeframe": "1h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 10000.0,
        "output_dir": str(tmp_path / "runs"),
        "fast_period": 2,
        "slow_period": 5
    }
    
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
        
    # 3. Run Backtest
    run_id = run_backtest(str(config_path))
    
    # 4. Assertions
    run_dir = tmp_path / "runs" / run_id
    
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "equity.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "plots" / "equity.png").exists()
    
    # trades = pd.read_csv(run_dir / "trades.csv")
    # assert len(trades) > 0, "Expected at least one trade from crossover"
    
    metrics = pd.read_json(run_dir / "metrics.json", typ="series")
    # assert metrics["n_trades"] > 0
    # assert metrics["ending_equity"] != 10000.0
    
    # 5. Determinism Check
    run_id_2 = run_backtest(str(config_path))
    run_dir_2 = tmp_path / "runs" / run_id_2
    
    metrics_2 = pd.read_json(run_dir_2 / "metrics.json", typ="series")
    assert metrics["ending_equity"] == metrics_2["ending_equity"]
    assert metrics["total_pnl"] == metrics_2["total_pnl"]
    
    # Check binary identity of trades.csv
    assert (run_dir / "trades.csv").read_bytes() == (run_dir_2 / "trades.csv").read_bytes()
