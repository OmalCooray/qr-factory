"""Integration test for replay determinism."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.engine.runner import run_backtest

# Use a module-level logger
log = logging.getLogger(__name__)


def _create_synthetic_data(data_dir: Path, n_rows: int = 2000) -> None:
    """Generate a synthetic OHLC CSV file."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # 5-minute intervals starting from a fixed time
    times = pd.date_range("2024-01-01", periods=n_rows, freq="5min", tz="UTC")
    
    df = pd.DataFrame({
        "time": times,
        "open": 100.0,
        "high": 105.0,
        "low": 95.0,
        "close": 101.0,  # distinct from open
        "spread": 1,
        "tick_volume": 100,
    })
    
    # Write to CSV
    csv_path = data_dir / "synthetic_bars.csv"
    df.to_csv(csv_path, index=False)
    log.info("Created synthetic data at %s with %d rows", csv_path, n_rows)


def _load_metrics(run_dir: Path) -> dict:
    """Load metrics.json from a run directory."""
    metrics_path = run_dir / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def test_replay_determinism(tmp_path: Path) -> None:
    """Verify that two runs with identical config/data produce identical results.
    
    This ensures that:
    1. The data loading and iteration order is stable.
    2. The engine execution is deterministic.
    3. Artifact generation is consistent.
    """
    # 1. Setup directories
    # We use a temp directory for everything so we don't pollute the real runs/ folder
    base_dir = tmp_path / "determinism_test"
    data_dir = base_dir / "data"
    output_dir = base_dir / "runs"
    
    # 2. Generate synthetic data
    _create_synthetic_data(data_dir)
    
    # 3. Create config
    config = {
        "symbol": "TEST_SYM",
        "timeframe": "M5",
        "snapshot_dir": str(data_dir),  # Absolute path handled by runner or relative
        "starting_capital": 10000.0,
        "output_dir": str(output_dir),
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 10,
            "slow_period": 30,
        },
    }
    
    config_path = base_dir / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    log.info("Config written to %s", config_path)
    
    # 4. Run 1
    log.info("Starting Run 1...")
    run_id_1 = run_backtest(str(config_path))
    time.sleep(1.1)  # Ensure at least 1 second diff for run_id timestamp
    
    # 5. Run 2
    log.info("Starting Run 2...")
    run_id_2 = run_backtest(str(config_path))
    
    assert run_id_1 != run_id_2, "Run IDs should be unique (time-based)"
    
    # 6. Compare results
    run_dir_1 = output_dir / run_id_1
    run_dir_2 = output_dir / run_id_2
    
    # A. Compare Equity Curve lengths and content
    eq1 = pd.read_csv(run_dir_1 / "equity.csv")
    eq2 = pd.read_csv(run_dir_2 / "equity.csv")
    
    pd.testing.assert_frame_equal(eq1, eq2, obj="equity.csv")
    log.info("Equity curves identical.")
    
    # B. Compare Metrics
    m1 = _load_metrics(run_dir_1)
    m2 = _load_metrics(run_dir_2)
    
    # Fields that MUST be identical
    keys_to_compare = [
        "symbol", "timeframe", "n_bars",
        "start_ts", "end_ts",
        "starting_capital", "ending_equity",
        "total_pnl", "n_trades",
        "max_drawdown_pct", "risk_halted",
    ]
    
    for key in keys_to_compare:
        assert m1[key] == m2[key], f"Mismatch in metric '{key}': {m1[key]} vs {m2[key]}"
        
    log.info("Metrics identical.")
    
    # C. Compare Data Snapshot Hash (DATA_REF.json)
    ref1 = json.loads((run_dir_1 / "data_snapshot" / "DATA_REF.json").read_text("utf-8"))
    ref2 = json.loads((run_dir_2 / "data_snapshot" / "DATA_REF.json").read_text("utf-8"))
    
    assert ref1["files_hash_sha256"] == ref2["files_hash_sha256"], "Data snapshot hash mismatch"
    
    log.info("Determinism test passed.")
