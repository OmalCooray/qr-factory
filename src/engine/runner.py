"""Backtest runner — orchestrates load → replay → artifact generation."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.engine.git_info import get_git_info
from src.replay.bar_iterator import BarIterator
from src.replay.validation import validate_bars
from src.reporting.plots import plot_close_price, plot_equity
from src.indicators.core.pipeline import FeaturePipeline, FeatureSpec
from src.indicators.impl.ma import SMA
from src.strategy.ma_crossover import MACrossoverStrategy
from src.strategy.base import StrategyContext

log = logging.getLogger(__name__)

# Repo root (two levels up from src/engine/runner.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def run_backtest(config_path: str) -> str:
    """Execute a deterministic backtest and write all run artifacts.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file (relative to repo root or absolute).

    Returns
    -------
    str
        The generated ``run_id``.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = _REPO_ROOT / cfg_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    symbol: str = cfg["symbol"]
    timeframe: str = cfg["timeframe"]
    snapshot_dir = _REPO_ROOT / cfg["snapshot_dir"]
    starting_capital: float = cfg["starting_capital"]
    output_dir = _REPO_ROOT / cfg["output_dir"]
    
    # Strategy parameters from config
    fast_period = cfg.get("fast_period", 10)
    slow_period = cfg.get("slow_period", 30)
    # Define feature names based on SMA convention: sma_{period}_close
    fast_col = f"sma_{fast_period}_close"
    slow_col = f"sma_{slow_period}_close"

    # ── Generate run_id ──────────────────────────────────────────────
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data_snapshot").mkdir(exist_ok=True)

    # ── Capture git state ────────────────────────────────────────────
    git = get_git_info(_REPO_ROOT)

    log.info("Run ID   : %s", run_id)
    log.info("Output   : %s", run_dir)
    log.info("Git      : %s (dirty=%s)", git["git_commit"][:8], git["git_dirty"])

    # ── Load data ────────────────────────────────────────────────────
    csv_files = sorted(snapshot_dir.glob("*.csv"))
    if not csv_files:
        log.error("No CSV files found in %s", snapshot_dir)
        sys.exit(1)

    frames = []
    for csv_file in csv_files:
        df_part = pd.read_csv(csv_file)
        frames.append(df_part)
        log.info("  Loaded %s  (%s rows)", csv_file.name, f"{len(df_part):,}")

    df = pd.concat(frames, ignore_index=True)
    log.info("Total raw rows: %s", f"{len(df):,}")

    # ── Handle missing volume column ─────────────────────────────────
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            log.info("Aliasing 'tick_volume' to 'volume'")
            df["volume"] = df["tick_volume"]
        elif "real_volume" in df.columns:
            log.info("Aliasing 'real_volume' to 'volume'")
            df["volume"] = df["real_volume"]
        else:
            log.warning("No volume column found; validation may fail.")

    # ── Validate data integrity (fail-fast) ──────────────────────────
    validate_bars(df)
    log.info("Data validation passed ✓")
    
    # ── Feature Computation ──────────────────────────────────────────
    log.info("Computing features...")
    pipeline = FeaturePipeline([
        FeatureSpec(SMA(fast_period), []),
        FeatureSpec(SMA(slow_period), []),
    ])
    
    # Ensure OHLCV has correct index for alignment if needed, but pipeline expects RangeIndex usually
    # transform returns a DataFrame with same index as input
    X = pipeline.transform(df)
    
    # Verify alignment
    if len(X) != len(df):
        raise ValueError("Feature DataFrame length mismatch")
    
    log.info("Features computed: %s", list(X.columns))

    # ── Build bar iterator (validates + sorts) ───────────────────────
    # Note: BarIterator sorts df by time. We must ensure X aligns.
    # Approach: 
    # 1. BarIterator cleans and sorts df.
    # 2. We should probably compute features ON the sorted df to be safe,
    #    OR ensure we reindex X to match the sorted df.
    #    Given BarIterator takes df in constructor and does validation/sorting internally, 
    #    we should let it do its job, get the clean df, AND THEN compute features.
    
    bars = BarIterator(df)
    clean_df = bars.df  # This is the sorted, validated dataframe
    
    # Re-compute features on the clean, sorted DataFrame to guarantee strict alignment
    X = pipeline.transform(clean_df)

    # ── Strategy Initialization ──────────────────────────────────────
    strategy = MACrossoverStrategy(
        fast_col=fast_col,
        slow_col=slow_col,
        size=1.0,  # Fixed size per requirements
        _warmup_bars=slow_period
    )

    # ── Replay Loop ──────────────────────────────────────────────────
    equity = starting_capital
    position = 0.0
    entry_price = 0.0  # Average entry price
    
    trades = []
    equity_records = []
    
    # Execution state
    target_position = 0.0
    
    # For loop implementation of the replay
    # We iterate through the clean_df directly to have index access for features
    
    log.info("Starting replay...")
    
    steps = len(clean_df)
    for i in range(steps):
        bar_row = clean_df.iloc[i]
        bar_ts = bar_row["time"]
        
        # 1. Execution Phase (at Open)
        # Execute target from PREVIOUS bar against CURRENT Open
        # Rule: All position changes execute at the next bar's OPEN price.
        current_open = bar_row["open"]
        
        executed_qty = 0.0
        trade_pnl = 0.0
        
        if position != target_position:
            # We need to adjust position
            # 1. Close existing if needed (flip or flatten)
            if position != 0:
                # Closing entire position logic for flip/flatten simplicity
                # Or just net change. 
                # Task says: "If target != current: Close existing ... Open new..."
                # effectively a flip if signs differ, or close if target is 0.
                
                # Let's handle it as net quantity change for PnL calculation simplicity?
                # "A trade is completed when: position flips OR position goes from non-zero to zero"
                
                # Case 1: Flattening or Flipping -> Close current position
                if (position > 0 and target_position <= 0) or (position < 0 and target_position >= 0):
                    # Close entire current position
                    qty_to_close = -position
                    exit_price = current_open
                    
                    # Calculate PnL
                    # long: (exit - entry) * size
                    # short: (entry - exit) * size
                    if position > 0:
                        pnl = (exit_price - entry_price) * abs(position)
                        side = "long"
                    else:
                        pnl = (entry_price - exit_price) * abs(position)
                        side = "short"
                        
                    equity += pnl
                    trades.append({
                        "entry_ts": current_trade_entry_ts, # We need to track this
                        "exit_ts": bar_ts.isoformat(),
                        "side": side,
                        "qty": abs(position),
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl
                    })
                    
                    position = 0.0
                    entry_price = 0.0

            # Case 2: Opening new position (or flipping into new)
            if target_position != 0 and position == 0:
                position = target_position
                entry_price = current_open
                current_trade_entry_ts = bar_ts.isoformat()
        
        # 2. Strategy Phase (on Close)
        # Prepare context for THIS bar
        
        # Current Unrealized PnL (Mark to Market at Close)
        unrealized_pnl = 0.0
        current_close = bar_row["close"]
        if position != 0:
            if position > 0:
                unrealized_pnl = (current_close - entry_price) * abs(position)
            else:
                unrealized_pnl = (entry_price - current_close) * abs(position)
        
        current_equity = equity + unrealized_pnl
        
        # Record state
        equity_records.append({
            "timestamp": bar_ts.isoformat(),
            "equity": current_equity,
            "position": position,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": equity - starting_capital 
        })
        
        # Check for last bar - do not attempt next-bar execution
        if i == steps - 1:
            break
            
        # Call Strategy
        ctx = StrategyContext(
            ts=bar_ts,
            bar=bar_row,
            features=X.iloc[i],
            position=position,
            equity=current_equity,
            bar_index=i
        )
        
        
        decision = strategy.on_bar(ctx)
        target_position = decision.target_position
        


    # ── Post-Replay Data Construction ────────────────────────────────
    equity_df = pd.DataFrame(equity_records)
    trades_df = pd.DataFrame(trades)
    
    # ── Write artifacts ──────────────────────────────────────────────
    # 1. config.yaml
    shutil.copy2(cfg_path, run_dir / "config.yaml")

    # 2. equity.csv
    equity_df.to_csv(run_dir / "equity.csv", index=False)
    log.info("Wrote equity.csv  (%s rows)", f"{len(equity_df):,}")
    
    # 3. trades.csv
    if not trades_df.empty:
        trades_df = trades_df.sort_values("entry_ts")
        trades_df.to_csv(run_dir / "trades.csv", index=False)
        log.info("Wrote trades.csv  (%s trades)", f"{len(trades_df):,}")
    else:
        (run_dir / "trades.csv").touch()
        log.info("Wrote empty trades.csv")

    # 4. metrics.json
    start_ts = clean_df["time"].iloc[0].isoformat()
    end_ts = clean_df["time"].iloc[-1].isoformat()
    
    # Helper stats
    n_trades = len(trades_df)
    total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
    win_rate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0.0
    avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if not trades_df.empty else 0.0
    avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if not trades_df.empty else 0.0
    
    # Fill NaN means with 0
    if pd.isna(avg_win): avg_win = 0.0
    if pd.isna(avg_loss): avg_loss = 0.0
    
    metrics = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "n_bars": len(bars),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "starting_capital": starting_capital,
        "ending_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else starting_capital,
        "n_trades": int(n_trades),
        "total_pnl": float(total_pnl),
        "win_rate": float(win_rate),
        "average_win": float(avg_win),
        "average_loss": float(avg_loss),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    log.info("Wrote metrics.json")

    # 5. plots
    plot_close_price(clean_df, run_dir / "plots" / "close_price.png")
    plot_equity(equity_df, run_dir / "plots" / "equity.png")

    # 6. data_snapshot/DATA_REF.json
    snapshot_rel = cfg["snapshot_dir"]
    file_entries = []
    hash_parts = []
    for csv_file in csv_files:
        size = csv_file.stat().st_size
        file_entries.append({"name": csv_file.name, "size_bytes": size})
        hash_parts.append(f"{csv_file.name}:{size}")

    snapshot_hash = hashlib.sha256(
        "|".join(hash_parts).encode("utf-8"),
    ).hexdigest()

    data_ref = {
        "snapshot_path": snapshot_rel,
        "row_count": len(bars),
        "files": file_entries,
        "files_hash_sha256": snapshot_hash,
    }
    (run_dir / "data_snapshot" / "DATA_REF.json").write_text(
        json.dumps(data_ref, indent=2), encoding="utf-8",
    )
    log.info("Wrote DATA_REF.json")

    # 7. README.md
    readme_lines = [
        "MA Crossover Backtest",
        f"Symbol: {symbol} {timeframe}",
        f"Parameters: fast={fast_period}, slow={slow_period}",
        f"Run ID: {run_id}",
        f"Trades: {n_trades}, Total PnL: {total_pnl:.2f}",
        f"Reproduce: uv run python -m src backtest --config {config_path}",
    ]
    (run_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8",
    )
    log.info("Wrote README.md")

    log.info("✓ Run complete: %s", run_dir)
    return run_id
