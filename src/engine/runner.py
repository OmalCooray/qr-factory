"""Backtest runner — orchestrates the trading pipeline.

Pipeline stages:
  1. Configuration
  2. Run setup
  3. Data loading
  4. Data validation & preparation
  5. Features
  6. Replay (model → signal → decision → execution, per bar)
  7. Metrics
  8. Artifacts
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.engine.core import TradingEngine
from src.engine.git_info import get_git_info
from src.indicators.core.pipeline import FeaturePipeline, FeatureSpec
from src.replay.bar_iterator import BarIterator
from src.replay.validation import validate_bars
from src.reporting.plots import plot_close_price, plot_equity
from src.risk import RiskConfig, RiskManager
from src.strategy.base import Strategy
from src.strategy.registry import build_strategy

log = logging.getLogger(__name__)

# Repo root (two levels up from src/engine/runner.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── 1. CONFIGURATION ────────────────────────────────────────────────────────


def _load_config(
    config_path: str,
) -> tuple[Path, dict, Strategy, list[FeatureSpec], RiskConfig]:
    """Parse YAML, build strategy, extract risk config.

    Returns (cfg_path, cfg_dict, strategy, feature_specs, risk_config).
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = _REPO_ROOT / cfg_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Strategy
    strategy_cfg = cfg.get("strategy")
    if strategy_cfg is None:
        raise ValueError("Config must contain a 'strategy' block")
    strategy, feature_specs = build_strategy(strategy_cfg)

    # Risk
    risk_config = RiskConfig(
        max_drawdown_pct=cfg.get("max_drawdown_pct", None),
        daily_dd_limit=cfg.get("daily_drawdown_pct", None),
        monthly_dd_limit=cfg.get("monthly_drawdown_pct", None),
    )

    return cfg_path, cfg, strategy, feature_specs, risk_config


# ── 2. RUN SETUP ────────────────────────────────────────────────────────────


def _create_run_dir(cfg: dict) -> tuple[str, Path]:
    """Create timestamped run directory and subdirectories.

    Returns (run_id, run_dir).
    """
    output_dir = _REPO_ROOT / cfg["output_dir"]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data_snapshot").mkdir(exist_ok=True)
    return run_id, run_dir


# ── 3. DATA ─────────────────────────────────────────────────────────────────


def _load_market_data(snapshot_dir: str | Path) -> tuple[pd.DataFrame, list[Path]]:
    """Load and concatenate all CSVs from a snapshot directory.

    Handles volume column aliasing (tick_volume / real_volume → volume).

    Returns (dataframe, csv_file_list).
    """
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = _REPO_ROOT / snapshot_dir

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

    # Handle missing volume column
    if "volume" not in df.columns:
        if "tick_volume" in df.columns:
            log.info("Aliasing 'tick_volume' to 'volume'")
            df["volume"] = df["tick_volume"]
        elif "real_volume" in df.columns:
            log.info("Aliasing 'real_volume' to 'volume'")
            df["volume"] = df["real_volume"]
        else:
            log.warning("No volume column found; validation may fail.")

    return df, csv_files


# ── 6. REPLAY ────────────────────────────────────────────────────────────────


def _run_replay(
    engine: TradingEngine,
    clean_df: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Iterate bars, calling engine.process_bar() for each."""
    steps = len(clean_df)
    log.info("Starting replay...")
    for i in range(steps):
        engine.process_bar(
            bar_row=clean_df.iloc[i],
            features_row=features.iloc[i],
            bar_index=i,
            is_last=(i == steps - 1),
        )


# ── 7. METRICS ───────────────────────────────────────────────────────────────


def _compute_metrics(
    engine: TradingEngine,
    bars: BarIterator,
    clean_df: pd.DataFrame,
    cfg: dict,
    git: dict,
    run_id: str,
) -> dict[str, Any]:
    """Compute trade stats, risk metrics, and cross-validate drawdown."""
    equity_df = pd.DataFrame(engine.equity_records)
    trades_df = pd.DataFrame([f.to_dict() for f in engine.all_fills])

    # Cross-validate DrawdownTracker against numpy
    if not equity_df.empty:
        equity_values = equity_df["equity"].values
        rolling_peak = np.maximum.accumulate(equity_values)
        recomputed = float(
            np.where(
                rolling_peak > 0,
                (rolling_peak - equity_values) / rolling_peak * 100,
                0.0,
            ).max()
        )
        assert abs(engine.risk_manager.metrics()["max_drawdown_pct"] - recomputed) < 1e-10, (
            f"Drawdown cross-validation failed: "
            f"tracker={engine.risk_manager.metrics()['max_drawdown_pct']}, "
            f"recomputed={recomputed}"
        )

    start_ts = clean_df["time"].iloc[0].isoformat()
    end_ts = clean_df["time"].iloc[-1].isoformat()

    n_trades = len(trades_df)
    total_pnl = trades_df["pnl"].sum() if not trades_df.empty else 0.0
    win_rate = (trades_df["pnl"] > 0).mean() if not trades_df.empty else 0.0
    avg_win = trades_df[trades_df["pnl"] > 0]["pnl"].mean() if not trades_df.empty else 0.0
    avg_loss = trades_df[trades_df["pnl"] < 0]["pnl"].mean() if not trades_df.empty else 0.0

    if pd.isna(avg_win):
        avg_win = 0.0
    if pd.isna(avg_loss):
        avg_loss = 0.0

    return {
        "run_id": run_id,
        "symbol": cfg["symbol"],
        "timeframe": cfg["timeframe"],
        "n_bars": len(bars),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "starting_capital": engine.starting_capital,
        "ending_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else engine.starting_capital,
        "n_trades": int(n_trades),
        "total_pnl": float(total_pnl),
        "win_rate": float(win_rate),
        "average_win": float(avg_win),
        "average_loss": float(avg_loss),
        **engine.risk_manager.metrics(),
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
    }


# ── 8. ARTIFACTS ─────────────────────────────────────────────────────────────


def _write_artifacts(
    run_dir: Path,
    cfg_path: Path,
    engine: TradingEngine,
    clean_df: pd.DataFrame,
    csv_files: list[Path],
    cfg: dict,
    metrics: dict,
    strategy: Strategy,
    config_path_str: str,
) -> None:
    """Write all run artifacts: equity, trades, metrics, plots, DATA_REF, README."""
    equity_df = pd.DataFrame(engine.equity_records)
    trades_df = pd.DataFrame([f.to_dict() for f in engine.all_fills])

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
        "row_count": len(clean_df),
        "files": file_entries,
        "files_hash_sha256": snapshot_hash,
    }
    (run_dir / "data_snapshot" / "DATA_REF.json").write_text(
        json.dumps(data_ref, indent=2), encoding="utf-8",
    )
    log.info("Wrote DATA_REF.json")

    # 7. README.md
    n_trades = metrics["n_trades"]
    total_pnl = metrics["total_pnl"]
    readme_lines = [
        f"{strategy.name} Backtest",
        f"Symbol: {cfg['symbol']} {cfg['timeframe']}",
        f"Run ID: {metrics['run_id']}",
        f"Trades: {n_trades}, Total PnL: {total_pnl:.2f}",
        f"Reproduce: uv run python -m src backtest --config {config_path_str}",
    ]
    (run_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8",
    )
    log.info("Wrote README.md")


# ── PUBLIC ENTRY POINT ───────────────────────────────────────────────────────


def run_backtest(config_path: str) -> str:
    """Execute a deterministic backtest and write all run artifacts.

    Pipeline: config → data → validate → features → replay → metrics → artifacts.

    Parameters
    ----------
    config_path : str
        Path to a YAML config file (relative to repo root or absolute).

    Returns
    -------
    str
        The generated ``run_id``.
    """
    # ── 1. CONFIGURATION ──
    cfg_path, cfg, strategy, feature_specs, risk_config = _load_config(config_path)

    # ── 2. RUN SETUP ──
    run_id, run_dir = _create_run_dir(cfg)
    git = get_git_info(_REPO_ROOT)

    log.info("Run ID   : %s", run_id)
    log.info("Output   : %s", run_dir)
    log.info("Git      : %s (dirty=%s)", git["git_commit"][:8], git["git_dirty"])

    # ── 3. DATA ──
    df, csv_files = _load_market_data(cfg["snapshot_dir"])

    # ── 4. DATA VALIDATION & PREPARATION ──
    validate_bars(df)
    log.info("Data validation passed ✓")

    bars = BarIterator(df)
    clean_df = bars.df

    # ── 5. FEATURES ──
    log.info("Computing features...")
    pipeline = FeaturePipeline(feature_specs)
    X = pipeline.transform(clean_df)

    if len(X) != len(clean_df):
        raise ValueError("Feature DataFrame length mismatch")
    log.info("Features computed: %s", list(X.columns))

    # ── 6. REPLAY (model → signal → decision → execution, per bar) ──
    starting_capital: float = cfg["starting_capital"]
    position_size: float = cfg.get("position_size", 1.0)
    risk_manager = RiskManager(risk_config, starting_capital)

    engine = TradingEngine(strategy, risk_manager, starting_capital, position_size)
    _run_replay(engine, clean_df, X)

    # ── 7. METRICS ──
    metrics = _compute_metrics(engine, bars, clean_df, cfg, git, run_id)

    # ── 8. ARTIFACTS ──
    _write_artifacts(run_dir, cfg_path, engine, clean_df, csv_files, cfg, metrics, strategy, config_path)

    log.info("✓ Run complete: %s", run_dir)
    return run_id
