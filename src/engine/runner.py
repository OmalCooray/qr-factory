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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src._paths import REPO_ROOT
from src.engine.core import TradingEngine
from src.engine.git_info import get_git_info
from src.execution.costs import CostConfig
from src.indicators.core.pipeline import FeaturePipeline, FeatureSpec
from src.replay.bar_iterator import BarIterator
from src.replay.validation import validate_bars
from src.reporting.plots import plot_close_price, plot_equity, plot_fold_pnl, plot_gross_vs_net_equity
from src.risk import RiskConfig, RiskManager
from src.strategy.base import Strategy
from src.strategy.registry import build_strategy

log = logging.getLogger(__name__)


# ── TYPED STRUCTURES ────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    """Typed structure for per-fold aggregation results.

    Replaces ad-hoc dict usage with explicit, typed attributes.
    """

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    test_bars: int
    starting_capital: float
    n_trades: int
    gross_pnl: float
    net_pnl: float
    max_drawdown_pct: float
    ending_equity: float
    win_rate: float
    # Optional ML metrics (set only when ML config present)
    brier: float | None = None
    log_loss: float | None = None
    n_train_rows: int | None = None
    n_test_rows: int | None = None
    cal_n_train: int | None = None
    cal_n_cal: int | None = None


# ── 1. CONFIGURATION ────────────────────────────────────────────────────────


def _load_config(
    config_path: str,
) -> tuple[Path, dict, Strategy, list[FeatureSpec], RiskConfig, CostConfig]:
    """Parse YAML, build strategy, extract risk and cost configs.

    Returns (cfg_path, cfg_dict, strategy, feature_specs, risk_config, cost_config).
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Validate required config keys
    required_keys = ["symbol", "timeframe", "snapshot_dir", "output_dir", "starting_capital", "strategy"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")

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

    # Execution costs (defaults to zero if section missing)
    exec_cfg = cfg.get("execution", {})
    cost_config = CostConfig(
        spread_points=exec_cfg.get("spread_points", 0.0),
        slippage_points=exec_cfg.get("slippage_points", 0.0),
        point_value=exec_cfg.get("point_value", 1.0),
    )

    return cfg_path, cfg, strategy, feature_specs, risk_config, cost_config


# ── 2. RUN SETUP ────────────────────────────────────────────────────────────


def _create_run_dir(cfg: dict) -> tuple[str, Path]:
    """Create timestamped run directory and subdirectories.

    Returns (run_id, run_dir).
    """
    output_dir = REPO_ROOT / cfg["output_dir"]
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
        snapshot_dir = REPO_ROOT / snapshot_dir

    csv_files = sorted(snapshot_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {snapshot_dir}")

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
    log_every = max(1, steps // 10)
    log.info("Starting replay (%s bars)...", f"{steps:,}")
    for i in range(steps):
        engine.process_bar(
            bar_row=clean_df.iloc[i],
            features_row=features.iloc[i],
            bar_index=i,
            is_last=(i == steps - 1),
        )
        if (i + 1) % log_every == 0:
            log.info("  Replay progress: %s/%s (%d%%)", f"{i+1:,}", f"{steps:,}", (i+1)*100//steps)


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
        tracker_dd = engine.risk_manager.metrics()["max_drawdown_pct"]
        if abs(tracker_dd - recomputed) >= 1e-10:
            raise RuntimeError(
                f"Drawdown cross-validation failed: "
                f"tracker={tracker_dd}, recomputed={recomputed}"
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

    # Cost breakdown
    total_gross_pnl = float(trades_df["gross_pnl"].sum()) if not trades_df.empty else 0.0
    total_net_pnl = float(total_pnl)
    total_cost_pnl = total_gross_pnl - total_net_pnl
    avg_cost_per_trade = total_cost_pnl / max(1, n_trades)
    cost_as_pct_of_gross_abs = (
        total_cost_pnl / max(1e-12, abs(total_gross_pnl)) * 100
    )

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
        "total_gross_pnl": total_gross_pnl,
        "total_net_pnl": total_net_pnl,
        "total_cost_pnl": float(total_cost_pnl),
        "avg_cost_per_trade": float(avg_cost_per_trade),
        "cost_as_pct_of_gross_abs": float(cost_as_pct_of_gross_abs),
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

    # 2. equity.csv (NET — excludes equity_gross column)
    equity_net_df = equity_df.drop(columns=["equity_gross"])
    equity_net_df.to_csv(run_dir / "equity.csv", index=False)
    log.info("Wrote equity.csv  (%s rows)", f"{len(equity_net_df):,}")

    # 2b. equity_gross.csv (GROSS)
    equity_gross_df = equity_df[["timestamp", "equity_gross"]].rename(
        columns={"equity_gross": "equity"}
    )
    equity_gross_df.to_csv(run_dir / "equity_gross.csv", index=False)
    log.info("Wrote equity_gross.csv  (%s rows)", f"{len(equity_gross_df):,}")

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
    plot_equity(equity_net_df, run_dir / "plots" / "equity.png")
    plot_gross_vs_net_equity(equity_df, run_dir / "plots" / "gross_vs_net_equity.png")

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

    # 7. Strategy-specific extra artifacts (explicit method call, not attribute access)
    if hasattr(strategy, "extra_artifacts") and callable(strategy.extra_artifacts):
        extra = strategy.extra_artifacts()
        for artifact_name, records in extra.items():
            if records:
                artifact_df = pd.DataFrame(records)
                artifact_path = run_dir / f"{artifact_name}.csv"
                artifact_df.to_csv(artifact_path, index=False)
                log.info("Wrote %s  (%s rows)", artifact_path.name, f"{len(artifact_df):,}")

    # 8. README.md
    n_trades = metrics["n_trades"]
    total_pnl = metrics["total_pnl"]
    total_cost = metrics["total_cost_pnl"]
    readme_lines = [
        f"{strategy.name} Backtest (spread+slippage cost model)",
        f"Symbol: {cfg['symbol']} {cfg['timeframe']}",
        f"Run ID: {metrics['run_id']}",
        f"Trades: {n_trades}, Net PnL: {total_pnl:.2f}, Cost: {total_cost:.2f}",
        f"Reproduce: uv run python -m src backtest --config {config_path_str}",
        f"Artifacts: gross_vs_net_equity.png, metrics.json (cost breakdown)",
        f"Sanity: cost=0 => net_pnl == gross_pnl for all trades",
    ]
    (run_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8",
    )
    log.info("Wrote README.md")


# ── WALK-FORWARD HELPERS ─────────────────────────────────────────────────────


def _run_ml_for_fold(
    ml_cfg: dict,
    label_cfg: dict,
    cal_cfg: dict,
    train_bars: pd.DataFrame,
    train_features: pd.DataFrame,
    test_bars: pd.DataFrame,
    test_features: pd.DataFrame,
    fold_id: int,
    clean_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Run ML training, calibration, and prediction for a single fold.

    Parameters
    ----------
    ml_cfg : dict
        Model configuration
    label_cfg : dict
        Label generation configuration
    cal_cfg : dict
        Calibration configuration
    train_bars, train_features : pd.DataFrame
        Training data
    test_bars, test_features : pd.DataFrame
        Test data
    fold_id : int
        Current fold index
    clean_df : pd.DataFrame
        Full dataset (for timestamp lookups in predictions)

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]
        (test_features_with_yhat, ml_fold_metrics, pred_records)
    """
    from src.ml.frame import build_forecast_frame
    from src.ml.predictor import build_predictor

    train_ff = build_forecast_frame(train_bars, train_features, label_cfg)
    test_ff = build_forecast_frame(test_bars, test_features, label_cfg)

    # ── Calibration path ──
    cal_method = cal_cfg.get("method", "none")
    cal_active = cal_method != "none"

    if cal_active:
        from src.ml.calibration import PlattCalibrator

        cal_fraction = cal_cfg.get("cal_fraction", 0.2)
        n_train = len(train_ff)
        n_model_train = int(n_train * (1 - cal_fraction))
        model_train = train_ff.iloc[:n_model_train]
        cal_slice = train_ff.iloc[n_model_train:]

        # Fit predictor on model_train only
        predictor = build_predictor(ml_cfg)
        predictor.fit(model_train)

        # Fit calibrator on cal_slice
        cal_raw_probs = predictor.predict(cal_slice)
        calibrator = PlattCalibrator()
        calibrator.fit(cal_raw_probs.values, cal_slice["y"].values)

        # Get raw + calibrated probs on test set
        yhat_raw = predictor.predict(test_ff)
        yhat = calibrator.transform_series(yhat_raw)

        log.info(
            "    Calibration: model_train=%d, cal_slice=%d, method=%s",
            len(model_train), len(cal_slice), cal_method,
        )
    else:
        predictor = build_predictor(ml_cfg)
        predictor.fit(train_ff)
        yhat = predictor.predict(test_ff)
        yhat_raw = None

    # ── Per-fold ML metrics ──
    y_true = test_ff["y"].values
    y_prob = yhat.reindex(test_ff.index).values
    brier = float(np.mean((y_prob - y_true) ** 2))

    try:
        from sklearn.metrics import log_loss as sk_log_loss
        ll = float(sk_log_loss(y_true, y_prob))
    except Exception:
        ll = None

    ml_fold_metrics: dict[str, Any] = {
        "brier": brier,
        "n_train_rows": len(train_ff) if not cal_active else len(model_train),
        "n_test_rows": len(test_ff),
    }
    if ll is not None:
        ml_fold_metrics["log_loss"] = ll
    if cal_active:
        ml_fold_metrics["cal_n_train"] = len(model_train)
        ml_fold_metrics["cal_n_cal"] = len(cal_slice)

    # Inject predictions into features so strategy sees ctx.features["yhat"]
    test_features_out = test_features.copy()
    test_features_out["yhat"] = np.nan
    test_features_out.loc[yhat.index, "yhat"] = yhat

    # Persist per-fold predictions for evaluation
    pred_records = pd.DataFrame({
        "fold_id": fold_id,
        "timestamp": [clean_df["time"].iloc[idx].isoformat() for idx in test_ff.index],
        "y_true": y_true,
        "y_prob": y_prob,
    })
    if cal_active and yhat_raw is not None:
        pred_records["y_prob_raw"] = yhat_raw.reindex(test_ff.index).values

    log.info(
        "    ML: predictor=%s, train_ff=%d rows, test_ff=%d rows, yhat=%d",
        ml_cfg.get("type", "?"), len(train_ff), len(test_ff), len(yhat),
    )

    return test_features_out, ml_fold_metrics, pred_records


def _compute_fold_metrics(
    fold_id: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    test_start: int,
    fold_starting_capital: float,
    engine: TradingEngine,
    ml_cfg: dict | None,
    ml_fold_metrics: dict[str, Any],
) -> FoldResult:
    """Compute metrics for a single fold after replay.

    Parameters
    ----------
    fold_id : int
        Current fold index
    train_idx, test_idx : np.ndarray
        Train and test index arrays
    test_start : int
        Test slice start index
    fold_starting_capital : float
        Starting capital for this fold
    engine : TradingEngine
        Trading engine with filled trades and equity records
    ml_cfg : dict | None
        ML configuration (None if no ML)
    ml_fold_metrics : dict[str, Any]
        ML metrics dictionary (populated if ml_cfg is not None)

    Returns
    -------
    FoldResult
        Typed fold result structure
    """
    trades_df = pd.DataFrame([f.to_dict() for f in engine.all_fills])
    n_trades = len(trades_df)
    net_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    gross_pnl = float(trades_df["gross_pnl"].sum()) if not trades_df.empty else 0.0
    max_dd = engine.risk_manager.metrics()["max_drawdown_pct"]
    eq_df = pd.DataFrame(engine.equity_records)
    ending_equity = (
        float(eq_df["equity"].iloc[-1])
        if not eq_df.empty
        else fold_starting_capital
    )
    win_rate = float((trades_df["pnl"] > 0).mean()) if n_trades > 0 else 0.0

    # Build typed FoldResult
    return FoldResult(
        fold_id=fold_id,
        train_start=int(train_idx[0]),
        train_end=int(train_idx[-1]),
        test_start=test_start,
        test_end=int(test_idx[-1]),
        test_bars=len(test_idx),
        starting_capital=fold_starting_capital,
        n_trades=n_trades,
        gross_pnl=gross_pnl,
        net_pnl=net_pnl,
        max_drawdown_pct=max_dd,
        ending_equity=ending_equity,
        win_rate=win_rate,
        # ML metrics (populated only if ML config present)
        brier=ml_fold_metrics.get("brier") if ml_cfg is not None else None,
        log_loss=ml_fold_metrics.get("log_loss") if ml_cfg is not None else None,
        n_train_rows=ml_fold_metrics.get("n_train_rows") if ml_cfg is not None else None,
        n_test_rows=ml_fold_metrics.get("n_test_rows") if ml_cfg is not None else None,
        cal_n_train=ml_fold_metrics.get("cal_n_train") if ml_cfg is not None else None,
        cal_n_cal=ml_fold_metrics.get("cal_n_cal") if ml_cfg is not None else None,
    )


# ── WALK-FORWARD ORCHESTRATION ───────────────────────────────────────────────


def _run_walk_forward(
    cfg: dict,
    cfg_path: Path,
    config_path_str: str,
    clean_df: pd.DataFrame,
    features: pd.DataFrame,
    csv_files: list[Path],
    risk_config: RiskConfig,
    cost_config: CostConfig,
    git: dict,
    run_id: str,
    run_dir: Path,
) -> str:
    """Run walk-forward backtest: multiple folds via TimeSeriesSplitter.

    Each fold gets a fresh strategy and engine.  Only the test slice is traded.
    The training slice exists for future ML model fitting.

    **Equity semantics — capital carryover:** ending equity from fold *N*
    becomes the starting capital for fold *N+1*.  This produces a single
    continuous compounded equity curve across all folds.  The aggregate
    ``ending_equity`` in metrics.json is the final fold's ending equity.

    **Feature causality assumption:** ``features`` are pre-computed on the
    full dataset *before* splitting.  This is safe only if every indicator
    and transform in the ``FeaturePipeline`` is strictly causal (uses only
    past/current bars, never future bars).  All current implementations
    (SMA, EMA, ADX, ATR, RSI, BB, Donchian, rolling Z-score, diff, lag)
    satisfy this property.  A regression test
    (``test_feature_pipeline_causality``) guards against future violations.

    Returns the run_id.
    """
    from src.validation import TimeSeriesSplitter

    val_cfg = cfg["validation"]
    splitter = TimeSeriesSplitter(
        mode=val_cfg["mode"],
        train_size=val_cfg["train_size"],
        test_size=val_cfg["test_size"],
        step=val_cfg.get("step"),
        drop_last=val_cfg.get("drop_last", True),
    )

    folds = list(splitter.split(len(clean_df)))
    if len(folds) == 0:
        raise ValueError("Splitter produced no folds — check validation config")
    log.info("Walk-forward: %d folds (%s)", len(folds), splitter)

    starting_capital: float = cfg["starting_capital"]
    position_size: float = cfg.get("position_size", 1.0)
    latency_bars: int = cfg.get("execution", {}).get("latency_bars", 0)

    fold_results: list[FoldResult] = []
    all_equity_records: list[dict] = []
    all_fills: list = []
    all_predictions: list[pd.DataFrame] = []
    fold_capital = starting_capital  # carryover: fold N+1 starts where fold N ended

    for fold_id, (train_idx, test_idx) in enumerate(folds):
        if train_idx[-1] >= test_idx[0]:
            raise ValueError(
                f"Fold {fold_id}: train/test leakage detected — "
                f"train_end={train_idx[-1]} >= test_start={test_idx[0]}"
            )

        test_start = int(test_idx[0])
        test_end = int(test_idx[-1]) + 1

        log.info(
            "  Fold %d/%d: train[%d:%d] test[%d:%d] (%d bars)",
            fold_id + 1,
            len(folds),
            int(train_idx[0]),
            int(train_idx[-1]) + 1,
            test_start,
            test_end,
            len(test_idx),
        )

        # Fresh strategy + engine per fold (capital carries over)
        fold_starting_capital = fold_capital
        strategy, _ = build_strategy(cfg["strategy"])
        risk_manager = RiskManager(risk_config, fold_capital)
        engine = TradingEngine(
            strategy, risk_manager, fold_capital, position_size, cost_config,
            latency_bars=latency_bars,
        )

        # Run replay on test slice only
        test_bars = clean_df.iloc[test_start:test_end]
        test_features = features.iloc[test_start:test_end]

        # ── Optional ML path ──
        ml_cfg = cfg.get("model")
        label_cfg = cfg.get("label")
        ml_fold_metrics: dict[str, Any] = {}

        if ml_cfg is not None:
            train_start = int(train_idx[0])
            train_end = int(train_idx[-1]) + 1
            train_bars = clean_df.iloc[train_start:train_end]
            train_features = features.iloc[train_start:train_end]
            cal_cfg = cfg.get("calibration", {})

            test_features, ml_fold_metrics, pred_records = _run_ml_for_fold(
                ml_cfg=ml_cfg,
                label_cfg=label_cfg,
                cal_cfg=cal_cfg,
                train_bars=train_bars,
                train_features=train_features,
                test_bars=test_bars,
                test_features=test_features,
                fold_id=fold_id,
                clean_df=clean_df,
            )
            all_predictions.append(pred_records)

        _run_replay(engine, test_bars, test_features)

        # ── Per-fold metrics ──
        fold_result = _compute_fold_metrics(
            fold_id=fold_id,
            train_idx=train_idx,
            test_idx=test_idx,
            test_start=test_start,
            fold_starting_capital=fold_starting_capital,
            engine=engine,
            ml_cfg=ml_cfg,
            ml_fold_metrics=ml_fold_metrics,
        )

        # Carry ending equity to next fold
        fold_capital = fold_result.ending_equity
        fold_results.append(fold_result)

        all_equity_records.extend(engine.equity_records)
        all_fills.extend(engine.all_fills)

    # ── Concatenate predictions into single predictions.csv ──
    if all_predictions:
        pd.concat(all_predictions, ignore_index=True).to_csv(
            run_dir / "predictions.csv", index=False,
        )
        log.info("Wrote predictions.csv (%d rows)", sum(len(d) for d in all_predictions))

    # ── Aggregate across folds ──
    total_trades = sum(f.n_trades for f in fold_results)
    total_net_pnl = sum(f.net_pnl for f in fold_results)
    total_gross_pnl = sum(f.gross_pnl for f in fold_results)
    total_cost_pnl = total_gross_pnl - total_net_pnl
    worst_dd = max(f.max_drawdown_pct for f in fold_results)

    all_trades_df = pd.DataFrame([f.to_dict() for f in all_fills])
    overall_win_rate = (
        float((all_trades_df["pnl"] > 0).mean())
        if not all_trades_df.empty
        else 0.0
    )

    aggregate = {
        "n_folds": len(folds),
        "total_trades": total_trades,
        "total_net_pnl": total_net_pnl,
        "total_gross_pnl": total_gross_pnl,
        "total_cost_pnl": total_cost_pnl,
        "worst_drawdown_pct": worst_dd,
        "avg_net_pnl_per_fold": total_net_pnl / len(folds),
        "overall_win_rate": overall_win_rate,
    }

    # ── Aggregate ML metrics ──
    if any(f.brier is not None for f in fold_results):
        brier_vals = [f.brier for f in fold_results if f.brier is not None]
        aggregate["avg_brier"] = float(np.mean(brier_vals))
        ll_vals = [f.log_loss for f in fold_results if f.log_loss is not None]
        if ll_vals:
            aggregate["avg_log_loss"] = float(np.mean(ll_vals))

    # ── Write fold_metrics.json ──
    fold_output = {"folds": [asdict(f) for f in fold_results], "aggregate": aggregate}
    (run_dir / "fold_metrics.json").write_text(
        json.dumps(fold_output, indent=2), encoding="utf-8",
    )
    log.info("Wrote fold_metrics.json  (%d folds)", len(folds))

    # ── Per-fold PnL bar chart ──
    plot_fold_pnl([asdict(f) for f in fold_results], run_dir / "plots" / "fold_pnl.png")

    # ── Build metrics.json (compatible with single-run format) ──
    avg_win = (
        float(all_trades_df.loc[all_trades_df["pnl"] > 0, "pnl"].mean())
        if not all_trades_df.empty and (all_trades_df["pnl"] > 0).any()
        else 0.0
    )
    avg_loss = (
        float(all_trades_df.loc[all_trades_df["pnl"] < 0, "pnl"].mean())
        if not all_trades_df.empty and (all_trades_df["pnl"] < 0).any()
        else 0.0
    )

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "mode": "walk_forward",
        "equity_note": (
            "Capital carryover: ending equity from fold N becomes starting "
            "capital for fold N+1, producing a continuous compounded equity curve."
        ),
        "symbol": cfg["symbol"],
        "timeframe": cfg["timeframe"],
        "n_bars": len(clean_df),
        "n_bars_tested": sum(f.test_bars for f in fold_results),
        "start_ts": clean_df["time"].iloc[0].isoformat(),
        "end_ts": clean_df["time"].iloc[-1].isoformat(),
        "starting_capital": starting_capital,
        "ending_equity": fold_capital,
        "n_folds": len(folds),
        "n_trades": total_trades,
        "total_pnl": total_net_pnl,
        "win_rate": overall_win_rate,
        "average_win": avg_win,
        "average_loss": avg_loss,
        "total_gross_pnl": total_gross_pnl,
        "total_net_pnl": total_net_pnl,
        "total_cost_pnl": total_cost_pnl,
        "avg_cost_per_trade": total_cost_pnl / max(1, total_trades),
        "cost_as_pct_of_gross_abs": (
            total_cost_pnl / max(1e-12, abs(total_gross_pnl)) * 100
        ),
        "worst_drawdown_pct": worst_dd,
        "max_drawdown_pct": worst_dd,
        "git_commit": git["git_commit"],
        "git_dirty": git["git_dirty"],
    }

    # ── ML note ──
    ml_cfg = cfg.get("model")
    label_cfg = cfg.get("label")
    if ml_cfg is not None and label_cfg is not None:
        metrics["ml_note"] = (
            f"label={label_cfg.get('type')}, model={ml_cfg.get('type')}"
        )

    # ── Write standard artifacts via combined engine ──
    artifact_strategy, _ = build_strategy(cfg["strategy"])
    artifact_rm = RiskManager(risk_config, starting_capital)
    combined_engine = TradingEngine(
        artifact_strategy, artifact_rm, starting_capital, position_size, cost_config,
    )
    combined_engine.equity_records = all_equity_records
    combined_engine.all_fills = all_fills

    _write_artifacts(
        run_dir, cfg_path, combined_engine, clean_df, csv_files,
        cfg, metrics, artifact_strategy, config_path_str,
    )

    # Append walk-forward specific notes to README
    wf_note = (
        f"\nWalk-forward: {len(folds)} folds ({splitter.mode}, "
        f"train={splitter.train_size}, test={splitter.test_size}, "
        f"step={splitter.step})\n"
        "Equity note: capital carries over across folds (continuous compounded curve).\n"
        "See fold_metrics.json for per-fold breakdown.\n"
    )
    readme_path = run_dir / "README.md"
    with open(readme_path, "a", encoding="utf-8") as f:
        f.write(wf_note)

    log.info("Walk-forward run complete: %s (%d folds)", run_dir, len(folds))
    return run_id


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
    cfg_path, cfg, strategy, feature_specs, risk_config, cost_config = _load_config(config_path)

    # ── 2. RUN SETUP ──
    run_id, run_dir = _create_run_dir(cfg)
    git = get_git_info(REPO_ROOT)

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

    # ── WALK-FORWARD CHECK ──
    # Features are computed once on the full dataset above.  This is safe
    # because FeaturePipeline uses only causal (backward-looking) operations
    # — see test_feature_pipeline_causality for the regression guard.
    validation_cfg = cfg.get("validation")
    if validation_cfg is not None and validation_cfg.get("type") == "walk_forward":
        return _run_walk_forward(
            cfg=cfg,
            cfg_path=cfg_path,
            config_path_str=config_path,
            clean_df=clean_df,
            features=X,
            csv_files=csv_files,
            risk_config=risk_config,
            cost_config=cost_config,
            git=git,
            run_id=run_id,
            run_dir=run_dir,
        )

    # ── 6. REPLAY (model → signal → decision → execution, per bar) ──
    starting_capital: float = cfg["starting_capital"]
    position_size: float = cfg.get("position_size", 1.0)
    latency_bars: int = cfg.get("execution", {}).get("latency_bars", 0)
    risk_manager = RiskManager(risk_config, starting_capital)

    engine = TradingEngine(strategy, risk_manager, starting_capital, position_size, cost_config, latency_bars=latency_bars)
    _run_replay(engine, clean_df, X)

    # ── 7. METRICS ──
    metrics = _compute_metrics(engine, bars, clean_df, cfg, git, run_id)

    # ── 8. ARTIFACTS ──
    _write_artifacts(run_dir, cfg_path, engine, clean_df, csv_files, cfg, metrics, strategy, config_path)

    log.info("✓ Run complete: %s", run_dir)
    return run_id
