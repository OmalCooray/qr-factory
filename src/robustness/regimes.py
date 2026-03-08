"""Regime slicing — post-run stratification by volatility and session.

Operates on completed run artifacts without retraining or replaying.
Assigns each trade to a regime bucket and computes per-slice metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src._paths import REPO_ROOT

log = logging.getLogger(__name__)

# ── Session definitions (UTC hour ranges) ─────────────────────────────────────

SESSIONS = {
    "asia": (0, 8),       # 00:00–07:59 UTC
    "london": (8, 16),    # 08:00–15:59 UTC
    "new_york": (16, 24), # 16:00–23:59 UTC
}


# ── Volatility regime computation ─────────────────────────────────────────────


def compute_volatility_regimes(
    close: pd.Series,
    window: int = 100,
) -> pd.Series:
    """Assign each bar a volatility regime from rolling realized vol.

    Uses rolling std-dev of log returns.  Bars are split at the median
    into ``"low_vol"`` and ``"high_vol"``.  The first ``window`` bars
    where vol is undefined are labelled ``"warmup"``.

    Parameters
    ----------
    close : pd.Series
        Close prices indexed consistently with the market DataFrame.
    window : int
        Rolling window for realized volatility (default 100 bars).

    Returns
    -------
    pd.Series
        Same index as *close*, values in ``{"warmup", "low_vol", "high_vol"}``.
    """
    log_ret = np.log(close / close.shift(1))
    rolling_vol = log_ret.rolling(window).std()

    median_vol = rolling_vol.median()
    regime = pd.Series("warmup", index=close.index, dtype="object")
    valid = rolling_vol.notna()
    regime[valid & (rolling_vol <= median_vol)] = "low_vol"
    regime[valid & (rolling_vol > median_vol)] = "high_vol"
    return regime


# ── Session assignment ────────────────────────────────────────────────────────


def assign_session(timestamps: pd.Series) -> pd.Series:
    """Assign each timestamp to a trading session based on UTC hour.

    Returns
    -------
    pd.Series
        Values in ``{"asia", "london", "new_york"}``.
    """
    ts = pd.to_datetime(timestamps, utc=True)
    hours = ts.dt.hour

    labels = pd.Series("asia", index=timestamps.index, dtype="object")
    for session_name, (start_h, end_h) in SESSIONS.items():
        if end_h <= 24:
            mask = (hours >= start_h) & (hours < end_h)
        else:
            mask = hours >= start_h
        labels[mask] = session_name
    return labels


# ── Per-slice metric aggregation ──────────────────────────────────────────────


def aggregate_slice_metrics(
    trades: pd.DataFrame,
    labels: pd.Series,
    label_name: str = "slice",
) -> dict[str, dict[str, Any]]:
    """Compute trading metrics per slice/regime bucket.

    Parameters
    ----------
    trades : pd.DataFrame
        Must have columns: ``pnl``, ``gross_pnl``.
    labels : pd.Series
        Same length as *trades*, categorical labels for each trade.
    label_name : str
        Name used in output keys (e.g. "vol_regime", "session").

    Returns
    -------
    dict[str, dict]
        Mapping of label value → metrics dict.
    """
    trades = trades.copy()
    trades[label_name] = labels.values

    result: dict[str, dict[str, Any]] = {}
    for group_val, grp in trades.groupby(label_name, sort=True):
        n = len(grp)
        gross_pnl = float(grp["gross_pnl"].sum())
        net_pnl = float(grp["pnl"].sum())
        cost_pnl = gross_pnl - net_pnl
        win_rate = float((grp["pnl"] > 0).mean()) if n > 0 else 0.0
        avg_trade_pnl = float(grp["pnl"].mean()) if n > 0 else 0.0
        avg_cost = cost_pnl / max(1, n)

        result[str(group_val)] = {
            "n_trades": n,
            "gross_pnl": round(gross_pnl, 6),
            "net_pnl": round(net_pnl, 6),
            "cost_pnl": round(cost_pnl, 6),
            "avg_cost_per_trade": round(avg_cost, 6),
            "win_rate": round(win_rate, 4),
            "avg_trade_pnl": round(avg_trade_pnl, 6),
        }

    return result


# ── Market data loader (minimal) ─────────────────────────────────────────────


def _load_ohlcv(snapshot_dir: str | Path) -> pd.DataFrame:
    """Load OHLCV data from a snapshot directory (minimal loader for analysis)."""
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.is_absolute():
        snapshot_dir = REPO_ROOT / snapshot_dir

    csv_files = sorted(snapshot_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs in {snapshot_dir}")

    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True)
    return df


# ── Full run regime analysis ─────────────────────────────────────────────────


def analyze_run_regimes(
    run_dir: Path,
    vol_window: int = 100,
) -> dict[str, Any]:
    """Compute volatility and session regime metrics for a completed run.

    Loads market data from the snapshot referenced in the run's config,
    computes volatility regimes, assigns each trade to a regime/session,
    and writes ``regime_slice_metrics.json`` and ``session_slice_metrics.json``.

    Parameters
    ----------
    run_dir : Path
        Path to a completed run directory.
    vol_window : int
        Rolling window for volatility regime computation.

    Returns
    -------
    dict
        ``{"volatility": {...}, "session": {...}}`` with per-slice metrics.
    """
    config_path = run_dir / "config.yaml"
    trades_path = run_dir / "trades.csv"

    if not config_path.exists() or not trades_path.exists():
        log.warning("Missing config or trades in %s — skipping regime analysis", run_dir)
        return {}

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    trades = pd.read_csv(trades_path)

    if trades.empty:
        log.info("No trades in %s — skipping regime analysis", run_dir)
        empty_result: dict[str, Any] = {"volatility": {}, "session": {}}
        _write_regime_artifacts(run_dir, empty_result)
        return empty_result

    # ── Load market data ──
    try:
        market_df = _load_ohlcv(config["snapshot_dir"])
    except (FileNotFoundError, KeyError):
        log.warning("Cannot load market data for %s — skipping regime analysis", run_dir)
        return {}

    # ── Volatility regime ──
    vol_regimes = compute_volatility_regimes(market_df["close"], window=vol_window)
    # Build a lookup: timestamp → regime
    market_df["_vol_regime"] = vol_regimes.values
    market_df["_time_key"] = market_df["time"]

    # Match trades to bars via merge_asof on entry timestamp
    trades["_entry_dt"] = pd.to_datetime(trades["entry_ts"], utc=True)
    trades_sorted = trades.sort_values("_entry_dt")
    market_sorted = market_df[["_time_key", "_vol_regime"]].sort_values("_time_key")

    matched = pd.merge_asof(
        trades_sorted,
        market_sorted,
        left_on="_entry_dt",
        right_on="_time_key",
        direction="backward",
    )
    trade_vol_labels = matched["_vol_regime"].fillna("unknown")
    vol_metrics = aggregate_slice_metrics(
        matched, trade_vol_labels, label_name="vol_regime",
    )

    # ── Session slicing ──
    session_labels = assign_session(trades["entry_ts"])
    session_metrics = aggregate_slice_metrics(
        trades, session_labels, label_name="session",
    )

    result: dict[str, Any] = {
        "volatility": vol_metrics,
        "session": session_metrics,
        "vol_window": vol_window,
    }

    _write_regime_artifacts(run_dir, result)
    return result


def _write_regime_artifacts(run_dir: Path, result: dict) -> None:
    """Write regime and session metrics to the run directory."""
    vol_path = run_dir / "regime_slice_metrics.json"
    vol_path.write_text(
        json.dumps(result.get("volatility", {}), indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", vol_path.name)

    session_path = run_dir / "session_slice_metrics.json"
    session_path.write_text(
        json.dumps(result.get("session", {}), indent=2),
        encoding="utf-8",
    )
    log.info("Wrote %s", session_path.name)
