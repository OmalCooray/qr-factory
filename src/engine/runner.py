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

from src.replay.bar_iterator import BarIterator
from src.reporting.plots import plot_close_price

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

    # ── Generate run_id ──────────────────────────────────────────────
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "data_snapshot").mkdir(exist_ok=True)

    log.info("Run ID   : %s", run_id)
    log.info("Output   : %s", run_dir)

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

    # ── Build bar iterator (validates + dedupes + sorts) ─────────────
    bars = BarIterator(df)
    clean_df = bars.df

    # ── Flat equity replay ───────────────────────────────────────────
    equity_records = []
    for bar in bars:
        equity_records.append({
            "timestamp": bar["time"].isoformat(),
            "equity": starting_capital,
        })

    equity_df = pd.DataFrame(equity_records)

    # ── Write artifacts ──────────────────────────────────────────────
    # 1. config.yaml
    shutil.copy2(cfg_path, run_dir / "config.yaml")

    # 2. equity.csv
    equity_df.to_csv(run_dir / "equity.csv", index=False)
    log.info("Wrote equity.csv  (%s rows)", f"{len(equity_df):,}")

    # 3. metrics.json
    start_ts = clean_df["time"].iloc[0].isoformat()
    end_ts = clean_df["time"].iloc[-1].isoformat()
    ending_equity = starting_capital  # flat — no trades
    pnl_abs = ending_equity - starting_capital
    pnl_pct = (pnl_abs / starting_capital) * 100 if starting_capital else 0.0

    metrics = {
        "run_id": run_id,
        "symbol": symbol,
        "timeframe": timeframe,
        "n_bars": len(bars),
        "start_ts": start_ts,
        "end_ts": end_ts,
        "starting_capital": starting_capital,
        "ending_equity": ending_equity,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8",
    )
    log.info("Wrote metrics.json")

    # 4. plots/close_price.png
    plot_close_price(clean_df, run_dir / "plots" / "close_price.png")

    # 5. data_snapshot/DATA_REF.json
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

    # 6. README.md (exactly 5 lines)
    readme_lines = [
        "Added deterministic replay runner (Week 1 milestone)",
        f"Dataset: {snapshot_rel}",
        f"Reproduce: uv run python -m src backtest --config {config_path}",
        f"Run ID: {run_id}",
        f"Flat equity (no trades); {len(bars):,} bars replayed deterministically.",
    ]
    (run_dir / "README.md").write_text(
        "\n".join(readme_lines) + "\n", encoding="utf-8",
    )
    log.info("Wrote README.md")

    log.info("✓ Run complete: %s", run_dir)
    return run_id
