"""Batch runner: runs all 6 experiment configs and prints a comparison table."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Ensure repo root is importable
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.engine.runner import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENT_CONFIGS = [
    "configs/exp_sma_10_30.yaml",
    "configs/exp_sma_50_200.yaml",
    "configs/exp_ema_10_30.yaml",
    "configs/exp_ema_50_200.yaml",
    "configs/exp_sma_10_30_adx.yaml",
    "configs/exp_ema_10_30_adx.yaml",
]

METRIC_COLS = [
    "config",
    "run_id",
    "n_trades",
    "total_pnl",
    "win_rate",
    "average_win",
    "average_loss",
    "max_drawdown_pct",
    "ending_equity",
]


def main() -> None:
    runs_dir = _REPO_ROOT / "runs"
    rows: list[dict] = []

    for config_path in EXPERIMENT_CONFIGS:
        config_name = Path(config_path).stem
        log.info("=" * 60)
        log.info("Running experiment: %s", config_name)
        log.info("=" * 60)

        try:
            run_id = run_backtest(config_path)
        except Exception:
            log.exception("Experiment %s FAILED", config_name)
            continue

        # Read metrics from the completed run
        metrics_path = runs_dir / run_id / "metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        rows.append({
            "config": config_name,
            "run_id": run_id,
            "n_trades": metrics["n_trades"],
            "total_pnl": metrics["total_pnl"],
            "win_rate": metrics["win_rate"],
            "average_win": metrics["average_win"],
            "average_loss": metrics["average_loss"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "ending_equity": metrics["ending_equity"],
        })

    if not rows:
        log.error("No experiments completed successfully.")
        sys.exit(1)

    # Build comparison table
    summary = pd.DataFrame(rows, columns=METRIC_COLS)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPARISON")
    print("=" * 80)
    print(summary.to_string(index=False))
    print("=" * 80 + "\n")

    # Save to CSV
    out_path = runs_dir / "experiment_summary.csv"
    summary.to_csv(out_path, index=False)
    log.info("Summary saved to %s", out_path)


if __name__ == "__main__":
    main()
