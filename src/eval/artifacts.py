"""Run artifacts loader — reads backtest output into a frozen dataclass."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunArtifacts:
    """Immutable container for all artifacts produced by a single backtest run."""

    run_dir: Path
    config: dict
    metrics: dict
    fold_metrics: dict | None
    equity: pd.DataFrame
    equity_gross: pd.DataFrame | None
    trades: pd.DataFrame
    predictions: pd.DataFrame | None

    @property
    def run_id(self) -> str:
        return self.metrics.get("run_id", self.run_dir.name)

    @property
    def strategy_type(self) -> str:
        return self.config.get("strategy", {}).get("type", "unknown")

    @property
    def is_walk_forward(self) -> bool:
        return self.metrics.get("mode") == "walk_forward"

    @property
    def starting_capital(self) -> float:
        return float(self.metrics.get("starting_capital", self.config.get("starting_capital", 0)))


def load_run_artifacts(run_dir: str | Path) -> RunArtifacts:
    """Load all artifacts from a backtest run directory.

    Required files: metrics.json, config.yaml, equity.csv.
    Optional files: fold_metrics.json, equity_gross.csv, predictions.csv.

    Parameters
    ----------
    run_dir : str | Path
        Path to the run directory.

    Returns
    -------
    RunArtifacts
    """
    run_dir = Path(run_dir)

    # Required
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    equity = pd.read_csv(run_dir / "equity.csv")

    # Optional: fold_metrics
    fold_path = run_dir / "fold_metrics.json"
    fold_metrics = (
        json.loads(fold_path.read_text(encoding="utf-8"))
        if fold_path.exists()
        else None
    )

    # Optional: equity_gross
    gross_path = run_dir / "equity_gross.csv"
    equity_gross = pd.read_csv(gross_path) if gross_path.exists() else None

    # trades.csv — handle zero-byte from touch()
    trades_path = run_dir / "trades.csv"
    if trades_path.exists() and trades_path.stat().st_size > 0:
        trades = pd.read_csv(trades_path)
    else:
        trades = pd.DataFrame()

    # Optional: predictions
    pred_path = run_dir / "predictions.csv"
    predictions = pd.read_csv(pred_path) if pred_path.exists() else None

    return RunArtifacts(
        run_dir=run_dir,
        config=config,
        metrics=metrics,
        fold_metrics=fold_metrics,
        equity=equity,
        equity_gross=equity_gross,
        trades=trades,
        predictions=predictions,
    )
