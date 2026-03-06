"""Evaluation orchestrator — combines prediction + trading metrics per run."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from .artifacts import RunArtifacts, load_run_artifacts
from .prediction_metrics import compute_prediction_metrics
from .trading_metrics import compute_trading_metrics

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """Combined evaluation output for a single run."""

    run_id: str
    strategy_type: str
    prediction_metrics: dict
    trading_metrics: dict
    metadata: dict


def evaluate_run(run_dir: str | Path) -> EvaluationResult:
    """Load artifacts, compute all metrics, and write evaluation.json.

    Parameters
    ----------
    run_dir : str | Path
        Path to the backtest run directory.

    Returns
    -------
    EvaluationResult
    """
    run_dir = Path(run_dir)
    artifacts = load_run_artifacts(run_dir)

    pred_metrics = compute_prediction_metrics(artifacts.predictions)
    trade_metrics = compute_trading_metrics(
        equity=artifacts.equity,
        trades=artifacts.trades,
        starting_capital=artifacts.starting_capital,
        equity_gross=artifacts.equity_gross,
    )

    metadata = {
        "symbol": artifacts.metrics.get("symbol"),
        "timeframe": artifacts.metrics.get("timeframe"),
        "n_bars": artifacts.metrics.get("n_bars"),
        "start_ts": artifacts.metrics.get("start_ts"),
        "end_ts": artifacts.metrics.get("end_ts"),
        "starting_capital": artifacts.starting_capital,
        "ending_equity": artifacts.metrics.get("ending_equity"),
        "git_commit": artifacts.metrics.get("git_commit"),
        "git_dirty": artifacts.metrics.get("git_dirty"),
        "is_walk_forward": artifacts.is_walk_forward,
        "n_folds": artifacts.metrics.get("n_folds"),
    }

    result = EvaluationResult(
        run_id=artifacts.run_id,
        strategy_type=artifacts.strategy_type,
        prediction_metrics=pred_metrics,
        trading_metrics=trade_metrics,
        metadata=metadata,
    )

    # Write evaluation.json
    out = asdict(result)
    (run_dir / "evaluation.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8",
    )
    log.info("Wrote evaluation.json → %s", run_dir / "evaluation.json")

    return result
