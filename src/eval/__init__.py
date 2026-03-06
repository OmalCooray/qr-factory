"""Evaluation suite for qr-factory backtest runs."""

from .aggregator import EvaluationResult, evaluate_run
from .artifacts import RunArtifacts, load_run_artifacts
from .comparison import compare_runs
from .prediction_metrics import compute_prediction_metrics
from .trading_metrics import compute_trading_metrics

__all__ = [
    "RunArtifacts",
    "load_run_artifacts",
    "compute_prediction_metrics",
    "compute_trading_metrics",
    "evaluate_run",
    "EvaluationResult",
    "compare_runs",
]
