"""Thin MLflow wrapper for logging a completed backtest run."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import yaml

log = logging.getLogger(__name__)


def log_run_to_mlflow(
    run_dir: Path,
    evaluation: dict,
    calibration: dict | None = None,
    experiment_name: str = "qr-factory",
) -> str | None:
    """Log a completed run to MLflow (local SQLite-backed tracking).

    Parameters
    ----------
    run_dir : Path
        Path to the backtest run directory.
    evaluation : dict
        Output of ``asdict(EvaluationResult)``.
    calibration : dict | None
        Output of ``compute_calibration()`` (optional).
    experiment_name : str
        MLflow experiment name.

    Returns
    -------
    str | None
        MLflow run ID, or None if mlflow is unavailable.
    """
    try:
        import mlflow
    except ImportError:
        log.warning("mlflow not installed — skipping MLflow logging. Install with: uv sync --extra tracking")
        return None

    # Use local SQLite-backed tracking in repo root
    tracking_uri = "sqlite:///" + str(Path.cwd() / "mlflow.db").replace("\\", "/")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Load config
    config_path = run_dir / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    # Load metrics.json for git info and other metadata
    metrics_path = run_dir / "metrics.json"
    run_metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    meta = evaluation.get("metadata", {})
    pred_metrics = evaluation.get("prediction_metrics", {})
    trade_metrics = evaluation.get("trading_metrics", {})
    cal_metrics = evaluation.get("calibration_metrics", {})

    with mlflow.start_run() as mlf_run:
        # -- Params --
        cal_cfg = config.get("calibration", {})
        cal_method = cal_cfg.get("method", "none")
        params = {
            "symbol": config.get("symbol", meta.get("symbol", "")),
            "timeframe": config.get("timeframe", meta.get("timeframe", "")),
            "strategy_type": config.get("strategy", {}).get("type", ""),
            "model_type": config.get("model", {}).get("type", ""),
            "validation_mode": config.get("validation", {}).get("mode", ""),
            "label_type": config.get("label", {}).get("type", ""),
            "starting_capital": str(config.get("starting_capital", "")),
            "calibrated": str(cal_method != "none").lower(),
            "calibration_method": cal_method,
            "cal_fraction": str(cal_cfg.get("cal_fraction", "")),
        }
        mlflow.log_params({k: v for k, v in params.items() if v})

        # -- Metrics --
        all_metrics: dict[str, float] = {}
        for k, v in pred_metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                all_metrics[f"pred_{k}"] = float(v)
        for k, v in trade_metrics.items():
            if isinstance(v, (int, float)) and v is not None:
                all_metrics[f"trade_{k}"] = float(v)
        if calibration and "ece" in calibration:
            all_metrics["cal_ece"] = float(calibration["ece"])
        elif cal_metrics and "ece" in cal_metrics:
            all_metrics["cal_ece"] = float(cal_metrics["ece"])

        # Filter out inf/nan
        safe_metrics = {
            k: v for k, v in all_metrics.items()
            if isinstance(v, (int, float)) and v != float("inf") and v != float("-inf") and v == v
        }
        if safe_metrics:
            mlflow.log_metrics(safe_metrics)

        # -- Artifacts --
        artifact_files = [
            "config.yaml",
            "metrics.json",
            "equity.csv",
            "trades.csv",
            "predictions.csv",
            "evaluation.json",
            "calibration.json",
            "report.md",
        ]
        for fname in artifact_files:
            fpath = run_dir / fname
            if fpath.exists():
                mlflow.log_artifact(str(fpath))

        # Log plot files
        plots_dir = run_dir / "plots"
        if plots_dir.exists():
            for plot_file in plots_dir.iterdir():
                if plot_file.is_file():
                    mlflow.log_artifact(str(plot_file), artifact_path="plots")

        # -- Tags --
        tags = {
            "run_id": evaluation.get("run_id", run_dir.name),
            "symbol": params.get("symbol", ""),
            "timeframe": params.get("timeframe", ""),
            "model_type": params.get("model_type", ""),
            "validation_type": config.get("validation", {}).get("type", ""),
        }
        git_commit = run_metrics.get("git_commit", meta.get("git_commit"))
        git_dirty = run_metrics.get("git_dirty", meta.get("git_dirty"))
        if git_commit:
            tags["git_commit"] = str(git_commit)
        if git_dirty is not None:
            tags["git_dirty"] = str(git_dirty)

        mlflow.set_tags({k: v for k, v in tags.items() if v})

        mlf_run_id = mlf_run.info.run_id
        log.info("Logged to MLflow: experiment=%s, run_id=%s", experiment_name, mlf_run_id)
        return mlf_run_id
