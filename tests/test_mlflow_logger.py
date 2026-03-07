"""Tests for the MLflow logger module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml


def _make_run_dir(tmp_path: Path) -> Path:
    """Create a minimal synthetic run directory."""
    run_dir = tmp_path / "run_mlflow_test"
    run_dir.mkdir()
    (run_dir / "plots").mkdir()

    config = {
        "strategy": {"type": "test_strat"},
        "starting_capital": 10000.0,
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "model": {"type": "logistic"},
        "validation": {"type": "walk_forward", "mode": "expanding"},
        "label": {"type": "return_sign_cost_aware"},
    }
    (run_dir / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    metrics = {
        "run_id": "run_mlflow_test",
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "n_bars": 1000,
        "starting_capital": 10000.0,
        "ending_equity": 10500.0,
        "git_commit": "abc123",
        "git_dirty": False,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    equity = pd.DataFrame({"equity": np.linspace(10000, 10500, 50)})
    equity.to_csv(run_dir / "equity.csv", index=False)
    (run_dir / "trades.csv").write_text("", encoding="utf-8")

    return run_dir


def _make_evaluation() -> dict:
    """Create a minimal evaluation dict."""
    return {
        "run_id": "run_mlflow_test",
        "strategy_type": "test_strat",
        "prediction_metrics": {"brier": 0.25, "roc_auc": 0.55, "n_predictions": 200},
        "trading_metrics": {"total_net_pnl": 500.0, "sharpe_per_bar": 0.001, "num_trades": 10},
        "calibration_metrics": {"ece": 0.045, "brier": 0.25},
        "metadata": {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "git_commit": "abc123",
            "git_dirty": False,
        },
    }


class TestMlflowUnavailable:
    """Tests for graceful degradation when mlflow is not installed."""

    def test_mlflow_unavailable_returns_none(self, tmp_path):
        """When mlflow import fails, returns None without crashing."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "mlflow":
                raise ImportError("No module named 'mlflow'")
            return real_import(name, *args, **kwargs)

        run_dir = _make_run_dir(tmp_path)
        evaluation = _make_evaluation()

        from src.eval.mlflow_logger import log_run_to_mlflow

        with patch("builtins.__import__", side_effect=mock_import):
            result = log_run_to_mlflow(run_dir, evaluation)

        assert result is None


class TestMlflowIntegration:
    """Integration tests that require mlflow to be installed."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_mlflow(self):
        pytest.importorskip("mlflow")

    def test_log_run_creates_mlflow_run(self, tmp_path):
        """Verify a run is logged to MLflow with correct params and metrics."""
        import mlflow

        from src.eval.mlflow_logger import log_run_to_mlflow

        run_dir = _make_run_dir(tmp_path)
        evaluation = _make_evaluation()
        calibration = {"ece": 0.045, "brier": 0.25}

        # Point MLflow to tmp dir (SQLite)
        tracking_uri = "sqlite:///" + str(tmp_path / "mlflow.db").replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)

        with patch("src.eval.mlflow_logger.Path.cwd", return_value=tmp_path):
            run_id = log_run_to_mlflow(
                run_dir, evaluation, calibration=calibration,
            )

        assert run_id is not None

        # Verify the run exists in MLflow
        client = mlflow.tracking.MlflowClient(tracking_uri)
        run = client.get_run(run_id)
        assert run.data.params["symbol"] == "XAUUSD"
        assert run.data.params["strategy_type"] == "test_strat"
        assert "pred_brier" in run.data.metrics
        assert "trade_total_net_pnl" in run.data.metrics
        assert "cal_ece" in run.data.metrics

    def test_artifacts_logged(self, tmp_path):
        """Verify expected artifact files are logged."""
        import mlflow

        from src.eval.mlflow_logger import log_run_to_mlflow

        run_dir = _make_run_dir(tmp_path)
        # Add evaluation.json to the run dir
        evaluation = _make_evaluation()
        (run_dir / "evaluation.json").write_text(
            json.dumps(evaluation), encoding="utf-8",
        )

        tracking_uri = "sqlite:///" + str(tmp_path / "mlflow.db").replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)

        with patch("src.eval.mlflow_logger.Path.cwd", return_value=tmp_path):
            run_id = log_run_to_mlflow(run_dir, evaluation)

        assert run_id is not None

        client = mlflow.tracking.MlflowClient(tracking_uri)
        artifacts = [a.path for a in client.list_artifacts(run_id)]
        assert "config.yaml" in artifacts
        assert "metrics.json" in artifacts
        assert "equity.csv" in artifacts
        assert "evaluation.json" in artifacts
