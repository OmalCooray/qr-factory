"""Tests for the src.eval evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


# ── Prediction Metrics ──────────────────────────────────────────────────────


class TestPredictionMetricsBinaryClassification:
    """Tests for compute_prediction_metrics()."""

    def test_perfect_predictions(self):
        from src.eval.prediction_metrics import compute_prediction_metrics

        df = pd.DataFrame({
            "y_true": [1, 0, 1, 0, 1, 0],
            "y_prob": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        })
        m = compute_prediction_metrics(df)
        assert m["brier"] == pytest.approx(0.0)
        assert m["roc_auc"] == pytest.approx(1.0)
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["n_predictions"] == 6

    def test_single_class_roc_auc_is_none(self):
        from src.eval.prediction_metrics import compute_prediction_metrics

        df = pd.DataFrame({
            "y_true": [1, 1, 1, 1],
            "y_prob": [0.9, 0.8, 0.7, 0.6],
        })
        m = compute_prediction_metrics(df)
        assert m["roc_auc"] is None
        assert m["accuracy"] == pytest.approx(1.0)
        assert m["n_predictions"] == 4

    def test_empty_returns_empty_dict(self):
        from src.eval.prediction_metrics import compute_prediction_metrics

        assert compute_prediction_metrics(None) == {}
        assert compute_prediction_metrics(pd.DataFrame()) == {}

    def test_random_predictions(self):
        from src.eval.prediction_metrics import compute_prediction_metrics

        rng = np.random.RandomState(42)
        n = 1000
        df = pd.DataFrame({
            "y_true": rng.randint(0, 2, size=n),
            "y_prob": rng.uniform(0, 1, size=n),
        })
        m = compute_prediction_metrics(df)
        # Random predictions: roc_auc ~ 0.5, brier ~ 0.25
        assert 0.4 < m["roc_auc"] < 0.6
        assert 0.2 < m["brier"] < 0.35

    def test_nan_rows_dropped(self):
        from src.eval.prediction_metrics import compute_prediction_metrics

        df = pd.DataFrame({
            "y_true": [1, 0, np.nan, 1],
            "y_prob": [0.9, 0.1, 0.5, np.nan],
        })
        m = compute_prediction_metrics(df)
        assert m["n_predictions"] == 2


# ── Trading Metrics ─────────────────────────────────────────────────────────


class TestTradingMaxDrawdown:
    """Tests for compute_trading_metrics() drawdown computation."""

    def test_known_equity_drawdown(self):
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=4, freq="h"),
            "equity": [100.0, 110.0, 90.0, 95.0],
        })
        trades = pd.DataFrame()
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)

        # Peak=110, trough=90 → dd_abs=20, dd_pct=20/110*100
        assert m["max_drawdown"] == pytest.approx(20.0)
        assert m["max_dd_pct"] == pytest.approx(20.0 / 110.0 * 100)

    def test_monotonic_increasing_zero_drawdown(self):
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({
            "equity": [100.0, 105.0, 110.0, 115.0, 120.0],
        })
        trades = pd.DataFrame()
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)

        assert m["max_drawdown"] == pytest.approx(0.0)
        assert m["max_dd_pct"] == pytest.approx(0.0)

    def test_zero_trades_all_zeroed(self):
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({"equity": [100.0, 100.0]})
        trades = pd.DataFrame()
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)

        assert m["num_trades"] == 0
        assert m["total_net_pnl"] == 0.0
        assert m["win_rate"] == 0.0
        assert m["profit_factor"] == 0.0

    def test_profit_factor_with_trades(self):
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({"equity": [100.0, 120.0]})
        trades = pd.DataFrame({
            "pnl": [30.0, -10.0],
            "gross_pnl": [32.0, -8.0],
            "cost_pnl": [2.0, 2.0],
        })
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)
        assert m["num_trades"] == 2
        assert m["profit_factor"] == pytest.approx(3.0)  # 30/10
        assert m["win_rate"] == pytest.approx(0.5)

    def test_gross_equity_metrics(self):
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({"equity": [100.0, 90.0, 95.0]})
        equity_gross = pd.DataFrame({"equity": [100.0, 95.0, 100.0]})
        trades = pd.DataFrame()
        m = compute_trading_metrics(equity, trades, 100.0, equity_gross=equity_gross)

        assert "max_drawdown_gross" in m
        assert "sharpe_per_bar_gross" in m


# ── Comparison & Ranking Disagreement ────────────────────────────────────────


def _make_synthetic_run(
    tmp_path: Path,
    run_name: str,
    net_pnl: float,
    win_rate: float,
    roc_auc: float | None = None,
    brier: float | None = None,
    config_overrides: dict | None = None,
) -> Path:
    """Create a minimal synthetic run directory for comparison tests."""
    run_dir = tmp_path / run_name
    run_dir.mkdir()
    (run_dir / "plots").mkdir()

    starting_capital = 10000.0
    ending_equity = starting_capital + net_pnl

    config = {
        "strategy": {"type": "test_strat"},
        "starting_capital": starting_capital,
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "snapshot_dir": "data/20260217_030324",
        "execution": {
            "spread_points": 40,
            "slippage_points": 10,
            "point_value": 0.01,
        },
        "validation": {
            "type": "walk_forward",
            "mode": "expanding",
            "train_size": 100000,
            "test_size": 50000,
            "step": 50000,
            "drop_last": True,
        },
    }
    if config_overrides:
        config.update(config_overrides)
    (run_dir / "config.yaml").write_text(yaml.dump(config), encoding="utf-8")

    metrics = {
        "run_id": run_name,
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "n_bars": 1000,
        "start_ts": "2024-01-01T00:00:00",
        "end_ts": "2024-06-01T00:00:00",
        "starting_capital": starting_capital,
        "ending_equity": ending_equity,
        "n_trades": 50,
        "total_pnl": net_pnl,
        "win_rate": win_rate,
        "total_net_pnl": net_pnl,
        "total_gross_pnl": net_pnl + 100,
        "total_cost_pnl": 100,
        "git_commit": "abc123",
        "git_dirty": False,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    # Equity curve
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
        "equity": np.linspace(starting_capital, ending_equity, 100),
    })
    equity.to_csv(run_dir / "equity.csv", index=False)

    # Trades
    trades = pd.DataFrame({
        "entry_ts": ["2024-01-01"] * 2,
        "exit_ts": ["2024-01-02"] * 2,
        "side": ["long", "short"],
        "qty": [1.0, 1.0],
        "entry_price": [2000.0, 2050.0],
        "exit_price": [2050.0, 2020.0],
        "pnl": [net_pnl / 2, net_pnl / 2],
        "gross_pnl": [net_pnl / 2 + 50, net_pnl / 2 + 50],
        "net_pnl": [net_pnl / 2, net_pnl / 2],
        "cost_pnl": [50.0, 50.0],
    })
    trades.to_csv(run_dir / "trades.csv", index=False)

    # Predictions (optional)
    if roc_auc is not None:
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, 2, size=n)
        # Scale y_prob to approximate target roc_auc
        noise = rng.uniform(0, 1, size=n)
        if roc_auc > 0.7:
            y_prob = y_true * 0.7 + (1 - y_true) * 0.3 + noise * 0.2
        else:
            y_prob = noise
        y_prob = np.clip(y_prob, 0.01, 0.99)

        preds = pd.DataFrame({
            "fold_id": 0,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
            "y_true": y_true,
            "y_prob": y_prob,
        })
        preds.to_csv(run_dir / "predictions.csv", index=False)

    return run_dir


class TestCompareRankingDisagreement:
    """Tests for compare_runs() ranking and disagreement detection."""

    def test_disagreement_detected(self, tmp_path):
        from src.eval.comparison import compare_runs

        # Run A: good predictions (high roc_auc), bad trading (negative PnL)
        run_a = _make_synthetic_run(
            tmp_path, "run_a", net_pnl=-500.0, win_rate=0.4, roc_auc=0.85,
        )
        # Run B: mediocre predictions, good trading (positive PnL)
        run_b = _make_synthetic_run(
            tmp_path, "run_b", net_pnl=2000.0, win_rate=0.6, roc_auc=0.51,
        )

        out_dir = tmp_path / "compare_out"
        compare_runs([run_a, run_b], out_dir)

        ranking = pd.read_csv(out_dir / "ranking.csv")
        assert (ranking["disagreement_roc_vs_sharpe"] > 0).any()

        # Verify all files exist
        assert (out_dir / "comparison.csv").exists()
        assert (out_dir / "ranking.csv").exists()
        assert (out_dir / "comparison.md").exists()

    def test_no_predictions_no_crash(self, tmp_path):
        from src.eval.comparison import compare_runs

        # Both runs without predictions
        run_a = _make_synthetic_run(tmp_path, "run_c", net_pnl=100.0, win_rate=0.5)
        run_b = _make_synthetic_run(tmp_path, "run_d", net_pnl=200.0, win_rate=0.55)

        out_dir = tmp_path / "compare_no_pred"
        compare_runs([run_a, run_b], out_dir)

        assert (out_dir / "comparison.csv").exists()
        assert (out_dir / "ranking.csv").exists()
        assert (out_dir / "comparison.md").exists()

        # No prediction columns → disagreement should be 0
        ranking = pd.read_csv(out_dir / "ranking.csv")
        assert (ranking["disagreement_roc_vs_sharpe"] == 0).all()

    def test_single_run_evaluate(self, tmp_path):
        """Test evaluate_run produces evaluation.json."""
        from src.eval.aggregator import evaluate_run

        run_dir = _make_synthetic_run(
            tmp_path, "run_single", net_pnl=500.0, win_rate=0.55, roc_auc=0.65,
        )
        result = evaluate_run(run_dir)
        assert (run_dir / "evaluation.json").exists()
        assert result.run_id == "run_single"
        assert result.trading_metrics["num_trades"] == 2
        assert result.prediction_metrics["n_predictions"] == 200

    def test_report_generation(self, tmp_path):
        """Test generate_report produces report.md and drawdown.png."""
        from dataclasses import asdict

        from src.eval.aggregator import evaluate_run
        from src.eval.artifacts import load_run_artifacts
        from src.eval.report import generate_report

        run_dir = _make_synthetic_run(
            tmp_path, "run_report", net_pnl=300.0, win_rate=0.52,
        )
        result = evaluate_run(run_dir)
        artifacts = load_run_artifacts(run_dir)

        report_path = generate_report(
            run_dir, asdict(result), artifacts.equity, artifacts.equity_gross,
        )
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "## Run Summary" in content
        assert "## Trading Metrics" in content
        assert "## Prediction Metrics" in content
        assert "## Metric Disagreement Notes" in content
        assert (run_dir / "plots" / "drawdown.png").exists()


# ── Validation Tests (F7) ──────────────────────────────────────────────────


class TestEvaluateRunIdempotent:
    """evaluate_run() called twice on the same input produces identical output."""

    def test_evaluate_run_idempotent(self, tmp_path):
        from src.eval.aggregator import evaluate_run

        run_dir = _make_synthetic_run(
            tmp_path, "run_idem", net_pnl=500.0, win_rate=0.55, roc_auc=0.65,
        )
        evaluate_run(run_dir)
        first = (run_dir / "evaluation.json").read_bytes()

        evaluate_run(run_dir)
        second = (run_dir / "evaluation.json").read_bytes()

        assert first == second


class TestEdgeCaseTrades:
    """All-winners, all-losers, and zero-equity edge cases."""

    def test_all_winners_profit_factor_inf(self):
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({"equity": [100.0, 120.0, 140.0]})
        trades = pd.DataFrame({
            "pnl": [20.0, 20.0],
            "gross_pnl": [22.0, 22.0],
            "cost_pnl": [2.0, 2.0],
        })
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)
        assert m["win_rate"] == pytest.approx(1.0)
        assert m["profit_factor"] == float("inf")

    def test_all_losers_profit_factor_zero(self):
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({"equity": [100.0, 80.0, 60.0]})
        trades = pd.DataFrame({
            "pnl": [-20.0, -20.0],
            "gross_pnl": [-18.0, -18.0],
            "cost_pnl": [2.0, 2.0],
        })
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)
        assert m["win_rate"] == pytest.approx(0.0)
        assert m["profit_factor"] == pytest.approx(0.0)

    def test_zero_equity_no_crash(self):
        """Equity passes through zero — no inf/nan in Sharpe."""
        from src.eval.trading_metrics import compute_trading_metrics

        equity = pd.DataFrame({
            "equity": [100.0, 50.0, 0.0, 10.0, 30.0],
        })
        trades = pd.DataFrame()
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)

        assert np.isfinite(m["sharpe_per_bar"])
        assert np.isfinite(m["max_dd_pct"])


class TestCostAccountingInvariant:
    """Per-row and aggregate: gross - cost = net."""

    def test_cost_accounting_invariant(self):
        from src.eval.trading_metrics import compute_trading_metrics

        trades = pd.DataFrame({
            "pnl": [15.0, -5.0, 10.0],
            "gross_pnl": [20.0, 0.0, 15.0],
            "cost_pnl": [5.0, 5.0, 5.0],
        })
        equity = pd.DataFrame({"equity": [100.0, 115.0, 110.0, 120.0]})

        m = compute_trading_metrics(equity, trades, starting_capital=100.0)

        # Row-level: gross_pnl - cost_pnl == pnl (net)
        assert (trades["gross_pnl"] - trades["cost_pnl"]).tolist() == trades["pnl"].tolist()

        # Aggregate: total_gross - total_cost == total_net
        assert m["total_gross_pnl"] - m["total_cost_pnl"] == pytest.approx(m["total_net_pnl"])


class TestWalkForwardDrawdownCrossFold:
    """Cross-fold drawdown is captured by the full compounded equity curve."""

    def test_walkforward_drawdown_cross_fold(self):
        from src.eval.trading_metrics import compute_trading_metrics

        # Simulate two folds: fold 1 rises then dips, fold 2 starts lower
        # Cross-fold DD spans the boundary
        fold1 = [100.0, 110.0, 120.0, 115.0]  # per-fold DD: 120→115 = 4.17%
        fold2 = [108.0, 100.0, 105.0]          # per-fold DD: 108→100 = 7.41%
        combined = fold1 + fold2                # combined DD: 120→100 = 16.67%

        equity = pd.DataFrame({"equity": combined})
        trades = pd.DataFrame()
        m = compute_trading_metrics(equity, trades, starting_capital=100.0)

        # Cross-fold DD is larger than either per-fold DD
        max_dd_pct = m["max_dd_pct"]
        fold1_dd = (120 - 115) / 120 * 100
        fold2_dd = (108 - 100) / 108 * 100
        assert max_dd_pct > fold1_dd
        assert max_dd_pct > fold2_dd
        assert max_dd_pct == pytest.approx((120 - 100) / 120 * 100)


class TestCLISmoke:
    """CLI smoke tests for evaluate-run and compare-runs."""

    def test_cli_evaluate_run_smoke(self, tmp_path):
        """CLI evaluate-run exits 0 and produces evaluation.json + report.md."""
        import subprocess
        import sys

        run_dir = _make_synthetic_run(
            tmp_path, "run_cli_eval", net_pnl=200.0, win_rate=0.5,
        )
        result = subprocess.run(
            [sys.executable, "-m", "src.eval.cli", "evaluate-run",
             "--run_dir", str(run_dir)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (run_dir / "evaluation.json").exists()
        assert (run_dir / "report.md").exists()

    def test_cli_compare_runs_smoke(self, tmp_path):
        """CLI compare-runs exits 0 and produces comparison artifacts."""
        import subprocess
        import sys

        run_a = _make_synthetic_run(tmp_path, "run_cmp_a", net_pnl=100.0, win_rate=0.5)
        run_b = _make_synthetic_run(tmp_path, "run_cmp_b", net_pnl=300.0, win_rate=0.6)
        out_dir = tmp_path / "compare_cli"

        result = subprocess.run(
            [sys.executable, "-m", "src.eval.cli", "compare-runs",
             "--run_dirs", str(run_a), str(run_b),
             "--out_dir", str(out_dir)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert (out_dir / "comparison.csv").exists()
        assert (out_dir / "ranking.csv").exists()
        assert (out_dir / "comparison.md").exists()


class TestWalkForwardReportCaveats:
    """Walk-forward runs include DD and prediction caveats in report."""

    def test_walk_forward_report_has_dd_caveat(self, tmp_path):
        from dataclasses import asdict

        from src.eval.aggregator import evaluate_run
        from src.eval.artifacts import load_run_artifacts
        from src.eval.report import generate_report

        run_dir = _make_synthetic_run(
            tmp_path, "run_wf", net_pnl=400.0, win_rate=0.55, roc_auc=0.7,
        )
        # Patch metrics.json to mark as walk-forward
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["mode"] = "walk_forward"
        metrics["n_folds"] = 3
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

        result = evaluate_run(run_dir)
        artifacts = load_run_artifacts(run_dir)
        report_path = generate_report(
            run_dir, asdict(result), artifacts.equity, artifacts.equity_gross,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "continuous compounded equity curve" in content
        assert "Prediction metrics are pooled" in content


# ── Config Match Tests ─────────────────────────────────────────────────────


class TestConfigMatch:
    """Tests for _check_config_match()."""

    def test_matched_configs_return_no_warnings(self, tmp_path):
        from src.eval.comparison import _check_config_match

        run_a = _make_synthetic_run(tmp_path, "match_a", net_pnl=100.0, win_rate=0.5)
        run_b = _make_synthetic_run(tmp_path, "match_b", net_pnl=200.0, win_rate=0.5)

        matched, warnings = _check_config_match([run_a, run_b])
        assert matched is True
        assert warnings == []

    def test_mismatched_snapshot_detected(self, tmp_path):
        from src.eval.comparison import _check_config_match

        run_a = _make_synthetic_run(tmp_path, "snap_a", net_pnl=100.0, win_rate=0.5)
        run_b = _make_synthetic_run(
            tmp_path, "snap_b", net_pnl=200.0, win_rate=0.5,
            config_overrides={"snapshot_dir": "data/different_snapshot"},
        )

        matched, warnings = _check_config_match([run_a, run_b])
        assert matched is False
        assert any("snapshot_dir" in w for w in warnings)

    def test_mismatched_execution_detected(self, tmp_path):
        from src.eval.comparison import _check_config_match

        run_a = _make_synthetic_run(tmp_path, "exec_a", net_pnl=100.0, win_rate=0.5)
        run_b = _make_synthetic_run(
            tmp_path, "exec_b", net_pnl=200.0, win_rate=0.5,
            config_overrides={"execution": {"spread_points": 80, "slippage_points": 10, "point_value": 0.01}},
        )

        matched, warnings = _check_config_match([run_a, run_b])
        assert matched is False
        assert any("spread_points" in w for w in warnings)

    def test_mismatched_label_detected(self, tmp_path):
        from src.eval.comparison import _check_config_match

        label_a = {"type": "return_sign_cost_aware", "horizon": 12, "price_col": "close", "direction": "up"}
        label_b = {"type": "return_sign_cost_aware", "horizon": 6, "price_col": "close", "direction": "up"}
        run_a = _make_synthetic_run(
            tmp_path, "lbl_a", net_pnl=100.0, win_rate=0.5,
            config_overrides={"label": label_a},
        )
        run_b = _make_synthetic_run(
            tmp_path, "lbl_b", net_pnl=200.0, win_rate=0.5,
            config_overrides={"label": label_b},
        )

        matched, warnings = _check_config_match([run_a, run_b])
        assert matched is False
        assert any("label.horizon" in w for w in warnings)

    def test_label_skipped_when_one_run_has_no_label(self, tmp_path):
        """Rule vs ML: rule has no label block — no label mismatch warnings."""
        from src.eval.comparison import _check_config_match

        run_a = _make_synthetic_run(tmp_path, "nolbl_a", net_pnl=100.0, win_rate=0.5)
        run_b = _make_synthetic_run(
            tmp_path, "nolbl_b", net_pnl=200.0, win_rate=0.5,
            config_overrides={"label": {"type": "return_sign_cost_aware", "horizon": 12}},
        )

        matched, warnings = _check_config_match([run_a, run_b])
        assert matched is True
        assert not any("label" in w for w in warnings)


# ── Auto-Conclusion Tests ──────────────────────────────────────────────────


class TestAutoConclusion:
    """Tests for auto-generated conclusions in comparison.md."""

    def test_conclusion_present_in_comparison_md(self, tmp_path):
        from src.eval.comparison import compare_runs

        run_a = _make_synthetic_run(
            tmp_path, "concl_a", net_pnl=-500.0, win_rate=0.4, roc_auc=0.85,
        )
        run_b = _make_synthetic_run(
            tmp_path, "concl_b", net_pnl=2000.0, win_rate=0.6, roc_auc=0.51,
        )

        out_dir = tmp_path / "concl_out"
        compare_runs([run_a, run_b], out_dir)

        md = (out_dir / "comparison.md").read_text(encoding="utf-8")
        assert "## Conclusion" in md
        assert "1. **What changed:**" in md

    def test_conclusion_handles_no_predictions(self, tmp_path):
        from src.eval.comparison import compare_runs

        run_a = _make_synthetic_run(tmp_path, "nopred_a", net_pnl=100.0, win_rate=0.5)
        run_b = _make_synthetic_run(tmp_path, "nopred_b", net_pnl=200.0, win_rate=0.55)

        out_dir = tmp_path / "nopred_out"
        compare_runs([run_a, run_b], out_dir)

        md = (out_dir / "comparison.md").read_text(encoding="utf-8")
        assert "## Conclusion" in md
        assert "1. **What changed:**" in md


# ── Policy Observability Tests ─────────────────────────────────────────────


class TestPolicyObservability:
    """Tests for strategy/label/model config section in report.md."""

    def test_report_includes_strategy_config(self, tmp_path):
        from dataclasses import asdict

        from src.eval.aggregator import evaluate_run
        from src.eval.artifacts import load_run_artifacts
        from src.eval.report import generate_report

        run_dir = _make_synthetic_run(
            tmp_path, "pol_ml", net_pnl=300.0, win_rate=0.52,
            config_overrides={
                "strategy": {"type": "ml_prob_threshold", "p_enter": 0.6, "p_exit": 0.45},
                "label": {"type": "return_sign_cost_aware", "horizon": 12},
                "model": {"type": "logistic", "C": 1.0},
            },
        )
        result = evaluate_run(run_dir)
        artifacts = load_run_artifacts(run_dir)
        report_path = generate_report(
            run_dir, asdict(result), artifacts.equity, artifacts.equity_gross,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "## Strategy & Label Config" in content
        assert "ml_prob_threshold" in content
        assert "p_enter" in content
        assert "logistic" in content

    def test_report_handles_rule_based(self, tmp_path):
        from dataclasses import asdict

        from src.eval.aggregator import evaluate_run
        from src.eval.artifacts import load_run_artifacts
        from src.eval.report import generate_report

        run_dir = _make_synthetic_run(
            tmp_path, "pol_rule", net_pnl=200.0, win_rate=0.5,
        )
        result = evaluate_run(run_dir)
        artifacts = load_run_artifacts(run_dir)
        report_path = generate_report(
            run_dir, asdict(result), artifacts.equity, artifacts.equity_gross,
        )
        content = report_path.read_text(encoding="utf-8")
        assert "## Strategy & Label Config" in content
        # Rule-based: no label or model sections
        assert "Label type" not in content
        assert "Model type" not in content
