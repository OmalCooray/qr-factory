"""Tests for walk-forward backtest integration."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.engine.runner import run_backtest
from src.indicators.core.pipeline import FeaturePipeline, FeatureSpec
from src.indicators.impl.ema import EMA
from src.indicators.impl.ma import SMA
from src.indicators.impl.transforms import ZScoreRolling
from src.validation import TimeSeriesSplitter


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_synthetic_bars(n: int = 1000) -> pd.DataFrame:
    """Create n rows of synthetic OHLCV with oscillating close prices.

    Oscillation guarantees MA crossover trades for fast=2, slow=5.
    """
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    rng = np.random.default_rng(42)
    base = 2000.0 + np.sin(np.arange(n) * 0.05) * 50
    noise = rng.normal(0, 0.5, n)
    close = base + noise
    return pd.DataFrame({
        "time": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
    })


def _make_config(
    tmp_path, *, n_bars: int = 1000, validation: dict | None = None,
) -> str:
    """Write synthetic data + YAML config; return config path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _make_synthetic_bars(n_bars).to_csv(data_dir / "test.csv", index=False)

    cfg = {
        "symbol": "TEST",
        "timeframe": "M5",
        "snapshot_dir": str(data_dir),
        "starting_capital": 10000.0,
        "output_dir": str(tmp_path / "runs"),
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    if validation is not None:
        cfg["validation"] = validation

    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)
    return str(config_path)


# ── Tests ────────────────────────────────────────────────────────────────────


class TestWalkForwardFoldCount:
    """Verify splitter produces the expected number of folds."""

    def test_expanding_5_folds(self):
        splitter = TimeSeriesSplitter(
            mode="expanding", train_size=500, test_size=100, step=100,
        )
        folds = list(splitter.split(1000))
        assert len(folds) == 5

    def test_rolling_5_folds(self):
        splitter = TimeSeriesSplitter(
            mode="rolling", train_size=500, test_size=100, step=100,
        )
        folds = list(splitter.split(1000))
        assert len(folds) == 5


class TestWalkForwardNoLeakage:
    """Verify train.max() < test.min() for every fold."""

    @pytest.mark.parametrize("mode", ["expanding", "rolling"])
    def test_no_leakage(self, mode):
        splitter = TimeSeriesSplitter(
            mode=mode, train_size=500, test_size=100, step=100,
        )
        for train_idx, test_idx in splitter.split(1000):
            assert train_idx.max() < test_idx.min(), (
                f"Leakage: train_max={train_idx.max()}, test_min={test_idx.min()}"
            )


class TestWalkForwardRunner:
    """Integration tests: run_backtest with walk-forward config."""

    def test_walk_forward_produces_fold_metrics(self, tmp_path):
        config_path = _make_config(tmp_path, validation={
            "type": "walk_forward",
            "mode": "expanding",
            "train_size": 500,
            "test_size": 100,
            "step": 100,
            "drop_last": True,
        })

        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        # fold_metrics.json exists
        fm_path = run_dir / "fold_metrics.json"
        assert fm_path.exists(), "fold_metrics.json not written"

        fm = json.loads(fm_path.read_text())
        assert len(fm["folds"]) == 5
        assert fm["aggregate"]["n_folds"] == 5

        # Each fold has required keys
        for fold in fm["folds"]:
            assert "fold_id" in fold
            assert "n_trades" in fold
            assert "net_pnl" in fold
            assert "gross_pnl" in fold
            assert "max_drawdown_pct" in fold
            assert "test_bars" in fold
            assert fold["test_bars"] == 100

    def test_walk_forward_standard_artifacts(self, tmp_path):
        config_path = _make_config(tmp_path, validation={
            "type": "walk_forward",
            "mode": "expanding",
            "train_size": 500,
            "test_size": 100,
            "step": 100,
        })

        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        # All standard artifacts still written
        assert (run_dir / "config.yaml").exists()
        assert (run_dir / "equity.csv").exists()
        assert (run_dir / "trades.csv").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "plots" / "equity.png").exists()
        assert (run_dir / "fold_metrics.json").exists()

        # metrics.json contains walk_forward mode
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["mode"] == "walk_forward"
        assert metrics["n_folds"] == 5

    def test_walk_forward_aggregate_consistency(self, tmp_path):
        config_path = _make_config(tmp_path, validation={
            "type": "walk_forward",
            "mode": "expanding",
            "train_size": 500,
            "test_size": 100,
            "step": 100,
        })

        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        fm = json.loads((run_dir / "fold_metrics.json").read_text())
        agg = fm["aggregate"]
        folds = fm["folds"]

        # Aggregate trades = sum of per-fold trades
        assert agg["total_trades"] == sum(f["n_trades"] for f in folds)
        # Aggregate net PnL = sum of per-fold net PnL
        assert abs(agg["total_net_pnl"] - sum(f["net_pnl"] for f in folds)) < 1e-10

    def test_walk_forward_rolling_mode(self, tmp_path):
        config_path = _make_config(tmp_path, validation={
            "type": "walk_forward",
            "mode": "rolling",
            "train_size": 500,
            "test_size": 100,
            "step": 100,
        })

        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        fm = json.loads((run_dir / "fold_metrics.json").read_text())
        assert len(fm["folds"]) == 5


class TestBackwardCompatibility:
    """No validation config = existing single-run behavior."""

    def test_no_validation_config(self, tmp_path):
        config_path = _make_config(tmp_path, validation=None)

        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        # Standard artifacts present
        assert (run_dir / "equity.csv").exists()
        assert (run_dir / "metrics.json").exists()
        assert (run_dir / "trades.csv").exists()

        # No fold_metrics.json
        assert not (run_dir / "fold_metrics.json").exists()

        # metrics.json has no "mode" key (single-run format)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert "mode" not in metrics


class TestFeaturePipelineCausality:
    """Guard that FeaturePipeline is strictly causal (no lookahead).

    Strategy: compute features on the full dataset, then on a prefix.
    If the pipeline is causal, the prefix result must exactly match the
    corresponding rows of the full result.  Any future-leaking operation
    (e.g. a centered rolling window or forward-fill from future bars)
    would cause a mismatch.
    """

    @staticmethod
    def _make_bars(n: int) -> pd.DataFrame:
        rng = np.random.default_rng(99)
        close = 2000.0 + np.cumsum(rng.normal(0, 1, n))
        return pd.DataFrame({
            "open": close - 0.3,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(100, 2000, n).astype(float),
        })

    @pytest.mark.parametrize("pipeline,name", [
        (
            FeaturePipeline([FeatureSpec(base=SMA(period=20))]),
            "SMA-20",
        ),
        (
            FeaturePipeline([FeatureSpec(base=EMA(period=50))]),
            "EMA-50",
        ),
        (
            FeaturePipeline([
                FeatureSpec(base=SMA(period=10), transforms=[ZScoreRolling(window=20)]),
            ]),
            "SMA-10+ZScore-20",
        ),
    ], ids=["SMA-20", "EMA-50", "SMA-10+ZScore-20"])
    def test_feature_pipeline_causality(self, pipeline, name):
        """Features on a prefix must equal the prefix slice of full features."""
        bars = self._make_bars(500)
        prefix_len = 300

        full_features = pipeline.transform(bars)
        prefix_features = pipeline.transform(bars.iloc[:prefix_len].reset_index(drop=True))

        full_prefix = full_features.iloc[:prefix_len].reset_index(drop=True)
        prefix_features = prefix_features.reset_index(drop=True)

        # Both NaN positions and numeric values must match exactly
        pd.testing.assert_frame_equal(
            prefix_features, full_prefix,
            check_exact=True,
            obj=f"prefix features ({name})",
        )


class TestWalkForwardEquitySemantics:
    """Verify equity interpretation metadata is explicit."""

    def test_metrics_json_has_equity_note(self, tmp_path):
        config_path = _make_config(tmp_path, validation={
            "type": "walk_forward",
            "mode": "expanding",
            "train_size": 500,
            "test_size": 100,
            "step": 100,
        })

        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert "equity_note" in metrics
        assert "independent" in metrics["equity_note"].lower()
        assert "NOT" in metrics["equity_note"]

    def test_readme_has_walk_forward_note(self, tmp_path):
        config_path = _make_config(tmp_path, validation={
            "type": "walk_forward",
            "mode": "expanding",
            "train_size": 500,
            "test_size": 100,
            "step": 100,
        })

        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        readme = (run_dir / "README.md").read_text()
        assert "Walk-forward" in readme
        assert "independent" in readme.lower()
        assert "fold_metrics.json" in readme
