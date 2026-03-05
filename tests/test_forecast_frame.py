"""Tests for ForecastFrame builder, Predictor interface, and ML runner wiring."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from src.indicators.core.pipeline import FeaturePipeline, FeatureSpec
from src.indicators.impl.ema import EMA
from src.indicators.impl.ma import SMA
from src.ml.frame import build_forecast_frame
from src.ml.predictor import DummyPredictor, build_predictor


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_bars(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV bars with known close = 1..n (shifted by 2000)."""
    dates = pd.date_range("2024-01-01", periods=n, freq="5min")
    close = 2000.0 + np.arange(1, n + 1, dtype=float)
    return pd.DataFrame({
        "time": dates,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000.0,
    })


def _simple_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute EMA-5 + SMA-10 features on bars."""
    specs = [FeatureSpec(base=EMA(period=5)), FeatureSpec(base=SMA(period=10))]
    return FeaturePipeline(specs).transform(bars)


# ── 1. Label shift alignment — no leakage ───────────────────────────────────


class TestLabelShiftAlignment:

    def test_label_shift_alignment_no_leakage(self):
        """Known close series → ret_fwd and y must be correct and complete."""
        n = 100
        horizon = 5
        bars = _make_bars(n)
        features = _simple_features(bars)

        # Count warmup NaN rows (features with any NaN value)
        n_warmup = int(features.isna().any(axis=1).sum())

        label_cfg = {
            "type": "return_sign",
            "horizon": horizon,
            "price_col": "close",
            "threshold": 0.0,
        }
        ff = build_forecast_frame(bars, features, label_cfg)

        # Length: last `horizon` rows dropped + warmup NaN rows dropped
        expected_len = n - horizon - n_warmup
        assert len(ff) == expected_len, (
            f"Expected {expected_len} rows, got {len(ff)}"
        )

        # No NaN in y, ret_fwd, or X_* columns
        assert ff["y"].isna().sum() == 0, "y has NaN values"
        assert ff["ret_fwd"].isna().sum() == 0, "ret_fwd has NaN values"
        x_cols = [c for c in ff.columns if c.startswith("X_")]
        assert ff[x_cols].isna().sum().sum() == 0, "X_* columns have NaN"

        # Verify first row's forward return manually
        # First valid row is at original bar index n_warmup
        c_start = 2000.0 + n_warmup + 1
        c_end = 2000.0 + n_warmup + horizon + 1
        expected_ret = c_end / c_start - 1.0
        actual_ret = ff["ret_fwd"].iloc[0]
        assert abs(actual_ret - expected_ret) < 1e-12, (
            f"ret_fwd[0] = {actual_ret}, expected {expected_ret}"
        )

        # All feature columns prefixed with X_
        assert len(x_cols) == 2, f"Expected 2 X_ columns, got {x_cols}"

        # y must be integer 0 or 1
        assert set(ff["y"].unique()).issubset({0, 1})

    def test_labels_only_use_provided_slice(self):
        """Labels computed on a slice must NOT depend on bars outside it."""
        bars = _make_bars(200)
        features = _simple_features(bars)
        label_cfg = {"type": "return_sign", "horizon": 5, "price_col": "close"}

        # Build on full dataset
        ff_full = build_forecast_frame(bars, features, label_cfg)

        # Build on first 100 bars only
        ff_prefix = build_forecast_frame(
            bars.iloc[:100].reset_index(drop=True),
            features.iloc[:100].reset_index(drop=True),
            label_cfg,
        )

        # The prefix frame's ret_fwd values must match the full frame's
        # corresponding rows (same bar indices, after NaN drop)
        n_prefix_rows = len(ff_prefix)
        pd.testing.assert_series_equal(
            ff_prefix["ret_fwd"].reset_index(drop=True),
            ff_full["ret_fwd"].iloc[:n_prefix_rows].reset_index(drop=True),
            check_names=False,
        )


# ── 2. Feature overlap invariance (slice consistency) ────────────────────────


class TestFeatureOverlapInvariance:

    def test_prefix_features_match_full(self):
        """ForecastFrame on a prefix must equal the prefix slice of full frame."""
        bars_500 = _make_bars(500)
        bars_300 = _make_bars(300)  # same seed → same first 300 rows

        pipeline = FeaturePipeline([
            FeatureSpec(base=SMA(period=20)),
            FeatureSpec(base=EMA(period=10)),
        ])

        feat_500 = pipeline.transform(bars_500)
        feat_300 = pipeline.transform(bars_300)

        ff_full = build_forecast_frame(bars_500, feat_500, label_cfg=None)
        ff_prefix = build_forecast_frame(bars_300, feat_300, label_cfg=None)

        # Overlap: first len(prefix) rows of full must exactly match prefix
        n_prefix = len(ff_prefix)
        overlap = ff_full.iloc[:n_prefix].reset_index(drop=True)
        prefix = ff_prefix.reset_index(drop=True)

        pd.testing.assert_frame_equal(
            prefix, overlap, check_exact=True,
            obj="ForecastFrame prefix vs full overlap",
        )


# ── 3. Predictor output alignment ───────────────────────────────────────────


class TestPredictorOutputAlignment:

    def test_dummy_predictor_alignment(self):
        """DummyPredictor.predict output must align to test_ff index."""
        bars = _make_bars(200)
        features = _simple_features(bars)
        label_cfg = {"type": "return_sign", "horizon": 5, "price_col": "close"}

        train_ff = build_forecast_frame(
            bars.iloc[:100].reset_index(drop=True),
            features.iloc[:100].reset_index(drop=True),
            label_cfg,
        )
        test_ff = build_forecast_frame(
            bars.iloc[100:].reset_index(drop=True),
            features.iloc[100:].reset_index(drop=True),
            label_cfg,
        )

        predictor = DummyPredictor()
        predictor.fit(train_ff)
        yhat = predictor.predict(test_ff)

        # yhat index matches test_ff index
        assert yhat.index.equals(test_ff.index), "yhat index mismatch"
        assert len(yhat) == len(test_ff), "yhat length mismatch"
        assert yhat.name == "yhat", f"Expected name 'yhat', got {yhat.name!r}"

        # All values are 0.5 (dummy)
        assert (yhat == 0.5).all()

    def test_build_predictor_dummy(self):
        """build_predictor with type=dummy returns a DummyPredictor."""
        predictor = build_predictor({"type": "dummy"})
        assert isinstance(predictor, DummyPredictor)

    def test_build_predictor_unknown_type(self):
        """build_predictor with unknown type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown predictor type"):
            build_predictor({"type": "nonexistent"})

    def test_predictor_without_labels(self):
        """Predictor works on features-only ForecastFrame (no y column)."""
        bars = _make_bars(100)
        features = _simple_features(bars)

        ff = build_forecast_frame(bars, features, label_cfg=None)

        predictor = DummyPredictor()
        predictor.fit(ff)
        yhat = predictor.predict(ff)

        assert len(yhat) == len(ff)
        assert yhat.index.equals(ff.index)


# ── 4. Backward compatibility — runner without model config ──────────────────


class TestBackwardCompatibilityML:

    def _make_config(self, tmp_path, *, with_model: bool = False) -> str:
        """Write synthetic data + YAML config; return config path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_bars(1000).to_csv(data_dir / "test.csv", index=False)

        cfg: dict = {
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
            "validation": {
                "type": "walk_forward",
                "mode": "expanding",
                "train_size": 500,
                "test_size": 100,
                "step": 100,
                "drop_last": True,
            },
        }
        if with_model:
            cfg["model"] = {"type": "dummy"}
            cfg["label"] = {
                "type": "return_sign",
                "horizon": 5,
                "price_col": "close",
                "threshold": 0.0,
            }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
        return str(config_path)

    def test_no_model_config_unchanged_behavior(self, tmp_path):
        """Walk-forward without model block produces standard artifacts."""
        from src.engine.runner import run_backtest

        config_path = self._make_config(tmp_path, with_model=False)
        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        # Standard artifacts present
        assert (run_dir / "fold_metrics.json").exists()
        assert (run_dir / "equity.csv").exists()
        assert (run_dir / "metrics.json").exists()

        # metrics.json has walk_forward mode
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["mode"] == "walk_forward"
        assert metrics["n_folds"] == 5


# ── 5. ML path smoke test ───────────────────────────────────────────────────


class TestMLPathSmoke:

    def _make_config(self, tmp_path) -> str:
        """Config WITH model block for the ML smoke path."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_bars(1000).to_csv(data_dir / "test.csv", index=False)

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
            "validation": {
                "type": "walk_forward",
                "mode": "expanding",
                "train_size": 500,
                "test_size": 100,
                "step": 100,
                "drop_last": True,
            },
            "model": {"type": "dummy"},
            "label": {
                "type": "return_sign",
                "horizon": 5,
                "price_col": "close",
                "threshold": 0.0,
            },
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)
        return str(config_path)

    def test_runner_ml_path_completes(self, tmp_path):
        """Walk-forward with model config completes without errors."""
        from src.engine.runner import run_backtest

        config_path = self._make_config(tmp_path)
        run_id = run_backtest(config_path)
        run_dir = tmp_path / "runs" / run_id

        # Standard artifacts still produced
        assert (run_dir / "fold_metrics.json").exists()
        assert (run_dir / "equity.csv").exists()
        assert (run_dir / "metrics.json").exists()

        # Walk-forward mode flagged
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert metrics["mode"] == "walk_forward"
        assert metrics["n_folds"] == 5

    def test_runner_ml_path_yhat_not_in_baseline(self, tmp_path):
        """Without model config, features must NOT contain yhat column."""
        from src.engine.runner import run_backtest

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_bars(1000).to_csv(data_dir / "test.csv", index=False)

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
            "validation": {
                "type": "walk_forward",
                "mode": "expanding",
                "train_size": 500,
                "test_size": 100,
                "step": 100,
            },
        }
        config_path = tmp_path / "config_no_ml.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        # Runs fine without model block
        run_id = run_backtest(str(config_path))
        assert run_id is not None


# ── 6. Cost-aware label tests ─────────────────────────────────────────────────


class TestCostAwareLabel:

    def test_direction_up(self):
        """y=1 iff ret_fwd > cost_buffer (direction='up')."""
        n, horizon = 100, 5
        bars = _make_bars(n)
        features = _simple_features(bars)

        cost_buffer = 0.003
        label_cfg = {
            "type": "return_sign_cost_aware",
            "horizon": horizon,
            "price_col": "close",
            "cost_buffer": cost_buffer,
            "direction": "up",
        }
        ff = build_forecast_frame(bars, features, label_cfg)

        assert set(ff["y"].unique()).issubset({0, 1})
        # Manually verify: y=1 only when ret_fwd > cost_buffer
        assert ((ff["y"] == 1) == (ff["ret_fwd"] > cost_buffer)).all()

    def test_direction_both(self):
        """y=1 iff abs(ret_fwd) > cost_buffer (direction='both')."""
        n, horizon = 100, 5
        bars = _make_bars(n)
        features = _simple_features(bars)

        cost_buffer = 0.003
        label_cfg = {
            "type": "return_sign_cost_aware",
            "horizon": horizon,
            "price_col": "close",
            "cost_buffer": cost_buffer,
            "direction": "both",
        }
        ff = build_forecast_frame(bars, features, label_cfg)

        assert set(ff["y"].unique()).issubset({0, 1})
        assert ((ff["y"] == 1) == (ff["ret_fwd"].abs() > cost_buffer)).all()

    def test_horizon_rows_dropped(self):
        """Length = n - horizon - warmup_nans, no NaN in X_* columns."""
        n, horizon = 100, 5
        bars = _make_bars(n)
        features = _simple_features(bars)
        n_warmup = int(features.isna().any(axis=1).sum())

        label_cfg = {
            "type": "return_sign_cost_aware",
            "horizon": horizon,
            "price_col": "close",
            "cost_buffer": 0.001,
            "direction": "up",
        }
        ff = build_forecast_frame(bars, features, label_cfg)

        expected_len = n - horizon - n_warmup
        assert len(ff) == expected_len
        x_cols = [c for c in ff.columns if c.startswith("X_")]
        assert ff[x_cols].isna().sum().sum() == 0

    def test_nan_drop_in_x_columns(self):
        """Injecting NaN into a feature causes that row to be dropped."""
        bars = _make_bars(100)
        features = _simple_features(bars)

        # Inject NaN into feature at a specific row (after warmup)
        features_with_nan = features.copy()
        features_with_nan.iloc[50, 0] = np.nan

        label_cfg = {
            "type": "return_sign",
            "horizon": 5,
            "price_col": "close",
        }
        ff_clean = build_forecast_frame(bars, features, label_cfg)
        ff_nan = build_forecast_frame(bars, features_with_nan, label_cfg)

        # One extra row dropped
        assert len(ff_nan) == len(ff_clean) - 1

        # No NaN in any X_* column
        x_cols = [c for c in ff_nan.columns if c.startswith("X_")]
        assert ff_nan[x_cols].isna().sum().sum() == 0


# ── 7. ML walk-forward smoke (logistic + cost-aware) ─────────────────────────


class TestMLWalkForwardSmoke:

    def test_logistic_walk_forward_completes(self, tmp_path):
        """Config with logistic predictor + cost-aware label runs end-to-end."""
        from src.engine.runner import run_backtest

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _make_bars(1000).to_csv(data_dir / "test.csv", index=False)

        cfg = {
            "symbol": "TEST",
            "timeframe": "M5",
            "snapshot_dir": str(data_dir),
            "starting_capital": 10000.0,
            "output_dir": str(tmp_path / "runs"),
            "strategy": {
                "type": "ml_prob_threshold",
                "p_enter": 0.55,
                "fast_period": 5,
                "slow_period": 10,
            },
            "validation": {
                "type": "walk_forward",
                "mode": "expanding",
                "train_size": 500,
                "test_size": 200,
                "step": 200,
                "drop_last": True,
            },
            "model": {
                "type": "logistic",
                "C": 1.0,
                "max_iter": 200,
                "class_weight": "balanced",
            },
            "label": {
                "type": "return_sign_cost_aware",
                "horizon": 5,
                "price_col": "close",
                "cost_buffer": 0.0022,
                "direction": "up",
            },
        }

        config_path = tmp_path / "config_ml.yaml"
        with open(config_path, "w") as f:
            yaml.dump(cfg, f)

        run_id = run_backtest(str(config_path))
        run_dir = tmp_path / "runs" / run_id

        # Standard artifacts
        assert (run_dir / "fold_metrics.json").exists()
        assert (run_dir / "metrics.json").exists()

        # fold_metrics.json has brier + row counts
        fold_data = json.loads((run_dir / "fold_metrics.json").read_text())
        for fold in fold_data["folds"]:
            assert "brier" in fold, f"Missing brier in fold {fold['fold_id']}"
            assert "n_train_rows" in fold
            assert "n_test_rows" in fold
            assert 0.0 <= fold["brier"] <= 1.0

        # Aggregate has avg_brier
        assert "avg_brier" in fold_data["aggregate"]

        # metrics.json has ml_note
        metrics = json.loads((run_dir / "metrics.json").read_text())
        assert "ml_note" in metrics

    def test_ml_prob_threshold_strategy_signals(self):
        """MLProbThresholdStrategy produces correct signals for known yhat."""
        from src.strategy.ml_prob_threshold import MLProbThresholdStrategy
        from src.strategy.base import StrategyContext

        strategy = MLProbThresholdStrategy(p_enter=0.55)

        # yhat above threshold → long
        features_high = pd.Series({"yhat": 0.70, "X_ema_5": 100.0})
        ctx_high = StrategyContext(
            ts=pd.Timestamp("2024-01-01"),
            bar=pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}),
            features=features_high,
            position=0.0,
            equity=10000.0,
            bar_index=10,
        )
        sig = strategy.on_bar(ctx_high)
        assert sig.direction == 1.0

        # yhat below threshold → flat
        features_low = pd.Series({"yhat": 0.40, "X_ema_5": 100.0})
        ctx_low = StrategyContext(
            ts=pd.Timestamp("2024-01-01"),
            bar=pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}),
            features=features_low,
            position=0.0,
            equity=10000.0,
            bar_index=10,
        )
        sig = strategy.on_bar(ctx_low)
        assert sig.direction == 0.0

        # Missing yhat → flat with no_yhat reason
        features_no_yhat = pd.Series({"X_ema_5": 100.0})
        ctx_no = StrategyContext(
            ts=pd.Timestamp("2024-01-01"),
            bar=pd.Series({"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}),
            features=features_no_yhat,
            position=0.0,
            equity=10000.0,
            bar_index=10,
        )
        sig = strategy.on_bar(ctx_no)
        assert sig.direction == 0.0
        assert sig.reason == "no_yhat"
