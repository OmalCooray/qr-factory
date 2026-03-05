"""Tests for ML improvements: hysteresis, scaling, gradient boosting, rich features."""

import numpy as np
import pandas as pd
import pytest

from src.indicators.core.pipeline import FeaturePipeline, FeatureSpec
from src.indicators.impl.ema import EMA
from src.indicators.impl.ma import SMA
from src.ml.predictor import (
    SklearnGBClassifierPredictor,
    SklearnLogisticPredictor,
    build_predictor,
)
from src.strategy.base import StrategyContext
from src.strategy.ml_prob_threshold import (
    MLProbThresholdStrategy,
    _build_feature_specs,
    build,
)
from src.strategy.signal import Signal


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_ctx(
    yhat: float | None,
    position: float = 0.0,
    bar_index: int = 100,
) -> StrategyContext:
    """Build a StrategyContext with a given yhat and position."""
    features = {"X_ema_5": 100.0}
    if yhat is not None:
        features["yhat"] = yhat
    return StrategyContext(
        ts=pd.Timestamp("2024-01-01"),
        bar=pd.Series(
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
        ),
        features=pd.Series(features),
        position=position,
        equity=10000.0,
        bar_index=bar_index,
    )


def _make_synthetic_ff(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic ForecastFrame where y is learnable: y=1 if X_a > X_b."""
    rng = np.random.default_rng(seed)
    x_a = rng.normal(0, 1, n)
    x_b = rng.normal(0, 1, n)
    y = (x_a > x_b).astype(int)
    return pd.DataFrame(
        {
            "X_feat_a": x_a,
            "X_feat_b": x_b,
            "y": y,
            "ret_fwd": rng.normal(0, 0.001, n),
        }
    )


# ── 1. Hysteresis tests ─────────────────────────────────────────────────────


class TestHysteresis:

    def test_enter_above_p_enter(self):
        """When flat, yhat >= p_enter → enter long."""
        strategy = MLProbThresholdStrategy(p_enter=0.55, p_exit=0.45)
        sig = strategy.on_bar(_make_ctx(yhat=0.60, position=0.0))
        assert sig.direction == 1.0

    def test_stay_flat_below_p_enter(self):
        """When flat, yhat < p_enter → stay flat."""
        strategy = MLProbThresholdStrategy(p_enter=0.55, p_exit=0.45)
        sig = strategy.on_bar(_make_ctx(yhat=0.50, position=0.0))
        assert sig.direction == 0.0

    def test_hold_in_dead_band(self):
        """When long, yhat in [p_exit, p_enter) → hold (dead-band)."""
        strategy = MLProbThresholdStrategy(p_enter=0.55, p_exit=0.45)
        # Simulate being already long
        sig = strategy.on_bar(_make_ctx(yhat=0.50, position=1.0))
        assert sig.direction == 1.0, "Should hold in dead-band"

    def test_exit_below_p_exit(self):
        """When long, yhat < p_exit → exit."""
        strategy = MLProbThresholdStrategy(p_enter=0.55, p_exit=0.45)
        # First bar as long to set _bars_in_position
        strategy.on_bar(_make_ctx(yhat=0.60, position=1.0))
        sig = strategy.on_bar(_make_ctx(yhat=0.40, position=1.0))
        assert sig.direction == 0.0, "Should exit below p_exit"

    def test_no_enter_in_dead_band_from_flat(self):
        """When flat, yhat in [p_exit, p_enter) → remain flat (no entry)."""
        strategy = MLProbThresholdStrategy(p_enter=0.55, p_exit=0.45)
        sig = strategy.on_bar(_make_ctx(yhat=0.50, position=0.0))
        assert sig.direction == 0.0, "Should not enter from flat in dead-band"


# ── 2. Min hold bars ────────────────────────────────────────────────────────


class TestMinHoldBars:

    def test_min_hold_prevents_early_exit(self):
        """Even if yhat < p_exit, must hold for min_hold_bars."""
        strategy = MLProbThresholdStrategy(
            p_enter=0.55, p_exit=0.45, min_hold_bars=3
        )
        # Bar 1: enter
        strategy.on_bar(_make_ctx(yhat=0.60, position=0.0))
        # Bar 2: long, yhat drops below p_exit but min_hold not met
        sig = strategy.on_bar(_make_ctx(yhat=0.30, position=1.0))
        assert sig.direction == 1.0, "Must hold — min_hold_bars=3, only 1 bar held"

    def test_min_hold_allows_exit_after_enough_bars(self):
        """After holding min_hold_bars, exit when yhat < p_exit."""
        strategy = MLProbThresholdStrategy(
            p_enter=0.55, p_exit=0.45, min_hold_bars=2
        )
        # Enter
        strategy.on_bar(_make_ctx(yhat=0.60, position=0.0))
        # Bar 1 as long
        strategy.on_bar(_make_ctx(yhat=0.50, position=1.0))
        # Bar 2 as long — now bars_held=2 >= min_hold_bars
        sig = strategy.on_bar(_make_ctx(yhat=0.30, position=1.0))
        assert sig.direction == 0.0, "Should exit — held enough bars"


# ── 3. Backward compatibility ────────────────────────────────────────────────


class TestBackwardCompat:

    def test_no_p_exit_uses_p_enter(self):
        """p_exit=None → effective_p_exit = p_enter (old behavior)."""
        strategy = MLProbThresholdStrategy(p_enter=0.55, p_exit=None)

        # When flat: yhat=0.54 < p_enter → flat
        sig = strategy.on_bar(_make_ctx(yhat=0.54, position=0.0))
        assert sig.direction == 0.0

        # When flat: yhat=0.56 >= p_enter → enter
        sig = strategy.on_bar(_make_ctx(yhat=0.56, position=0.0))
        assert sig.direction == 1.0

        # When long: yhat=0.54 < p_enter (=effective_p_exit) → exit
        sig = strategy.on_bar(_make_ctx(yhat=0.54, position=1.0))
        assert sig.direction == 0.0

    def test_legacy_build_no_features_key(self):
        """build() without 'features' key uses legacy EMA+SMA."""
        strategy, specs = build({"p_enter": 0.55, "fast_period": 5, "slow_period": 10})
        assert len(specs) == 2
        assert isinstance(specs[0].base, EMA)
        assert isinstance(specs[1].base, SMA)
        assert strategy.p_exit is None
        assert strategy.min_hold_bars == 0


# ── 4. Config-driven feature specs ──────────────────────────────────────────


class TestConfigDrivenFeatures:

    def test_build_with_features_key(self):
        """build() with 'features' key creates rich FeatureSpecs."""
        params = {
            "p_enter": 0.55,
            "p_exit": 0.45,
            "min_hold_bars": 5,
            "features": [
                {"type": "ema", "period": 5},
                {"type": "sma", "period": 10},
                {"type": "rsi", "period": 14},
                {"type": "atr", "period": 14},
            ],
        }
        strategy, specs = build(params)
        assert len(specs) == 4
        assert strategy.p_exit == 0.45
        assert strategy.min_hold_bars == 5
        assert strategy._warmup_bars > 0

    def test_build_feature_specs_with_transforms(self):
        """_build_feature_specs handles transforms correctly."""
        cfgs = [
            {"type": "ema", "period": 5, "transforms": [{"type": "zscore", "window": 20}]},
        ]
        specs = _build_feature_specs(cfgs)
        assert len(specs) == 1
        assert len(specs[0].transforms) == 1
        assert "zscore" in specs[0].name

    def test_unknown_indicator_raises(self):
        """Unknown indicator type in features list raises ValueError."""
        with pytest.raises(ValueError, match="Unknown indicator type"):
            _build_feature_specs([{"type": "nonexistent"}])

    def test_unknown_transform_raises(self):
        """Unknown transform type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown transform type"):
            _build_feature_specs([
                {"type": "ema", "period": 5, "transforms": [{"type": "bad_transform"}]}
            ])


# ── 5. Scaling in LogisticPredictor ──────────────────────────────────────────


class TestScaling:

    def test_logistic_scale_false_default(self):
        """scale=False by default — no scaler created."""
        pred = SklearnLogisticPredictor(scale=False)
        assert pred._scaler is None

    def test_logistic_scale_true_produces_valid_proba(self):
        """scale=True: fit+predict work and produce valid probabilities."""
        ff = _make_synthetic_ff(500)
        train_ff = ff.iloc[:400]
        test_ff = ff.iloc[400:]

        pred = SklearnLogisticPredictor(scale=True)
        pred.fit(train_ff)
        assert pred._scaler is not None
        yhat = pred.predict(test_ff)

        assert len(yhat) == len(test_ff)
        assert (yhat >= 0.0).all()
        assert (yhat <= 1.0).all()

    def test_build_predictor_scale_kwarg(self):
        """build_predictor passes scale=True to SklearnLogisticPredictor."""
        pred = build_predictor({"type": "logistic", "scale": True})
        assert isinstance(pred, SklearnLogisticPredictor)
        assert pred._scale is True


# ── 6. Gradient Boosting predictor ───────────────────────────────────────────


class TestGradientBoosting:

    def test_gb_predictor_smoke(self):
        """SklearnGBClassifierPredictor fit+predict on synthetic data."""
        ff = _make_synthetic_ff(500)
        train_ff = ff.iloc[:400]
        test_ff = ff.iloc[400:]

        pred = SklearnGBClassifierPredictor(
            n_estimators=10, max_depth=2, min_samples_leaf=10
        )
        pred.fit(train_ff)
        yhat = pred.predict(test_ff)

        assert len(yhat) == len(test_ff)
        assert yhat.index.equals(test_ff.index)
        assert yhat.name == "yhat"
        assert (yhat >= 0.0).all()
        assert (yhat <= 1.0).all()

    def test_build_predictor_gradient_boosting(self):
        """build_predictor with type=gradient_boosting works."""
        pred = build_predictor({"type": "gradient_boosting", "n_estimators": 10})
        assert isinstance(pred, SklearnGBClassifierPredictor)

    def test_gb_determinism(self):
        """Two identical GB fits produce identical predictions."""
        ff = _make_synthetic_ff(300)
        train_ff = ff.iloc[:200]
        test_ff = ff.iloc[200:]

        pred_a = SklearnGBClassifierPredictor(n_estimators=10, random_state=0)
        pred_a.fit(train_ff)
        yhat_a = pred_a.predict(test_ff)

        pred_b = SklearnGBClassifierPredictor(n_estimators=10, random_state=0)
        pred_b.fit(train_ff)
        yhat_b = pred_b.predict(test_ff)

        pd.testing.assert_series_equal(yhat_a, yhat_b, check_exact=True)
