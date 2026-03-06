"""Tests for XGBoost and LightGBM predictors."""

import numpy as np
import pandas as pd

from src.ml.predictor import LightGBMPredictor, XGBoostPredictor, build_predictor


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


# ── XGBoost ──────────────────────────────────────────────────────────────────


class TestXGBoostPredictor:

    def test_xgboost_smoke(self):
        """Fit on synthetic data, predict, check output shape and range."""
        ff = _make_synthetic_ff(500)
        train_ff = ff.iloc[:400]
        test_ff = ff.iloc[400:]

        pred = XGBoostPredictor(n_estimators=10, max_depth=2)
        pred.fit(train_ff)
        yhat = pred.predict(test_ff)

        assert len(yhat) == len(test_ff)
        assert yhat.index.equals(test_ff.index)
        assert yhat.name == "yhat"
        assert (yhat >= 0.0).all()
        assert (yhat <= 1.0).all()

    def test_build_predictor_xgboost(self):
        """build_predictor with type=xgboost returns XGBoostPredictor."""
        pred = build_predictor({"type": "xgboost", "n_estimators": 10})
        assert isinstance(pred, XGBoostPredictor)

    def test_xgboost_determinism(self):
        """Two identical fits produce identical predictions."""
        ff = _make_synthetic_ff(300)
        train_ff = ff.iloc[:200]
        test_ff = ff.iloc[200:]

        pred_a = XGBoostPredictor(n_estimators=10, random_state=0)
        pred_a.fit(train_ff)
        yhat_a = pred_a.predict(test_ff)

        pred_b = XGBoostPredictor(n_estimators=10, random_state=0)
        pred_b.fit(train_ff)
        yhat_b = pred_b.predict(test_ff)

        pd.testing.assert_series_equal(yhat_a, yhat_b, check_exact=True)


# ── LightGBM ─────────────────────────────────────────────────────────────────


class TestLightGBMPredictor:

    def test_lightgbm_smoke(self):
        """Fit on synthetic data, predict, check output shape and range."""
        ff = _make_synthetic_ff(500)
        train_ff = ff.iloc[:400]
        test_ff = ff.iloc[400:]

        pred = LightGBMPredictor(n_estimators=10, max_depth=2)
        pred.fit(train_ff)
        yhat = pred.predict(test_ff)

        assert len(yhat) == len(test_ff)
        assert yhat.index.equals(test_ff.index)
        assert yhat.name == "yhat"
        assert (yhat >= 0.0).all()
        assert (yhat <= 1.0).all()

    def test_build_predictor_lightgbm(self):
        """build_predictor with type=lightgbm returns LightGBMPredictor."""
        pred = build_predictor({"type": "lightgbm", "n_estimators": 10})
        assert isinstance(pred, LightGBMPredictor)

    def test_lightgbm_determinism(self):
        """Two identical fits produce identical predictions."""
        ff = _make_synthetic_ff(300)
        train_ff = ff.iloc[:200]
        test_ff = ff.iloc[200:]

        pred_a = LightGBMPredictor(n_estimators=10, random_state=0)
        pred_a.fit(train_ff)
        yhat_a = pred_a.predict(test_ff)

        pred_b = LightGBMPredictor(n_estimators=10, random_state=0)
        pred_b.fit(train_ff)
        yhat_b = pred_b.predict(test_ff)

        pd.testing.assert_series_equal(yhat_a, yhat_b, check_exact=True)
