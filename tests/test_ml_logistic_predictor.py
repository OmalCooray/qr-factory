"""Tests for SklearnLogisticPredictor."""

import numpy as np
import pandas as pd

from src.ml.predictor import SklearnLogisticPredictor, build_predictor


def _make_synthetic_ff(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Synthetic ForecastFrame where y is learnable: y=1 if X_ema > X_sma."""
    rng = np.random.default_rng(seed)
    x_ema = rng.normal(0, 1, n)
    x_sma = rng.normal(0, 1, n)
    y = (x_ema > x_sma).astype(int)
    return pd.DataFrame({
        "X_ema_5": x_ema,
        "X_sma_10": x_sma,
        "y": y,
        "ret_fwd": rng.normal(0, 0.001, n),
    })


class TestSklearnLogisticPredictor:

    def test_logistic_predictor_smoke(self):
        """Fit on synthetic data, predict, check output shape and range."""
        ff = _make_synthetic_ff(500)
        train_ff = ff.iloc[:400]
        test_ff = ff.iloc[400:]

        predictor = SklearnLogisticPredictor(C=1.0, max_iter=200)
        predictor.fit(train_ff)
        yhat = predictor.predict(test_ff)

        # Output length and index alignment
        assert len(yhat) == len(test_ff)
        assert yhat.index.equals(test_ff.index)
        assert yhat.name == "yhat"

        # Probabilities in [0, 1]
        assert (yhat >= 0.0).all()
        assert (yhat <= 1.0).all()

    def test_build_predictor_logistic(self):
        """build_predictor with type=logistic returns SklearnLogisticPredictor."""
        predictor = build_predictor({"type": "logistic", "C": 0.5})
        assert isinstance(predictor, SklearnLogisticPredictor)

    def test_logistic_predictor_determinism(self):
        """Two identical fits produce identical predictions."""
        ff = _make_synthetic_ff(300)
        train_ff = ff.iloc[:200]
        test_ff = ff.iloc[200:]

        pred_a = SklearnLogisticPredictor(C=1.0, max_iter=200)
        pred_a.fit(train_ff)
        yhat_a = pred_a.predict(test_ff)

        pred_b = SklearnLogisticPredictor(C=1.0, max_iter=200)
        pred_b.fit(train_ff)
        yhat_b = pred_b.predict(test_ff)

        pd.testing.assert_series_equal(yhat_a, yhat_b, check_exact=True)
