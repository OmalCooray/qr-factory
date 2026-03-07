"""Tests for PlattCalibrator (src/ml/calibration.py)."""

import numpy as np
import pandas as pd
import pytest

from src.ml.calibration import PlattCalibrator


def _make_synthetic_probs(n: int = 500, seed: int = 42):
    """Generate synthetic raw probabilities and ground truth labels.

    Returns (y_prob_raw, y_true) where y_true is correlated with y_prob_raw.
    """
    rng = np.random.default_rng(seed)
    # Raw probs: biased (squeezed) so calibration has something to fix
    z = rng.normal(0, 1, n)
    y_prob_raw = 1 / (1 + np.exp(-0.5 * z))  # squeezed sigmoid
    y_true = (rng.random(n) < (1 / (1 + np.exp(-z)))).astype(int)
    return y_prob_raw, y_true


class TestPlattCalibrator:

    def test_unfitted_returns_input(self):
        """Unfitted calibrator returns raw probs unchanged (safety fallback)."""
        cal = PlattCalibrator()
        raw = np.array([0.1, 0.5, 0.9])
        result = cal.transform(raw)
        np.testing.assert_array_equal(result, raw)
        assert not cal.fitted

    def test_fitted_changes_probabilities(self):
        """After fitting, output differs from input."""
        y_prob_raw, y_true = _make_synthetic_probs(500)
        cal = PlattCalibrator()
        cal.fit(y_prob_raw, y_true)

        calibrated = cal.transform(y_prob_raw)
        assert cal.fitted
        # At least some values should change
        assert not np.allclose(calibrated, y_prob_raw, atol=1e-6)

    def test_output_in_valid_range(self):
        """Calibrated probs are in [0, 1]."""
        y_prob_raw, y_true = _make_synthetic_probs(1000, seed=99)
        cal = PlattCalibrator()
        cal.fit(y_prob_raw, y_true)

        calibrated = cal.transform(y_prob_raw)
        assert (calibrated >= 0.0).all()
        assert (calibrated <= 1.0).all()

    def test_perfect_model_stays_calibrated(self):
        """Already-calibrated probs don't get distorted much."""
        rng = np.random.default_rng(123)
        n = 2000
        # Generate well-calibrated probabilities
        y_prob = rng.random(n)
        y_true = (rng.random(n) < y_prob).astype(int)

        cal = PlattCalibrator()
        cal.fit(y_prob, y_true)
        calibrated = cal.transform(y_prob)

        # The mean shift should be small for well-calibrated inputs
        mean_shift = abs(calibrated.mean() - y_prob.mean())
        assert mean_shift < 0.05, f"Mean shift too large: {mean_shift:.4f}"

    def test_transform_series(self):
        """transform_series preserves index and name."""
        y_prob_raw, y_true = _make_synthetic_probs(200)
        cal = PlattCalibrator()
        cal.fit(y_prob_raw, y_true)

        idx = pd.RangeIndex(100, 300)
        s = pd.Series(y_prob_raw, index=idx, name="yhat")
        result = cal.transform_series(s)

        assert isinstance(result, pd.Series)
        assert result.index.equals(idx)
        assert result.name == "yhat"
        assert len(result) == 200

    def test_predictions_csv_has_raw_column(self, tmp_path):
        """When calibration active, pred_records include y_prob_raw."""
        # Simulate the runner logic for building pred_records
        y_prob_raw, y_true = _make_synthetic_probs(100)
        cal = PlattCalibrator()
        cal.fit(y_prob_raw, y_true)
        y_prob_cal = cal.transform(y_prob_raw)

        pred_records = pd.DataFrame({
            "fold_id": 0,
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="5min"),
            "y_true": y_true,
            "y_prob": y_prob_cal,
        })
        # Calibration active: add raw column
        pred_records["y_prob_raw"] = y_prob_raw

        csv_path = tmp_path / "predictions.csv"
        pred_records.to_csv(csv_path, index=False)

        loaded = pd.read_csv(csv_path)
        assert "y_prob" in loaded.columns
        assert "y_prob_raw" in loaded.columns
        assert len(loaded) == 100
