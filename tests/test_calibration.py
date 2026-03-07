"""Tests for the calibration evaluation module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eval.calibration import compute_calibration, plot_reliability_diagram, write_calibration_artifacts


class TestComputeCalibration:
    """Core ECE computation tests."""

    def test_perfect_calibration_ece_zero(self):
        """When predicted probability matches observed rate per bin, ECE ~ 0."""
        rng = np.random.RandomState(42)
        n = 10_000
        # Generate y_prob spread across bins, then set y_true so that
        # each sample is 1 with probability = y_prob
        y_prob = rng.uniform(0.0, 1.0, size=n)
        y_true = (rng.uniform(0, 1, size=n) < y_prob).astype(int)

        df = pd.DataFrame({"y_true": y_true, "y_prob": y_prob})
        cal = compute_calibration(df, n_bins=10)

        assert cal["ece"] < 0.03  # should be very close to 0 with 10k samples
        assert cal["n_predictions"] == n

    def test_worst_calibration_high_ece(self):
        """All predictions = 1.0 but y_true = 0 → high ECE."""
        n = 100
        df = pd.DataFrame({
            "y_true": np.zeros(n, dtype=int),
            "y_prob": np.ones(n),
        })
        cal = compute_calibration(df, n_bins=10)

        # ECE should be 1.0: all in last bin, mean_pred=1.0, obs_rate=0.0
        assert cal["ece"] == pytest.approx(1.0, abs=1e-6)

    def test_uniform_random_predictions(self):
        """Random predictions → ECE in reasonable range (not extreme)."""
        rng = np.random.RandomState(99)
        n = 5000
        df = pd.DataFrame({
            "y_true": rng.randint(0, 2, size=n),
            "y_prob": rng.uniform(0, 1, size=n),
        })
        cal = compute_calibration(df, n_bins=10)

        # Random preds on balanced data: ECE in a reasonable range
        assert 0.0 < cal["ece"] < 0.35
        assert cal["n_predictions"] == n

    def test_empty_predictions_returns_empty(self):
        """None or empty DataFrame → empty dict."""
        assert compute_calibration(None) == {}
        assert compute_calibration(pd.DataFrame()) == {}

    def test_missing_columns_raises(self):
        """Missing y_true or y_prob → ValueError."""
        df_no_prob = pd.DataFrame({"y_true": [1, 0, 1]})
        with pytest.raises(ValueError, match="y_prob"):
            compute_calibration(df_no_prob)

        df_no_true = pd.DataFrame({"y_prob": [0.9, 0.1, 0.5]})
        with pytest.raises(ValueError, match="y_true"):
            compute_calibration(df_no_true)

    def test_bin_counts_sum_to_n(self):
        """Sum of bin counts should equal n_predictions."""
        rng = np.random.RandomState(7)
        n = 500
        df = pd.DataFrame({
            "y_true": rng.randint(0, 2, size=n),
            "y_prob": rng.uniform(0, 1, size=n),
        })
        cal = compute_calibration(df, n_bins=10)

        total_count = sum(b["count"] for b in cal["bins"])
        assert total_count == cal["n_predictions"]

    def test_edge_all_same_probability(self):
        """All y_prob = 0.5, only one bin non-empty."""
        n = 200
        df = pd.DataFrame({
            "y_true": np.concatenate([np.ones(100), np.zeros(100)]),
            "y_prob": np.full(n, 0.5),
        })
        cal = compute_calibration(df, n_bins=10)

        non_empty = [b for b in cal["bins"] if b["count"] > 0]
        assert len(non_empty) == 1
        assert non_empty[0]["count"] == n
        assert non_empty[0]["mean_predicted"] == pytest.approx(0.5)
        assert non_empty[0]["observed_positive_rate"] == pytest.approx(0.5)
        assert cal["ece"] == pytest.approx(0.0, abs=1e-6)


class TestCalibrationArtifacts:
    """End-to-end artifact writing tests."""

    def test_calibration_json_written(self, tmp_path):
        """write_calibration_artifacts writes calibration.json."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "y_true": rng.randint(0, 2, size=n),
            "y_prob": rng.uniform(0, 1, size=n),
        })
        (tmp_path / "plots").mkdir()

        cal = write_calibration_artifacts(tmp_path, df)

        cal_path = tmp_path / "calibration.json"
        assert cal_path.exists()
        data = json.loads(cal_path.read_text(encoding="utf-8"))
        assert "ece" in data
        assert "bins" in data
        assert len(data["bins"]) == 10
        assert data["n_predictions"] == n
        assert cal == data

    def test_reliability_diagram_written(self, tmp_path):
        """write_calibration_artifacts writes reliability_diagram.png."""
        rng = np.random.RandomState(42)
        n = 200
        df = pd.DataFrame({
            "y_true": rng.randint(0, 2, size=n),
            "y_prob": rng.uniform(0, 1, size=n),
        })
        (tmp_path / "plots").mkdir()

        write_calibration_artifacts(tmp_path, df)

        assert (tmp_path / "plots" / "reliability_diagram.png").exists()

    def test_no_predictions_skips_silently(self, tmp_path):
        """No predictions → empty dict, no files written."""
        cal = write_calibration_artifacts(tmp_path, None)
        assert cal == {}
        assert not (tmp_path / "calibration.json").exists()
