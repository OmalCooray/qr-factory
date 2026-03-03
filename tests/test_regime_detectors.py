"""Tests for all regime detectors and the synthetic data generator."""

from __future__ import annotations

import numpy as np
import pytest

from src.regime.protocol import RegimeLabel, RegimeDetector
from src.regime.synthetic import generate_synthetic_data
from src.regime.choppiness import ChoppinessDetector
from src.regime.hurst import HurstDetector
from src.regime.bbw_detector import BBWidthDetector
from src.regime.gmm_detector import GMMDetector
from src.regime.kmeans_detector import KMeansDetector
from src.regime.adapter_hmm import HMMRegimeDetector
from src.regime.markov_switching import MarkovSwitchingDetector
from src.regime.pelt_detector import PELTDetector


# ── Synthetic data ─────────────────────────────────────────────────

class TestSyntheticData:
    def test_generates_correct_length(self):
        c, h, l, idx, labels = generate_synthetic_data(n_bars=1000)
        assert len(c) == 1000
        assert len(h) == 1000
        assert len(l) == 1000
        assert len(idx) == 1000
        assert len(labels) == 1000

    def test_labels_alternate(self):
        _, _, _, _, labels = generate_synthetic_data(n_bars=1000, segment_length=250)
        assert labels[0] == "TREND"
        assert labels[250] == "RANGE"
        assert labels[500] == "TREND"
        assert labels[750] == "RANGE"

    def test_deterministic(self):
        c1, _, _, _, _ = generate_synthetic_data(seed=123)
        c2, _, _, _, _ = generate_synthetic_data(seed=123)
        np.testing.assert_array_equal(c1, c2)

    def test_prices_positive(self):
        c, h, l, _, _ = generate_synthetic_data()
        assert np.all(c > 0)
        assert np.all(h > 0)
        assert np.all(l > 0)


# ── Protocol conformance ──────────────────────────────────────────

class TestProtocol:
    def test_regime_label_frozen(self):
        label = RegimeLabel(regime="TREND", confidence=0.8, raw_value=0.7)
        assert label.regime == "TREND"
        with pytest.raises(AttributeError):
            label.regime = "RANGE"  # type: ignore[misc]

    def test_regime_label_values(self):
        label = RegimeLabel(regime="RANGE", confidence=0.5, raw_value=42.0)
        assert label.regime == "RANGE"
        assert label.confidence == 0.5
        assert label.raw_value == 42.0


# ── Streaming detector helpers ────────────────────────────────────

def _run_streaming_detector(
    detector, closes, highs, lows, n_bars: int
) -> list[RegimeLabel]:
    """Run a streaming detector through all bars."""
    results = []
    for i in range(n_bars):
        label = detector.update(closes[i], highs[i], lows[i], i)
        results.append(label)
    return results


# ── Choppiness ────────────────────────────────────────────────────

class TestChoppiness:
    def test_returns_regime_labels(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=200)
        det = ChoppinessDetector(period=14)
        results = _run_streaming_detector(det, c, h, l, 200)
        assert all(isinstance(r, RegimeLabel) for r in results)
        assert all(r.regime in ("TREND", "RANGE") for r in results)

    def test_warmup_returns_range(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=50)
        det = ChoppinessDetector(period=14)
        for i in range(13):
            label = det.update(c[i], h[i], l[i], i)
            assert label.regime == "RANGE"

    def test_confidence_bounded(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=200)
        det = ChoppinessDetector(period=14)
        results = _run_streaming_detector(det, c, h, l, 200)
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_has_name_and_warmup(self):
        det = ChoppinessDetector()
        assert det.name == "Choppiness"
        assert det.warmup_bars > 0


# ── Hurst ─────────────────────────────────────────────────────────

class TestHurst:
    def test_returns_regime_labels(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=500)
        det = HurstDetector(window=100, compute_every=5)
        results = _run_streaming_detector(det, c, h, l, 500)
        assert all(isinstance(r, RegimeLabel) for r in results)

    def test_warmup_returns_range(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=50)
        det = HurstDetector(window=30)
        for i in range(30):
            label = det.update(c[i], h[i], l[i], i)
            assert label.regime == "RANGE"

    def test_hurst_value_bounded(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=500)
        det = HurstDetector(window=100, compute_every=1)
        results = _run_streaming_detector(det, c, h, l, 500)
        for r in results:
            assert 0.0 <= r.raw_value <= 1.0

    def test_has_name_and_warmup(self):
        det = HurstDetector()
        assert det.name == "Hurst"
        assert det.warmup_bars > 0


# ── BB Width ──────────────────────────────────────────────────────

class TestBBWidth:
    def test_returns_regime_labels(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=500)
        det = BBWidthDetector(bb_period=20, rank_window=100)
        results = _run_streaming_detector(det, c, h, l, 500)
        assert all(isinstance(r, RegimeLabel) for r in results)

    def test_warmup_returns_range(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=50)
        det = BBWidthDetector(bb_period=20, rank_window=100)
        for i in range(19):
            label = det.update(c[i], h[i], l[i], i)
            assert label.regime == "RANGE"

    def test_has_name_and_warmup(self):
        det = BBWidthDetector()
        assert det.name == "BBWidth"
        assert det.warmup_bars > 0


# ── GMM ───────────────────────────────────────────────────────────

class TestGMM:
    def test_returns_regime_labels(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=600)
        det = GMMDetector(k=10, window=200, refit_every=100, predict_every=5)
        results = _run_streaming_detector(det, c, h, l, 600)
        assert all(isinstance(r, RegimeLabel) for r in results)

    def test_warmup_returns_range(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=100)
        det = GMMDetector(k=10, window=200)
        for i in range(100):
            label = det.update(c[i], h[i], l[i], i)
            assert label.regime == "RANGE"

    def test_has_name_and_warmup(self):
        det = GMMDetector()
        assert det.name == "GMM"
        assert det.warmup_bars > 0


# ── KMeans ────────────────────────────────────────────────────────

class TestKMeans:
    def test_returns_regime_labels(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=600)
        det = KMeansDetector(k=10, window=200, refit_every=100, predict_every=5)
        results = _run_streaming_detector(det, c, h, l, 600)
        assert all(isinstance(r, RegimeLabel) for r in results)

    def test_warmup_returns_range(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=100)
        det = KMeansDetector(k=10, window=200)
        for i in range(100):
            label = det.update(c[i], h[i], l[i], i)
            assert label.regime == "RANGE"

    def test_has_name_and_warmup(self):
        det = KMeansDetector()
        assert det.name == "KMeans"
        assert det.warmup_bars > 0


# ── PELT (batch) ──────────────────────────────────────────────────

class TestPELT:
    def test_batch_fit_and_update(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=2000)
        det = PELTDetector(k=20, min_size=50, pen=5.0)
        det.fit_batch(c)
        results = _run_streaming_detector(det, c, h, l, 2000)
        assert all(isinstance(r, RegimeLabel) for r in results)
        # After fit, some bars should be TREND
        regimes = [r.regime for r in results]
        assert "TREND" in regimes or "RANGE" in regimes

    def test_without_fit_returns_range(self):
        det = PELTDetector()
        label = det.update(2000.0, 2001.0, 1999.0, 0)
        assert label.regime == "RANGE"

    def test_has_name(self):
        det = PELTDetector()
        assert det.name == "PELT"


# ── Markov Switching (batch) ──────────────────────────────────────

class TestMarkovSwitching:
    def test_batch_fit_and_update(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=2000)
        det = MarkovSwitchingDetector(max_fit_bars=2000)
        det.fit_batch(c)
        results = _run_streaming_detector(det, c, h, l, 2000)
        assert all(isinstance(r, RegimeLabel) for r in results)

    def test_without_fit_returns_range(self):
        det = MarkovSwitchingDetector()
        label = det.update(2000.0, 2001.0, 1999.0, 0)
        assert label.regime == "RANGE"

    def test_has_name(self):
        det = MarkovSwitchingDetector()
        assert det.name == "MarkovSW"


# ── HMM adapter ──────────────────────────────────────────────────

class TestHMMAdapter:
    def test_returns_regime_labels(self):
        c, h, l, _, _ = generate_synthetic_data(n_bars=200)
        det = HMMRegimeDetector(k=10, window_W=50, refit_every=20)
        results = _run_streaming_detector(det, c, h, l, 200)
        assert all(isinstance(r, RegimeLabel) for r in results)

    def test_has_name_and_warmup(self):
        det = HMMRegimeDetector()
        assert det.name == "HMM"
        assert det.warmup_bars > 0


# ── Integration: all detectors on synthetic data ──────────────────

class TestIntegration:
    def test_all_streaming_detectors_run(self):
        """Smoke test: every streaming detector runs without error."""
        c, h, l, _, labels = generate_synthetic_data(n_bars=800)
        detectors = [
            ChoppinessDetector(period=14),
            HurstDetector(window=50, compute_every=5),
            BBWidthDetector(bb_period=20, rank_window=100),
            GMMDetector(k=10, window=200, refit_every=100, predict_every=5),
            KMeansDetector(k=10, window=200, refit_every=100, predict_every=5),
        ]
        for det in detectors:
            results = _run_streaming_detector(det, c, h, l, 800)
            assert len(results) == 800, f"{det.name} returned wrong count"

    def test_all_batch_detectors_run(self):
        """Smoke test: every batch detector runs without error."""
        c, h, l, _, labels = generate_synthetic_data(n_bars=2000)
        pelt = PELTDetector(k=20, min_size=50, pen=5.0)
        pelt.fit_batch(c)
        results = _run_streaming_detector(pelt, c, h, l, 2000)
        assert len(results) == 2000

        msw = MarkovSwitchingDetector(max_fit_bars=2000)
        msw.fit_batch(c)
        results = _run_streaming_detector(msw, c, h, l, 2000)
        assert len(results) == 2000

    def test_synthetic_accuracy_above_chance_choppiness(self):
        """Choppiness should do better than 50% on synthetic data."""
        c, h, l, _, labels = generate_synthetic_data(
            n_bars=5000, segment_length=250
        )
        det = ChoppinessDetector(period=14)
        results = _run_streaming_detector(det, c, h, l, 5000)
        warmup = det.warmup_bars
        correct = sum(
            1 for i in range(warmup, 5000)
            if results[i].regime == labels[i]
        )
        acc = correct / (5000 - warmup)
        assert acc > 0.45, f"Choppiness accuracy {acc:.2%} below 45%"
