"""Tests for regime feature computation."""

import numpy as np
import pytest

from src.regime.features import compute_regime_features


class TestComputeRegimeFeatures:
    def test_er_in_range(self):
        """ER must be in [0, 1]."""
        rng = np.random.default_rng(42)
        prices = 100.0 + np.cumsum(rng.standard_normal(200))
        prices = np.abs(prices) + 1.0  # keep positive for log

        for k in [5, 10, 20, 50]:
            er, vol, mom = compute_regime_features(prices, k)
            assert 0.0 <= er <= 1.0, f"ER={er} out of [0,1] for k={k}"

    def test_pure_trend_er_one(self):
        """Monotonically increasing prices should give ER == 1.0."""
        prices = np.arange(1.0, 22.0)  # 21 prices, k=20
        er, vol, mom = compute_regime_features(prices, 20)
        assert er == pytest.approx(1.0, abs=1e-10)
        assert mom > 0.0

    def test_zero_denom_guard(self):
        """Flat prices (zero returns) should give ER == 0, not raise."""
        prices = np.full(25, 100.0)
        er, vol, mom = compute_regime_features(prices, 20)
        assert er == 0.0
        assert vol == 0.0
        assert mom == 0.0

    def test_no_nans_beyond_warmup(self):
        """Features should not contain NaN for valid input."""
        rng = np.random.default_rng(123)
        prices = 100.0 + np.cumsum(rng.standard_normal(100))
        prices = np.abs(prices) + 1.0

        er, vol, mom = compute_regime_features(prices, 20)
        assert not np.isnan(er)
        assert not np.isnan(vol)
        assert not np.isnan(mom)

    def test_insufficient_data_raises(self):
        """Should raise ValueError when not enough closes."""
        prices = np.array([100.0, 101.0])
        with pytest.raises(ValueError, match="Need at least"):
            compute_regime_features(prices, 5)

    def test_vol_positive_for_varying_prices(self):
        """Vol should be > 0 when prices vary."""
        rng = np.random.default_rng(7)
        prices = 100.0 + np.cumsum(rng.standard_normal(50))
        prices = np.abs(prices) + 1.0
        er, vol, mom = compute_regime_features(prices, 20)
        assert vol > 0.0
