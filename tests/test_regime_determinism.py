"""Tests for HMM regime filter determinism."""

import numpy as np
import pytest

from src.regime.hmm_model import HMMTrendRangeModel


class TestHMMDeterminism:
    @pytest.fixture
    def synthetic_data(self):
        """Generate standardized 3-feature data resembling (ER, vol, mom)."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((500, 3))
        return X

    def test_fit_predict_deterministic(self, synthetic_data):
        """Two models with same seed must produce identical posteriors."""
        m1 = HMMTrendRangeModel(covariance_type="full", n_iter=100, random_state=42)
        m2 = HMMTrendRangeModel(covariance_type="full", n_iter=100, random_state=42)

        m1.fit(synthetic_data)
        m2.fit(synthetic_data)

        p1 = m1.predict_proba(synthetic_data)
        p2 = m2.predict_proba(synthetic_data)

        np.testing.assert_array_equal(p1, p2)

    def test_different_seed_different_result(self, synthetic_data):
        """Different seeds should (very likely) produce different posteriors."""
        m1 = HMMTrendRangeModel(covariance_type="full", n_iter=100, random_state=42)
        m2 = HMMTrendRangeModel(covariance_type="full", n_iter=100, random_state=99)

        m1.fit(synthetic_data)
        m2.fit(synthetic_data)

        p1 = m1.predict_proba(synthetic_data)
        p2 = m2.predict_proba(synthetic_data)

        # They should differ in at least some rows
        assert not np.allclose(p1, p2), "Different seeds produced identical results"

    def test_get_trend_state_consistent(self, synthetic_data):
        """trend_state should be the same across repeated fits with same seed."""
        m1 = HMMTrendRangeModel(covariance_type="full", n_iter=100, random_state=42)
        m2 = HMMTrendRangeModel(covariance_type="full", n_iter=100, random_state=42)

        m1.fit(synthetic_data)
        m2.fit(synthetic_data)

        feature_names = ["ER", "vol", "mom"]
        assert m1.get_trend_state(feature_names) == m2.get_trend_state(feature_names)

    def test_predict_before_fit_raises(self):
        """predict_proba before fit should raise RuntimeError."""
        m = HMMTrendRangeModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            m.predict_proba(np.zeros((10, 3)))
