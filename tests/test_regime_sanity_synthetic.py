"""Sanity test: trending segment should have higher avg p_trend than ranging."""

import numpy as np
import pytest

from src.regime import TrendRangeFilter


class TestRegimeSanitySynthetic:
    def test_trend_vs_range_segments(self):
        """Generate a trending segment followed by a ranging segment.

        The filter should assign higher average p_trend to the trending part
        (after warmup).
        """
        rng = np.random.default_rng(42)

        # Trending segment: strong upward drift + small noise
        n_trend = 6000
        trend_returns = 0.001 + rng.standard_normal(n_trend) * 0.0005
        trend_prices = 100.0 * np.exp(np.cumsum(trend_returns))

        # Ranging segment: zero drift + larger noise
        n_range = 3000
        range_returns = rng.standard_normal(n_range) * 0.002
        range_prices = trend_prices[-1] * np.exp(np.cumsum(range_returns))

        all_prices = np.concatenate([trend_prices, range_prices])

        filt = TrendRangeFilter(
            k=20,
            window_W=5000,
            refit_every=100,
            predict_every=1,
            enter_th=0.60,
            exit_th=0.40,
            random_state=42,
        )

        p_trends = []
        for price in all_prices:
            info = filt.update(float(price))
            p_trends.append(info.p_trend)

        p_trends = np.array(p_trends)

        # Compare: average p_trend in trending segment (after warmup) vs ranging segment
        # Warmup is k + window_W bars before HMM produces meaningful output
        warmup_end = 5100  # slightly after window_W + k
        trend_end = n_trend

        avg_p_trend_trending = p_trends[warmup_end:trend_end].mean()
        avg_p_trend_ranging = p_trends[trend_end:].mean()

        assert avg_p_trend_trending > avg_p_trend_ranging, (
            f"Trending segment avg p_trend ({avg_p_trend_trending:.3f}) "
            f"should exceed ranging segment ({avg_p_trend_ranging:.3f})"
        )

    def test_warmup_returns_range(self):
        """During warmup, filter should always return RANGE."""
        filt = TrendRangeFilter(k=20, window_W=5000)

        # Feed fewer bars than k+1
        for i in range(15):
            info = filt.update(100.0 + i * 0.1)
            assert info.regime == "RANGE"
            assert info.p_trend == 0.0
