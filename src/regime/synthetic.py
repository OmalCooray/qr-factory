"""Synthetic price data with known trend/range ground-truth labels."""

from __future__ import annotations

import numpy as np


def generate_synthetic_data(
    n_bars: int = 5000,
    segment_length: int = 250,
    trend_drift: float = 0.0003,
    trend_noise: float = 0.001,
    range_drift: float = 0.0,
    range_noise: float = 0.003,
    start_price: float = 2000.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Generate synthetic OHLC-like data alternating TREND and RANGE segments.

    Returns
    -------
    close, high, low : np.ndarray, shape (n_bars,)
    bar_indices : np.ndarray, shape (n_bars,)
    labels : list[str]
        Ground-truth "TREND" or "RANGE" per bar.
    """
    rng = np.random.default_rng(seed)

    closes = np.empty(n_bars, dtype=np.float64)
    highs = np.empty(n_bars, dtype=np.float64)
    lows = np.empty(n_bars, dtype=np.float64)
    labels: list[str] = []
    price = start_price

    for i in range(n_bars):
        seg_idx = i // segment_length
        is_trend = seg_idx % 2 == 0

        if is_trend:
            drift = trend_drift * (1 if seg_idx % 4 == 0 else -1)
            noise = trend_noise
            labels.append("TREND")
        else:
            drift = range_drift
            noise = range_noise
            labels.append("RANGE")

        ret = drift + rng.normal(0, noise)
        price *= np.exp(ret)
        spread = abs(rng.normal(0, price * noise * 0.5))
        closes[i] = price
        highs[i] = price + spread
        lows[i] = price - spread

    return closes, highs, lows, np.arange(n_bars), labels
