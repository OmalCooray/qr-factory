"""Regime feature computation from close prices."""

from __future__ import annotations

import numpy as np


def compute_regime_features(closes: np.ndarray, k: int) -> tuple[float, float, float]:
    """Compute trendiness features from the last k+1 close prices.

    Parameters
    ----------
    closes : np.ndarray
        Array of at least k+1 close prices.
    k : int
        Number of log-returns to use.

    Returns
    -------
    tuple of (ER, vol, mom)
        ER  — efficiency ratio: |sum(returns)| / sum(|returns|), in [0, 1]
        vol — mean absolute return (noise proxy)
        mom — sum of returns (path direction)
    """
    if len(closes) < k + 1:
        raise ValueError(f"Need at least {k + 1} closes, got {len(closes)}")

    prices = closes[-(k + 1) :]
    returns = np.diff(np.log(prices))

    mom = float(np.sum(returns))
    abs_returns = np.abs(returns)
    vol = float(np.mean(abs_returns))
    denom = float(np.sum(abs_returns))
    er = abs(mom) / denom if denom > 0 else 0.0

    return er, vol, mom
