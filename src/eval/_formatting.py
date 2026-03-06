"""Shared formatting helpers for eval reports and comparisons."""

from __future__ import annotations

import numpy as np


def fmt(value, decimals: int = 4) -> str:
    """Format a metric value for display. Handles None and NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)
