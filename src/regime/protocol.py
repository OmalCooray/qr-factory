"""Common protocol and data types for regime detectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RegimeLabel:
    """Standardised regime output from any detector."""

    regime: str        # "TREND" or "RANGE"
    confidence: float  # 0.0–1.0
    raw_value: float   # underlying numeric (e.g., Hurst H, CHOP value, p_trend)


@runtime_checkable
class RegimeDetector(Protocol):
    """Interface every regime detector must satisfy."""

    name: str
    warmup_bars: int

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        """Consume one bar and return a regime label."""
        ...
