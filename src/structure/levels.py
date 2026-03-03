"""Structural level detection — ICT-style SSL/BSL, FVGs, and Order Blocks.

All detection functions are pure: they operate on lists of floats and return
``Level`` instances.  No pandas or DataFrame dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class LevelType(Enum):
    SSL = "SSL"
    BSL = "BSL"
    FVG_BULL = "FVG_BULL"
    FVG_BEAR = "FVG_BEAR"
    OB_BULL = "OB_BULL"
    OB_BEAR = "OB_BEAR"


@dataclass(frozen=True)
class Level:
    """A structural price level detected on the chart.

    ``price_low`` / ``price_high`` define the zone (may be equal for a single
    price point like a swing high/low).  ``swept`` is ``True`` once price has
    traded through the level.
    """

    bar_index: int
    level_type: LevelType
    price_low: float
    price_high: float
    strength: float = 1.0
    swept: bool = False

    def mark_swept(self) -> Level:
        """Return a copy with ``swept=True``."""
        return Level(
            bar_index=self.bar_index,
            level_type=self.level_type,
            price_low=self.price_low,
            price_high=self.price_high,
            strength=self.strength,
            swept=True,
        )


# ---------------------------------------------------------------------------
# Swing detection
# ---------------------------------------------------------------------------

def detect_swing_highs(highs: list[float], left: int, right: int) -> list[int]:
    """Return bar indices that are swing highs.

    A swing high at index *i* requires ``highs[i]`` to be strictly greater
    than all *left* bars before and all *right* bars after.
    """
    result: list[int] = []
    n = len(highs)
    for i in range(left, n - right):
        is_swing = True
        for j in range(i - left, i):
            if highs[j] >= highs[i]:
                is_swing = False
                break
        if is_swing:
            for j in range(i + 1, i + right + 1):
                if highs[j] >= highs[i]:
                    is_swing = False
                    break
        if is_swing:
            result.append(i)
    return result


def detect_swing_lows(lows: list[float], left: int, right: int) -> list[int]:
    """Return bar indices that are swing lows.

    A swing low at index *i* requires ``lows[i]`` to be strictly less than
    all *left* bars before and all *right* bars after.
    """
    result: list[int] = []
    n = len(lows)
    for i in range(left, n - right):
        is_swing = True
        for j in range(i - left, i):
            if lows[j] <= lows[i]:
                is_swing = False
                break
        if is_swing:
            for j in range(i + 1, i + right + 1):
                if lows[j] <= lows[i]:
                    is_swing = False
                    break
        if is_swing:
            result.append(i)
    return result


# ---------------------------------------------------------------------------
# BSL / SSL (buy-side / sell-side liquidity)
# ---------------------------------------------------------------------------

def detect_bsl(
    highs: list[float],
    bar_offset: int,
    left: int,
    right: int,
) -> list[Level]:
    """Detect buy-side liquidity (BSL) levels from swing highs."""
    indices = detect_swing_highs(highs, left, right)
    return [
        Level(
            bar_index=bar_offset + i,
            level_type=LevelType.BSL,
            price_low=highs[i],
            price_high=highs[i],
        )
        for i in indices
    ]


def detect_ssl(
    lows: list[float],
    bar_offset: int,
    left: int,
    right: int,
) -> list[Level]:
    """Detect sell-side liquidity (SSL) levels from swing lows."""
    indices = detect_swing_lows(lows, left, right)
    return [
        Level(
            bar_index=bar_offset + i,
            level_type=LevelType.SSL,
            price_low=lows[i],
            price_high=lows[i],
        )
        for i in indices
    ]


# ---------------------------------------------------------------------------
# Fair-Value Gaps (FVGs)
# ---------------------------------------------------------------------------

def detect_fvg(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    bar_offset: int,
    min_gap_atr_mult: float,
    atr_value: float,
) -> list[Level]:
    """Detect 3-candle Fair-Value Gaps.

    Bullish FVG: bar[i+2].low > bar[i].high  (gap up)
    Bearish FVG: bar[i].low > bar[i+2].high  (gap down)

    Only gaps wider than ``min_gap_atr_mult * atr_value`` are kept.
    """
    result: list[Level] = []
    n = len(highs)
    if n < 3:
        return result

    min_gap = min_gap_atr_mult * atr_value

    for i in range(n - 2):
        # Bullish FVG: gap between bar[i].high and bar[i+2].low
        gap_bull = lows[i + 2] - highs[i]
        if gap_bull > 0 and gap_bull >= min_gap:
            result.append(Level(
                bar_index=bar_offset + i + 1,
                level_type=LevelType.FVG_BULL,
                price_low=highs[i],
                price_high=lows[i + 2],
                strength=gap_bull / atr_value if atr_value > 0 else 1.0,
            ))

        # Bearish FVG: gap between bar[i+2].high and bar[i].low
        gap_bear = lows[i] - highs[i + 2]
        if gap_bear > 0 and gap_bear >= min_gap:
            result.append(Level(
                bar_index=bar_offset + i + 1,
                level_type=LevelType.FVG_BEAR,
                price_low=highs[i + 2],
                price_high=lows[i],
                strength=gap_bear / atr_value if atr_value > 0 else 1.0,
            ))

    return result


# ---------------------------------------------------------------------------
# Order Blocks (OBs)
# ---------------------------------------------------------------------------

def detect_ob(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    bar_offset: int,
    min_impulse_atr_mult: float,
    atr_value: float,
) -> list[Level]:
    """Detect Order Blocks — last opposite candle before an impulsive move.

    Bullish OB: bearish candle (close < open) followed by a bullish impulse
    (next candle's close - current candle's low >= min_impulse_atr_mult * ATR).

    Bearish OB: bullish candle (close > open) followed by a bearish impulse
    (current candle's high - next candle's close >= min_impulse_atr_mult * ATR).
    """
    result: list[Level] = []
    n = len(opens)
    if n < 2:
        return result

    min_impulse = min_impulse_atr_mult * atr_value

    for i in range(n - 1):
        is_bearish_candle = closes[i] < opens[i]
        is_bullish_candle = closes[i] > opens[i]

        # Bullish OB: bearish candle followed by bullish impulse
        if is_bearish_candle:
            impulse = closes[i + 1] - lows[i]
            if impulse >= min_impulse:
                result.append(Level(
                    bar_index=bar_offset + i,
                    level_type=LevelType.OB_BULL,
                    price_low=lows[i],
                    price_high=highs[i],
                    strength=impulse / atr_value if atr_value > 0 else 1.0,
                ))

        # Bearish OB: bullish candle followed by bearish impulse
        if is_bullish_candle:
            impulse = highs[i] - closes[i + 1]
            if impulse >= min_impulse:
                result.append(Level(
                    bar_index=bar_offset + i,
                    level_type=LevelType.OB_BEAR,
                    price_low=lows[i],
                    price_high=highs[i],
                    strength=impulse / atr_value if atr_value > 0 else 1.0,
                ))

    return result
