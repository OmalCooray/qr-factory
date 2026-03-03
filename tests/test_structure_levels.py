"""Tests for structural level detection (swing, BSL/SSL, FVG, OB)."""

import pytest

from src.structure.levels import (
    Level,
    LevelType,
    detect_bsl,
    detect_fvg,
    detect_ob,
    detect_ssl,
    detect_swing_highs,
    detect_swing_lows,
)


class TestSwingHighs:
    def test_single_peak(self):
        highs = [1.0, 2.0, 3.0, 2.0, 1.0]
        result = detect_swing_highs(highs, left=2, right=2)
        assert result == [2]

    def test_two_peaks(self):
        highs = [1.0, 3.0, 1.0, 1.0, 4.0, 1.0]
        result = detect_swing_highs(highs, left=1, right=1)
        assert result == [1, 4]

    def test_monotonic_up_no_swings(self):
        highs = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = detect_swing_highs(highs, left=1, right=1)
        assert result == []

    def test_flat_no_swings(self):
        highs = [2.0, 2.0, 2.0, 2.0, 2.0]
        result = detect_swing_highs(highs, left=1, right=1)
        assert result == []

    def test_left_right_3(self):
        highs = [1.0, 2.0, 3.0, 5.0, 3.0, 2.0, 1.0]
        result = detect_swing_highs(highs, left=3, right=3)
        assert result == [3]


class TestSwingLows:
    def test_single_trough(self):
        lows = [5.0, 3.0, 1.0, 3.0, 5.0]
        result = detect_swing_lows(lows, left=2, right=2)
        assert result == [2]

    def test_two_troughs(self):
        lows = [5.0, 2.0, 5.0, 5.0, 1.0, 5.0]
        result = detect_swing_lows(lows, left=1, right=1)
        assert result == [1, 4]

    def test_monotonic_down_no_swings(self):
        lows = [5.0, 4.0, 3.0, 2.0, 1.0]
        result = detect_swing_lows(lows, left=1, right=1)
        assert result == []

    def test_flat_no_swings(self):
        lows = [3.0, 3.0, 3.0, 3.0, 3.0]
        result = detect_swing_lows(lows, left=1, right=1)
        assert result == []


class TestBSL:
    def test_basic(self):
        highs = [1.0, 3.0, 1.0]
        levels = detect_bsl(highs, bar_offset=100, left=1, right=1)
        assert len(levels) == 1
        lv = levels[0]
        assert lv.level_type == LevelType.BSL
        assert lv.bar_index == 101  # bar_offset + 1
        assert lv.price_high == 3.0

    def test_with_offset(self):
        highs = [1.0, 5.0, 1.0, 1.0, 4.0, 1.0]
        levels = detect_bsl(highs, bar_offset=50, left=1, right=1)
        assert len(levels) == 2
        assert levels[0].bar_index == 51
        assert levels[1].bar_index == 54


class TestSSL:
    def test_basic(self):
        lows = [5.0, 2.0, 5.0]
        levels = detect_ssl(lows, bar_offset=200, left=1, right=1)
        assert len(levels) == 1
        lv = levels[0]
        assert lv.level_type == LevelType.SSL
        assert lv.bar_index == 201
        assert lv.price_low == 2.0


class TestFVG:
    def test_bullish_fvg(self):
        # bar0: H=10, bar1: any, bar2: L=12 → gap = 12-10 = 2
        opens = [9.0, 10.5, 11.5]
        highs = [10.0, 11.0, 13.0]
        lows = [8.0, 10.0, 12.0]
        closes = [9.5, 10.8, 12.5]
        levels = detect_fvg(opens, highs, lows, closes,
                            bar_offset=0, min_gap_atr_mult=0.5, atr_value=1.0)
        bull = [lv for lv in levels if lv.level_type == LevelType.FVG_BULL]
        assert len(bull) == 1
        assert bull[0].price_low == 10.0  # highs[0]
        assert bull[0].price_high == 12.0  # lows[2]
        assert bull[0].bar_index == 1  # middle candle

    def test_bearish_fvg(self):
        # bar0: L=12, bar1: any, bar2: H=10 → gap = 12-10 = 2
        opens = [13.0, 11.5, 9.5]
        highs = [14.0, 12.0, 10.0]
        lows = [12.0, 10.5, 8.0]
        closes = [12.5, 11.0, 9.0]
        levels = detect_fvg(opens, highs, lows, closes,
                            bar_offset=0, min_gap_atr_mult=0.5, atr_value=1.0)
        bear = [lv for lv in levels if lv.level_type == LevelType.FVG_BEAR]
        assert len(bear) == 1
        assert bear[0].price_low == 10.0  # highs[2]
        assert bear[0].price_high == 12.0  # lows[0]

    def test_no_gap_too_small(self):
        opens = [9.0, 10.5, 11.5]
        highs = [10.0, 11.0, 13.0]
        lows = [8.0, 10.0, 10.2]  # gap = 10.2-10.0 = 0.2 < 0.5*1.0
        closes = [9.5, 10.8, 12.5]
        levels = detect_fvg(opens, highs, lows, closes,
                            bar_offset=0, min_gap_atr_mult=0.5, atr_value=1.0)
        assert len(levels) == 0

    def test_no_gap_at_all(self):
        opens = [9.0, 10.5, 11.5]
        highs = [10.5, 11.0, 13.0]
        lows = [8.0, 10.0, 10.0]  # no gap: lows[2]=10.0 == highs[0]=10.5 → negative
        closes = [9.5, 10.8, 12.5]
        levels = detect_fvg(opens, highs, lows, closes,
                            bar_offset=0, min_gap_atr_mult=0.0, atr_value=1.0)
        bull = [lv for lv in levels if lv.level_type == LevelType.FVG_BULL]
        assert len(bull) == 0

    def test_fewer_than_3_bars(self):
        levels = detect_fvg([1.0], [2.0], [0.5], [1.5],
                            bar_offset=0, min_gap_atr_mult=0.0, atr_value=1.0)
        assert levels == []


class TestOB:
    def test_bullish_ob(self):
        # Bar 0: bearish candle (close < open), bar 1: bullish impulse
        opens = [12.0, 10.0]
        highs = [12.5, 14.0]
        lows = [9.0, 9.5]
        closes = [10.0, 13.0]  # impulse = 13.0 - 9.0 = 4.0 >= 1.5*1.0
        levels = detect_ob(opens, highs, lows, closes,
                           bar_offset=0, min_impulse_atr_mult=1.5, atr_value=1.0)
        bull = [lv for lv in levels if lv.level_type == LevelType.OB_BULL]
        assert len(bull) == 1
        assert bull[0].price_low == 9.0
        assert bull[0].price_high == 12.5
        assert bull[0].bar_index == 0

    def test_bearish_ob(self):
        # Bar 0: bullish candle (close > open), bar 1: bearish impulse
        opens = [10.0, 13.0]
        highs = [13.0, 13.5]
        lows = [9.5, 8.0]
        closes = [12.0, 9.0]  # impulse = 13.0 - 9.0 = 4.0 >= 1.5*1.0
        levels = detect_ob(opens, highs, lows, closes,
                           bar_offset=0, min_impulse_atr_mult=1.5, atr_value=1.0)
        bear = [lv for lv in levels if lv.level_type == LevelType.OB_BEAR]
        assert len(bear) == 1
        assert bear[0].price_low == 9.5
        assert bear[0].price_high == 13.0
        assert bear[0].bar_index == 0

    def test_no_impulse_too_small(self):
        opens = [12.0, 10.0]
        highs = [12.5, 11.0]
        lows = [9.0, 9.5]
        closes = [10.0, 10.2]  # impulse = 10.2 - 9.0 = 1.2 < 1.5*1.0
        levels = detect_ob(opens, highs, lows, closes,
                           bar_offset=0, min_impulse_atr_mult=1.5, atr_value=1.0)
        assert len(levels) == 0

    def test_single_bar(self):
        levels = detect_ob([10.0], [12.0], [9.0], [11.0],
                           bar_offset=0, min_impulse_atr_mult=0.0, atr_value=1.0)
        assert levels == []


class TestLevelFrozen:
    def test_frozen(self):
        lv = Level(bar_index=0, level_type=LevelType.BSL,
                   price_low=100.0, price_high=100.0)
        with pytest.raises(AttributeError):
            lv.swept = True  # type: ignore[misc]

    def test_mark_swept(self):
        lv = Level(bar_index=5, level_type=LevelType.SSL,
                   price_low=90.0, price_high=90.0, strength=2.0)
        swept = lv.mark_swept()
        assert swept.swept is True
        assert swept.bar_index == 5
        assert swept.strength == 2.0
        assert lv.swept is False  # original unchanged
