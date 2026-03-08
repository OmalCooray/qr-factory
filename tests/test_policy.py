"""Tests for FixedThresholdPolicy and ProbabilitySizedPolicy."""

import pandas as pd
import pytest

from src.policy import FixedThresholdPolicy, Policy, ProbabilitySizedPolicy
from src.strategy.base import StrategyContext


# ── Helpers ──────────────────────────────────────────────────────────────────


def _ctx(position: float = 0.0, yhat: float = 0.5) -> StrategyContext:
    """Build a minimal StrategyContext."""
    return StrategyContext(
        ts=pd.Timestamp("2024-01-01"),
        bar=pd.Series(
            {"open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
        ),
        features=pd.Series({"yhat": yhat}),
        position=position,
        equity=10_000.0,
        bar_index=100,
    )


# ── Protocol conformance ────────────────────────────────────────────────────


class TestProtocol:

    def test_isinstance_check(self):
        """FixedThresholdPolicy satisfies the Policy protocol."""
        policy = FixedThresholdPolicy()
        assert isinstance(policy, Policy)


# ── Enter / flat transitions ────────────────────────────────────────────────


class TestEnterFlat:

    def test_enter_above_p_enter(self):
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=0.45)
        sig = policy.decide(_ctx(position=0.0), yhat=0.60)
        assert sig.direction == 1.0
        assert sig.reason == "enter yhat=0.600>=p_enter"

    def test_flat_below_p_enter(self):
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=0.45)
        sig = policy.decide(_ctx(position=0.0), yhat=0.50)
        assert sig.direction == 0.0
        assert sig.reason == "flat yhat=0.500<p_enter"


# ── Hold / exit transitions ─────────────────────────────────────────────────


class TestHoldExit:

    def test_hold_in_dead_band(self):
        """When long, yhat in [p_exit, p_enter) → hold."""
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=0.45)
        sig = policy.decide(_ctx(position=1.0), yhat=0.50)
        assert sig.direction == 1.0
        assert sig.reason == "hold yhat=0.500 bars=1"

    def test_exit_below_p_exit(self):
        """When long, yhat < p_exit → exit."""
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=0.45)
        # First bar as long → bars_held=1
        policy.decide(_ctx(position=1.0), yhat=0.60)
        # Second bar: yhat drops below p_exit, bars_held=2 >= 0
        sig = policy.decide(_ctx(position=1.0), yhat=0.40)
        assert sig.direction == 0.0
        assert sig.reason == "exit yhat=0.400<p_exit=0.450"


# ── Min hold bars ───────────────────────────────────────────────────────────


class TestMinHoldBars:

    def test_prevents_early_exit(self):
        """Must hold for min_hold_bars even if yhat < p_exit."""
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=0.45, min_hold_bars=3)
        # Enter
        policy.decide(_ctx(position=0.0), yhat=0.60)
        # Bar 1 as long: yhat below p_exit but bars_held=1 < 3
        sig = policy.decide(_ctx(position=1.0), yhat=0.30)
        assert sig.direction == 1.0, "Must hold — bars_held < min_hold_bars"

    def test_allows_exit_after_enough_bars(self):
        """After holding min_hold_bars, exit when yhat < p_exit."""
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=0.45, min_hold_bars=2)
        # Enter
        policy.decide(_ctx(position=0.0), yhat=0.60)
        # Bar 1 as long
        policy.decide(_ctx(position=1.0), yhat=0.50)
        # Bar 2 as long — bars_held=2 >= min_hold_bars=2
        sig = policy.decide(_ctx(position=1.0), yhat=0.30)
        assert sig.direction == 0.0, "Should exit — held enough bars"


# ── p_exit fallback ─────────────────────────────────────────────────────────


class TestPExitFallback:

    def test_none_p_exit_uses_p_enter(self):
        """p_exit=None → effective_p_exit = p_enter."""
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=None)
        # Enter
        policy.decide(_ctx(position=0.0), yhat=0.56)
        # Long, yhat=0.54 < p_enter (=effective_p_exit) → exit
        sig = policy.decide(_ctx(position=1.0), yhat=0.54)
        assert sig.direction == 0.0


# ── Reset ────────────────────────────────────────────────────────────────────


class TestReset:

    def test_reset_clears_bars_held(self):
        """reset() zeroes the bar counter, preventing stale state."""
        policy = FixedThresholdPolicy(p_enter=0.55, p_exit=0.45, min_hold_bars=3)
        # Enter and hold
        policy.decide(_ctx(position=0.0), yhat=0.60)
        policy.decide(_ctx(position=1.0), yhat=0.50)
        assert policy._bars_held == 2
        # Reset
        policy.reset()
        assert policy._bars_held == 0


# ══════════════════════════════════════════════════════════════════════════════
# ProbabilitySizedPolicy tests
# ══════════════════════════════════════════════════════════════════════════════


class TestProbSizedProtocol:

    def test_isinstance_check(self):
        """ProbabilitySizedPolicy satisfies the Policy protocol."""
        policy = ProbabilitySizedPolicy()
        assert isinstance(policy, Policy)


class TestProbSizedFlat:

    def test_below_p_enter_stays_flat(self):
        policy = ProbabilitySizedPolicy(p_enter=0.55, p_exit=0.45)
        sig = policy.decide(_ctx(position=0.0), yhat=0.50)
        assert sig.direction == 0.0
        assert sig.strength == 0.0
        assert "flat" in sig.reason

    def test_at_p_enter_enters(self):
        policy = ProbabilitySizedPolicy(p_enter=0.55, p_exit=0.45, p_full=0.65)
        sig = policy.decide(_ctx(position=0.0), yhat=0.55)
        assert sig.direction == 1.0
        # At p_enter, confidence=0 → strength=min_size
        assert sig.strength == pytest.approx(policy.min_size)


class TestProbSizedEnter:

    def test_enter_at_min_size(self):
        """At p_enter exactly → min_size."""
        policy = ProbabilitySizedPolicy(
            p_enter=0.55, p_full=0.65, min_size=0.25, max_size=1.0,
        )
        sig = policy.decide(_ctx(position=0.0), yhat=0.55)
        assert sig.strength == pytest.approx(0.25)

    def test_enter_at_max_size(self):
        """At p_full → max_size."""
        policy = ProbabilitySizedPolicy(
            p_enter=0.55, p_full=0.65, min_size=0.25, max_size=1.0,
        )
        sig = policy.decide(_ctx(position=0.0), yhat=0.65)
        assert sig.strength == pytest.approx(1.0)

    def test_enter_above_p_full_capped(self):
        """Above p_full → still max_size (clipped)."""
        policy = ProbabilitySizedPolicy(
            p_enter=0.55, p_full=0.65, min_size=0.25, max_size=1.0,
        )
        sig = policy.decide(_ctx(position=0.0), yhat=0.80)
        assert sig.strength == pytest.approx(1.0)

    def test_enter_interpolated_midpoint(self):
        """Midpoint between p_enter and p_full → halfway between min/max."""
        policy = ProbabilitySizedPolicy(
            p_enter=0.55, p_full=0.65, min_size=0.25, max_size=1.0,
        )
        sig = policy.decide(_ctx(position=0.0), yhat=0.60)
        # confidence = (0.60 - 0.55) / (0.65 - 0.55) = 0.5
        # strength = 0.25 + 0.5 * (1.0 - 0.25) = 0.625
        assert sig.strength == pytest.approx(0.625)


class TestProbSizedHoldExit:

    def test_hold_in_dead_band(self):
        """When long, yhat in [p_exit, p_enter) → hold."""
        policy = ProbabilitySizedPolicy(p_enter=0.55, p_exit=0.45, p_full=0.65)
        sig = policy.decide(_ctx(position=1.0), yhat=0.50)
        assert sig.direction == 1.0
        assert "hold" in sig.reason

    def test_exit_below_p_exit(self):
        """When long, yhat < p_exit after min_hold → exit."""
        policy = ProbabilitySizedPolicy(p_enter=0.55, p_exit=0.45, p_full=0.65)
        # First bar as long → bars_held=1
        policy.decide(_ctx(position=1.0), yhat=0.60)
        # Second bar: yhat drops below p_exit, bars_held=2 >= 0
        sig = policy.decide(_ctx(position=1.0), yhat=0.40)
        assert sig.direction == 0.0
        assert "exit" in sig.reason

    def test_min_hold_prevents_early_exit(self):
        """Must hold for min_hold_bars even if yhat < p_exit."""
        policy = ProbabilitySizedPolicy(
            p_enter=0.55, p_exit=0.45, p_full=0.65, min_hold_bars=3,
        )
        # Enter
        policy.decide(_ctx(position=0.0), yhat=0.60)
        # Bar 1 as long: yhat below p_exit but bars_held=1 < 3
        sig = policy.decide(_ctx(position=1.0), yhat=0.30)
        assert sig.direction == 1.0, "Must hold — bars_held < min_hold_bars"

    def test_exit_after_min_hold(self):
        """After holding min_hold_bars, exit when yhat < p_exit."""
        policy = ProbabilitySizedPolicy(
            p_enter=0.55, p_exit=0.45, p_full=0.65, min_hold_bars=2,
        )
        policy.decide(_ctx(position=0.0), yhat=0.60)
        policy.decide(_ctx(position=1.0), yhat=0.50)
        sig = policy.decide(_ctx(position=1.0), yhat=0.30)
        assert sig.direction == 0.0, "Should exit — held enough bars"


class TestProbSizedReset:

    def test_reset_clears_bars_held(self):
        policy = ProbabilitySizedPolicy(p_enter=0.55, p_exit=0.45, min_hold_bars=3)
        policy.decide(_ctx(position=0.0), yhat=0.60)
        policy.decide(_ctx(position=1.0), yhat=0.50)
        assert policy._bars_held == 2
        policy.reset()
        assert policy._bars_held == 0


class TestProbSizedScaling:
    """Verify linear interpolation at several points."""

    @pytest.mark.parametrize(
        "yhat, expected_strength",
        [
            (0.55, 0.25),   # at p_enter → min_size
            (0.575, 0.4375),  # 25% confidence → 0.25 + 0.25*0.75
            (0.60, 0.625),  # 50% confidence
            (0.625, 0.8125),  # 75% confidence
            (0.65, 1.0),    # at p_full → max_size
            (0.70, 1.0),    # above p_full → clipped to max_size
        ],
    )
    def test_scaling_points(self, yhat: float, expected_strength: float):
        policy = ProbabilitySizedPolicy(
            p_enter=0.55, p_full=0.65, min_size=0.25, max_size=1.0,
        )
        sig = policy.decide(_ctx(position=0.0), yhat=yhat)
        assert sig.strength == pytest.approx(expected_strength)


# ══════════════════════════════════════════════════════════════════════════════
# Integration: policy selection from config via build()
# ══════════════════════════════════════════════════════════════════════════════


class TestPolicyConfigDispatch:
    """Verify build() constructs the correct policy type from config."""

    def test_fixed_threshold_from_config(self):
        from src.strategy.ml_prob_threshold import build

        params = {
            "policy": {
                "type": "fixed_threshold",
                "p_enter": 0.55,
                "p_exit": 0.45,
                "min_hold_bars": 5,
            },
        }
        strategy, _ = build(params)
        assert isinstance(strategy._policy, FixedThresholdPolicy)
        assert strategy._policy.p_enter == 0.55
        assert strategy._policy.p_exit == 0.45
        assert strategy._policy.min_hold_bars == 5

    def test_probability_sized_from_config(self):
        from src.strategy.ml_prob_threshold import build

        params = {
            "policy": {
                "type": "probability_sized",
                "p_enter": 0.55,
                "p_exit": 0.45,
                "p_full": 0.65,
                "min_size": 0.25,
                "max_size": 1.0,
                "min_hold_bars": 5,
            },
        }
        strategy, _ = build(params)
        assert isinstance(strategy._policy, ProbabilitySizedPolicy)
        assert strategy._policy.p_enter == 0.55
        assert strategy._policy.p_full == 0.65
        assert strategy._policy.min_size == 0.25

    def test_inline_fallback_no_policy_block(self):
        """No policy block → inline FixedThresholdPolicy via __post_init__."""
        from src.strategy.ml_prob_threshold import build

        params = {
            "p_enter": 0.60,
            "p_exit": 0.50,
            "min_hold_bars": 3,
        }
        strategy, _ = build(params)
        assert isinstance(strategy._policy, FixedThresholdPolicy)
        assert strategy._policy.p_enter == 0.60

    def test_unknown_policy_type_raises(self):
        from src.strategy.ml_prob_threshold import build

        params = {
            "policy": {"type": "unknown_policy"},
        }
        with pytest.raises(ValueError, match="Unknown policy type"):
            build(params)
