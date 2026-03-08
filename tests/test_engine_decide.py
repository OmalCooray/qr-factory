"""Tests for TradingEngine._decide() sizing logic."""

import pytest

from src.engine.core import TradingEngine
from src.execution.models import OrderIntent
from src.risk import RiskAction, RiskConfig, RiskManager
from src.strategy.signal import Signal


def _make_engine(position_size: float = 1.0) -> TradingEngine:
    """Build a minimal TradingEngine for testing _decide()."""

    class StubStrategy:
        name = "stub"
        warmup_bars = 0

        def on_bar(self, ctx):
            return Signal(direction=0.0, strength=0.0, reason="stub")

    return TradingEngine(
        strategy=StubStrategy(),
        risk_manager=RiskManager(config=RiskConfig(), starting_capital=10_000.0),
        starting_capital=10_000.0,
        position_size=position_size,
    )


_PASS = RiskAction(flatten=False, halted=False, reason="")
_FLATTEN = RiskAction(flatten=True, halted=True, reason="drawdown")


class TestDecideFixedPath:
    """strength=1.0 (FixedThresholdPolicy default) → full position_size."""

    def test_enter_long_full_size(self):
        engine = _make_engine(position_size=1.0)
        intent = engine._decide(
            Signal(direction=1.0, strength=1.0, reason="enter"),
            _PASS,
        )
        assert intent.target_position == pytest.approx(1.0)

    def test_exit_zero_regardless_of_strength(self):
        engine = _make_engine(position_size=1.0)
        intent = engine._decide(
            Signal(direction=0.0, strength=1.0, reason="exit"),
            _PASS,
        )
        assert intent.target_position == pytest.approx(0.0)


class TestDecideFractionalPath:
    """strength in (0, 1) → fractional sizing via ProbabilitySizedPolicy."""

    def test_half_strength_half_size(self):
        engine = _make_engine(position_size=1.0)
        intent = engine._decide(
            Signal(direction=1.0, strength=0.5, reason="enter"),
            _PASS,
        )
        assert intent.target_position == pytest.approx(0.5)

    def test_quarter_strength(self):
        engine = _make_engine(position_size=1.0)
        intent = engine._decide(
            Signal(direction=1.0, strength=0.25, reason="enter"),
            _PASS,
        )
        assert intent.target_position == pytest.approx(0.25)

    def test_zero_direction_zero_target(self):
        """direction=0 → target=0, regardless of strength."""
        engine = _make_engine(position_size=1.0)
        intent = engine._decide(
            Signal(direction=0.0, strength=0.5, reason="flat"),
            _PASS,
        )
        assert intent.target_position == pytest.approx(0.0)


class TestDecideDynamicOverride:
    """strength > position_size → strategy computed its own lot size."""

    def test_strength_above_position_size(self):
        engine = _make_engine(position_size=1.0)
        intent = engine._decide(
            Signal(direction=1.0, strength=2.5, reason="breakout"),
            _PASS,
        )
        assert intent.target_position == pytest.approx(2.5)


class TestDecideRiskFlatten:
    """Risk flatten overrides all signals."""

    def test_flatten_overrides_long_signal(self):
        engine = _make_engine(position_size=1.0)
        intent = engine._decide(
            Signal(direction=1.0, strength=1.0, reason="enter"),
            _FLATTEN,
        )
        assert intent.target_position == pytest.approx(0.0)
        assert "risk:" in intent.reason
