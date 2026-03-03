"""Graph-based Market Structure Strategy.

Detects ICT-style structural levels (SSL/BSL, FVGs, Order Blocks), builds a
directed graph of causal relationships, collapses SCCs via Kosaraju's
algorithm, and derives regime (RANGE vs TREND) + directional bias for trade
signals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.indicators.core.pipeline import FeatureSpec
from src.structure import (
    StructureGraph,
    detect_bsl,
    detect_fvg,
    detect_ob,
    detect_ssl,
)
from src.structure.levels import Level, LevelType
from src.strategy.signal import Signal
from .base import StrategyContext, validate_features


@dataclass
class GraphMarketStructureStrategy:
    """ICT market-structure strategy driven by a structure graph.

    Entry: regime == TREND, bias_strength >= entry_threshold, and confluence
    (FVG or OB near price in the bias direction).

    Exit: ATR stop loss, ATR take profit, max hold bars, or bias flip.
    """

    atr_col: str
    swing_lookback: int = 3
    fvg_min_gap_atr: float = 0.5
    ob_min_impulse_atr: float = 1.5
    max_levels: int = 200
    proximity_atr: float = 3.0
    entry_threshold: float = 0.3
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 3.0
    max_hold_bars: int = 50
    risk_per_trade_pct: float = 0.0  # 0 = use engine's fixed position_size
    detect_every: int = 1
    recompute_every: int = 5
    window_bars: int = 50
    _name: str = "graph_mss"
    _warmup_bars: int = 0

    # Rolling bar buffers
    _opens: list[float] = field(default_factory=list, init=False, repr=False)
    _highs: list[float] = field(default_factory=list, init=False, repr=False)
    _lows: list[float] = field(default_factory=list, init=False, repr=False)
    _closes: list[float] = field(default_factory=list, init=False, repr=False)

    # Graph
    _graph: StructureGraph = field(init=False, repr=False)

    # Position tracking
    _in_position_side: float = field(default=0.0, init=False, repr=False)
    _entry_price: float = field(default=0.0, init=False, repr=False)
    _stop_level: float = field(default=0.0, init=False, repr=False)
    _tp_level: float = field(default=0.0, init=False, repr=False)
    _entry_bar: int = field(default=0, init=False, repr=False)
    _last_bias: float = field(default=0.0, init=False, repr=False)

    # Detection counter
    _bar_count: int = field(default=0, init=False, repr=False)

    # Artifact records
    graph_state_records: list[dict] = field(default_factory=list, init=False, repr=False)
    level_records: list[dict] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._graph = StructureGraph(
            max_levels=self.max_levels,
            proximity_atr=self.proximity_atr,
            recompute_every=self.recompute_every,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> Sequence[str]:
        return [self.atr_col]

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        if ctx.bar_index < self.warmup_bars:
            return False
        if not validate_features(ctx.features, self.required_features):
            return False
        return True

    def on_bar(self, ctx: StrategyContext) -> Signal:
        # Append to rolling buffers
        self._opens.append(float(ctx.bar["open"]))
        self._highs.append(float(ctx.bar["high"]))
        self._lows.append(float(ctx.bar["low"]))
        self._closes.append(float(ctx.bar["close"]))

        # Trim to window_bars
        if len(self._opens) > self.window_bars:
            excess = len(self._opens) - self.window_bars
            self._opens = self._opens[excess:]
            self._highs = self._highs[excess:]
            self._lows = self._lows[excess:]
            self._closes = self._closes[excess:]

        self._bar_count += 1

        if not self.can_trade(ctx):
            return Signal(direction=0.0, strength=0.0, reason="warmup/feature_nan")

        close = float(ctx.bar["close"])
        atr = float(ctx.features[self.atr_col])

        # Sync position with engine
        if ctx.position == 0.0:
            self._in_position_side = 0.0
            self._stop_level = 0.0
            self._tp_level = 0.0
            self._entry_price = 0.0
            self._entry_bar = 0

        # Detect structural levels periodically
        if self._bar_count % self.detect_every == 0 and len(self._opens) >= 3:
            bar_offset = ctx.bar_index - len(self._opens) + 1
            new_levels = self._detect_levels(bar_offset, atr)
            self._graph.add_levels(new_levels)
            # Record levels for artifacts
            for lv in new_levels:
                self.level_records.append({
                    "ts": str(ctx.ts),
                    "bar_index": lv.bar_index,
                    "level_type": lv.level_type.value,
                    "price_low": lv.price_low,
                    "price_high": lv.price_high,
                    "strength": lv.strength,
                })

        # Update graph state
        graph_state = self._graph.update(close, atr, ctx.bar_index)
        self._last_bias = graph_state.bias

        # Record graph state for artifacts
        self.graph_state_records.append({
            "ts": str(ctx.ts),
            "bar_index": ctx.bar_index,
            "regime": graph_state.regime,
            "bias": graph_state.bias,
            "bias_strength": graph_state.bias_strength,
            "n_active": graph_state.n_active,
            "n_sccs": graph_state.n_sccs,
            "n_edges": graph_state.n_edges,
        })

        # --- Exit logic (if in position) ---
        if self._in_position_side != 0.0:
            return self._check_exit(ctx, close, atr, graph_state.bias)

        # --- Entry logic (if flat) ---
        if graph_state.regime != "TREND":
            return Signal(direction=0.0, strength=0.0,
                          reason=f"flat|regime={graph_state.regime}")

        if graph_state.bias_strength < self.entry_threshold:
            return Signal(direction=0.0, strength=0.0,
                          reason=f"flat|bias_weak={graph_state.bias_strength:.2f}")

        # Check confluence: FVG or OB near price in bias direction
        if not self._has_confluence(close, atr, graph_state.bias):
            return Signal(direction=0.0, strength=0.0,
                          reason="flat|no_confluence")

        # Enter position
        self._in_position_side = graph_state.bias
        self._entry_price = close
        self._entry_bar = ctx.bar_index

        if graph_state.bias > 0:
            self._stop_level = close - atr * self.sl_atr_mult
            self._tp_level = close + atr * self.tp_atr_mult
        else:
            self._stop_level = close + atr * self.sl_atr_mult
            self._tp_level = close - atr * self.tp_atr_mult

        # ATR-fractional sizing: risk a fixed % of equity per trade
        if self.risk_per_trade_pct > 0 and atr > 0 and ctx.equity > 0:
            risk_dollars = ctx.equity * self.risk_per_trade_pct
            stop_distance = atr * self.sl_atr_mult
            sizing = risk_dollars / stop_distance
        else:
            sizing = graph_state.bias_strength

        return Signal(
            direction=graph_state.bias,
            strength=sizing,
            reason=f"entry|bias={graph_state.bias:.0f}|str={graph_state.bias_strength:.2f}",
        )

    def _check_exit(
        self,
        ctx: StrategyContext,
        close: float,
        atr: float,
        current_bias: float,
    ) -> Signal:
        """Check exit conditions for an open position."""
        bar_high = float(ctx.bar["high"])
        bar_low = float(ctx.bar["low"])

        # Stop loss
        if self._in_position_side > 0 and bar_low <= self._stop_level:
            self._in_position_side = 0.0
            return Signal(direction=0.0, strength=1.0, reason="exit_stop_loss")
        if self._in_position_side < 0 and bar_high >= self._stop_level:
            self._in_position_side = 0.0
            return Signal(direction=0.0, strength=1.0, reason="exit_stop_loss")

        # Take profit
        if self._in_position_side > 0 and bar_high >= self._tp_level:
            self._in_position_side = 0.0
            return Signal(direction=0.0, strength=1.0, reason="exit_take_profit")
        if self._in_position_side < 0 and bar_low <= self._tp_level:
            self._in_position_side = 0.0
            return Signal(direction=0.0, strength=1.0, reason="exit_take_profit")

        # Max hold bars
        if ctx.bar_index - self._entry_bar >= self.max_hold_bars:
            self._in_position_side = 0.0
            return Signal(direction=0.0, strength=1.0, reason="exit_max_hold")

        # Bias flip
        if current_bias != 0.0 and current_bias != self._in_position_side:
            self._in_position_side = 0.0
            return Signal(direction=0.0, strength=1.0, reason="exit_bias_flip")

        # Hold
        return Signal(
            direction=self._in_position_side,
            strength=1.0,
            reason="hold",
        )

    def _detect_levels(self, bar_offset: int, atr: float) -> list[Level]:
        """Run all detectors on the current window."""
        levels: list[Level] = []
        left = self.swing_lookback
        right = self.swing_lookback

        levels.extend(detect_bsl(self._highs, bar_offset, left, right))
        levels.extend(detect_ssl(self._lows, bar_offset, left, right))
        levels.extend(detect_fvg(
            self._opens, self._highs, self._lows, self._closes,
            bar_offset, self.fvg_min_gap_atr, atr,
        ))
        levels.extend(detect_ob(
            self._opens, self._highs, self._lows, self._closes,
            bar_offset, self.ob_min_impulse_atr, atr,
        ))
        return levels

    def _has_confluence(self, close: float, atr: float, bias: float) -> bool:
        """Check if there is a confluent FVG or OB near price in bias direction."""
        proximity = self.proximity_atr * atr
        for lv in self._graph.levels:
            if lv.swept:
                continue
            mid = (lv.price_low + lv.price_high) / 2.0
            if abs(mid - close) > proximity:
                continue
            # Bullish bias: look for FVG_BULL or OB_BULL below/near price
            if bias > 0 and lv.level_type in (LevelType.FVG_BULL, LevelType.OB_BULL):
                if mid <= close:
                    return True
            # Bearish bias: look for FVG_BEAR or OB_BEAR above/near price
            if bias < 0 and lv.level_type in (LevelType.FVG_BEAR, LevelType.OB_BEAR):
                if mid >= close:
                    return True
        return False


def build(params: dict) -> tuple[GraphMarketStructureStrategy, list[FeatureSpec]]:
    """Build a GraphMarketStructureStrategy and its FeatureSpecs from config."""
    from src.indicators.impl.atr import ATR
    from src.indicators.impl.transforms import Lag

    atr_period = params.get("atr_period", 14)
    atr_ind = ATR(atr_period)
    lag = Lag(1)

    specs = [FeatureSpec(atr_ind, [lag])]
    atr_col = f"{atr_ind.name}{lag.name}"  # e.g. atr_14_lag_1

    swing_lookback = params.get("swing_lookback", 3)
    window_bars = params.get("window_bars", 50)
    warmup = max(atr_period + 1, window_bars) + 1

    strategy = GraphMarketStructureStrategy(
        atr_col=atr_col,
        swing_lookback=swing_lookback,
        fvg_min_gap_atr=params.get("fvg_min_gap_atr", 0.5),
        ob_min_impulse_atr=params.get("ob_min_impulse_atr", 1.5),
        max_levels=params.get("max_levels", 200),
        proximity_atr=params.get("proximity_atr", 3.0),
        entry_threshold=params.get("entry_threshold", 0.3),
        sl_atr_mult=params.get("sl_atr_mult", 2.0),
        tp_atr_mult=params.get("tp_atr_mult", 3.0),
        max_hold_bars=params.get("max_hold_bars", 50),
        risk_per_trade_pct=params.get("risk_per_trade_pct", 0.0),
        detect_every=params.get("detect_every", 1),
        recompute_every=params.get("recompute_every", 5),
        window_bars=window_bars,
        _warmup_bars=warmup,
    )

    return strategy, specs
