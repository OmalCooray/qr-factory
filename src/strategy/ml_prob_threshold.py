"""Probability-threshold strategy for ML walk-forward with hysteresis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import pandas as pd

from src.indicators.core.pipeline import FeaturePipeline, FeatureSpec
from src.indicators.impl.adx import ADX
from src.indicators.impl.atr import ATR
from src.indicators.impl.bb import BollingerLower, BollingerUpper
from src.indicators.impl.cci import CCI
from src.indicators.impl.ema import EMA
from src.indicators.impl.ma import SMA
from src.indicators.impl.rsi import RSI
from src.indicators.impl.session import TradingSession
from src.indicators.impl.transforms import Diff, Lag, ZScoreRolling
from src.strategy.base import StrategyContext
from src.strategy.signal import Signal


# ── Indicator / transform maps for config-driven features ────────────────────

_INDICATOR_MAP: dict[str, callable] = {
    "ema": lambda p: EMA(period=p.get("period", 5)),
    "sma": lambda p: SMA(period=p.get("period", 10)),
    "rsi": lambda p: RSI(period=p.get("period", 14)),
    "atr": lambda p: ATR(period=p.get("period", 14)),
    "adx": lambda p: ADX(period=p.get("period", 14)),
    "bb_upper": lambda p: BollingerUpper(period=p.get("period", 20)),
    "bb_lower": lambda p: BollingerLower(period=p.get("period", 20)),
    "cci": lambda p: CCI(period=p.get("period", 20)),
    "trading_session": lambda p: TradingSession(),
}

_TRANSFORM_MAP: dict[str, callable] = {
    "zscore": lambda p: ZScoreRolling(window=p.get("window", 20)),
    "diff": lambda p: Diff(k=p.get("k", 1)),
    "lag": lambda p: Lag(k=p.get("k", 1)),
}


def _build_feature_specs(feature_cfgs: list[dict]) -> list[FeatureSpec]:
    """Convert a list of feature config dicts into FeatureSpec objects."""
    specs: list[FeatureSpec] = []
    for fcfg in feature_cfgs:
        fcfg = dict(fcfg)  # shallow copy
        ind_type = fcfg.pop("type")
        transforms_cfg = fcfg.pop("transforms", [])

        if ind_type not in _INDICATOR_MAP:
            raise ValueError(
                f"Unknown indicator type: {ind_type!r}. "
                f"Available: {sorted(_INDICATOR_MAP)}"
            )
        base = _INDICATOR_MAP[ind_type](fcfg)

        transforms = []
        for tcfg in transforms_cfg:
            tcfg = dict(tcfg)
            t_type = tcfg.pop("type")
            if t_type not in _TRANSFORM_MAP:
                raise ValueError(
                    f"Unknown transform type: {t_type!r}. "
                    f"Available: {sorted(_TRANSFORM_MAP)}"
                )
            transforms.append(_TRANSFORM_MAP[t_type](tcfg))

        specs.append(FeatureSpec(base=base, transforms=transforms))
    return specs


@dataclass
class MLProbThresholdStrategy:
    """Go long when yhat >= p_enter, with optional hysteresis dead-band.

    Hysteresis logic:
    - **When flat:** enter long only if ``yhat >= p_enter``
    - **When long:** hold if ``yhat >= p_exit`` (dead-band between p_exit..p_enter)
    - **When long:** exit only if ``yhat < p_exit`` AND ``bars_held >= min_hold_bars``

    If ``p_exit is None`` (default), falls back to ``p_enter`` for backward
    compatibility (no dead-band).
    """

    p_enter: float = 0.55
    p_exit: float | None = None
    min_hold_bars: int = 0
    _name: str = "ml_prob_threshold"
    _warmup_bars: int = 0
    _bars_in_position: list = field(default_factory=lambda: [0])

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> Sequence[str]:
        return ["yhat"]

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        if ctx.bar_index < self._warmup_bars:
            return False
        if "yhat" not in ctx.features.index:
            return False
        if pd.isna(ctx.features["yhat"]):
            return False
        return True

    def on_bar(self, ctx: StrategyContext) -> Signal:
        if not self.can_trade(ctx):
            self._bars_in_position[0] = 0
            return Signal(direction=0.0, strength=0.0, reason="no_yhat")

        yhat = float(ctx.features["yhat"])
        effective_p_exit = self.p_exit if self.p_exit is not None else self.p_enter
        is_long = ctx.position > 0

        if is_long:
            self._bars_in_position[0] += 1
            bars_held = self._bars_in_position[0]

            # Hold: still above p_exit OR haven't held long enough
            if yhat >= effective_p_exit or bars_held < self.min_hold_bars:
                return Signal(
                    direction=1.0, strength=1.0,
                    reason=f"hold yhat={yhat:.3f} bars={bars_held}",
                )
            # Exit: below p_exit AND held long enough
            self._bars_in_position[0] = 0
            return Signal(
                direction=0.0, strength=1.0,
                reason=f"exit yhat={yhat:.3f}<p_exit={effective_p_exit:.3f}",
            )
        else:
            # Flat: enter only if above p_enter
            if yhat >= self.p_enter:
                self._bars_in_position[0] = 1
                return Signal(
                    direction=1.0, strength=1.0,
                    reason=f"enter yhat={yhat:.3f}>=p_enter",
                )
            self._bars_in_position[0] = 0
            return Signal(
                direction=0.0, strength=1.0,
                reason=f"flat yhat={yhat:.3f}<p_enter",
            )


def build(params: dict) -> tuple[MLProbThresholdStrategy, list[FeatureSpec]]:
    """Builder for strategy registry.

    Supports two modes:
    1. **Legacy** (no ``features`` key): uses ``fast_period``/``slow_period``
       to create EMA + SMA specs.
    2. **Config-driven** (``features`` key present): builds FeatureSpecs from
       the list of indicator/transform dicts.
    """
    p_enter = params.get("p_enter", 0.55)
    p_exit = params.get("p_exit", None)
    min_hold_bars = params.get("min_hold_bars", 0)

    feature_cfgs = params.get("features")
    if feature_cfgs is not None:
        # Config-driven rich features
        specs = _build_feature_specs(feature_cfgs)
        warmup = FeaturePipeline(specs).max_lookback
    else:
        # Legacy: EMA + SMA
        fast_period = params.get("fast_period", 5)
        slow_period = params.get("slow_period", 10)
        specs = [
            FeatureSpec(base=EMA(period=fast_period)),
            FeatureSpec(base=SMA(period=slow_period)),
        ]
        warmup = slow_period

    strategy = MLProbThresholdStrategy(
        p_enter=p_enter,
        p_exit=p_exit,
        min_hold_bars=min_hold_bars,
        _warmup_bars=warmup,
    )
    return strategy, specs
