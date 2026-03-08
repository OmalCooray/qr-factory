"""Donchian breakout + ATR stop strategy, gated by HMM regime filter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from src.indicators.core.pipeline import FeatureSpec
from src.regime import TrendRangeFilter
from src.strategy.signal import Signal
from .base import StrategyContext, validate_features


@dataclass
class DonchianATRRegimeStrategy:
    """Donchian breakout entries + ATR trailing stop, gated by HMM regime.

    Only opens new positions when regime == TREND.  Exits are regime-agnostic
    (Donchian exit channel breach or ATR stop hit).
    """

    hh_col: str          # entry high channel (e.g. donchian_high_55_lag_1)
    ll_col: str          # entry low channel
    exit_high_col: str   # exit high channel (e.g. donchian_high_20_lag_1)
    exit_low_col: str    # exit low channel
    atr_col: str         # ATR column
    stop_atr_mult: float
    regime_filter: TrendRangeFilter
    long_only: bool = False
    risk_per_trade_pct: float = 0.0  # 0 = use engine's fixed position_size
    _name: str = "donchian_atr_regime"
    _warmup_bars: int = 0

    # Internal mutable state (not config)
    _stop_level: float = field(default=0.0, init=False, repr=False)
    _in_position_side: float = field(default=0.0, init=False, repr=False)  # +1, -1, 0

    # Regime recording for artifact writing
    regime_records: list[dict] = field(default_factory=list, init=False, repr=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> Sequence[str]:
        return [self.hh_col, self.ll_col, self.exit_high_col, self.exit_low_col, self.atr_col]

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        if ctx.bar_index < self.warmup_bars:
            return False
        if not validate_features(ctx.features, self.required_features):
            return False
        return True

    def extra_artifacts(self) -> dict[str, list[dict]]:
        """Return regime records for persistence."""
        return {"regime": self.regime_records}

    def on_bar(self, ctx: StrategyContext) -> Signal:
        # 1. Feed close to regime filter
        regime_info = self.regime_filter.update(float(ctx.bar["close"]))

        # 2. Record regime info
        self.regime_records.append({
            "ts": str(ctx.ts),
            "bar_index": ctx.bar_index,
            "p_trend": regime_info.p_trend,
            "regime": regime_info.regime,
            "trend_state": regime_info.trend_state,
            "er": regime_info.er,
            "vol": regime_info.vol,
            "mom": regime_info.mom,
        })

        if not self.can_trade(ctx):
            return Signal(direction=0.0, strength=0.0, reason="warmup/feature_nan")

        # 3. Read features
        hh = ctx.features[self.hh_col]
        ll = ctx.features[self.ll_col]
        exit_high = ctx.features[self.exit_high_col]
        exit_low = ctx.features[self.exit_low_col]
        atr = ctx.features[self.atr_col]
        close = float(ctx.bar["close"])
        bar_high = float(ctx.bar["high"])
        bar_low = float(ctx.bar["low"])

        # Sync internal side tracker with engine position
        if ctx.position == 0.0:
            self._in_position_side = 0.0
            self._stop_level = 0.0

        # Compute ATR-scaled sizing if enabled
        if self.risk_per_trade_pct > 0 and atr > 0 and ctx.equity > 0:
            risk_dollars = ctx.equity * self.risk_per_trade_pct
            stop_distance = atr * self.stop_atr_mult
            sizing_strength = risk_dollars / stop_distance
        else:
            sizing_strength = 0.0  # 0 means use engine's fixed position_size

        # 4. If flat and regime is TREND: check breakout entries
        if self._in_position_side == 0.0:
            if regime_info.regime == "TREND":
                if close > hh:
                    # Long breakout
                    self._in_position_side = 1.0
                    self._stop_level = close - atr * self.stop_atr_mult
                    return Signal(
                        direction=1.0,
                        strength=sizing_strength if sizing_strength > 0 else regime_info.p_trend,
                        reason=f"long_breakout|p_trend={regime_info.p_trend:.2f}",
                    )
                elif close < ll and not self.long_only:
                    # Short breakout
                    self._in_position_side = -1.0
                    self._stop_level = close + atr * self.stop_atr_mult
                    return Signal(
                        direction=-1.0,
                        strength=sizing_strength if sizing_strength > 0 else regime_info.p_trend,
                        reason=f"short_breakout|p_trend={regime_info.p_trend:.2f}",
                    )
            # No entry signal
            return Signal(direction=0.0, strength=0.0, reason=f"flat|regime={regime_info.regime}")

        # 5. If in position: check exits
        if self._in_position_side == 1.0:
            # Long exit: close drops below exit low channel, or bar low breaches stop
            if close < exit_low or bar_low <= self._stop_level:
                reason = "exit_donchian_low" if close < exit_low else "exit_atr_stop"
                self._in_position_side = 0.0
                self._stop_level = 0.0
                return Signal(direction=0.0, strength=1.0, reason=reason)
            # Stay long
            return Signal(direction=1.0, strength=1.0, reason="hold_long")

        if self._in_position_side == -1.0:
            # Short exit: close rises above exit high channel, or bar high breaches stop
            if close > exit_high or bar_high >= self._stop_level:
                reason = "exit_donchian_high" if close > exit_high else "exit_atr_stop"
                self._in_position_side = 0.0
                self._stop_level = 0.0
                return Signal(direction=0.0, strength=1.0, reason=reason)
            # Stay short
            return Signal(direction=-1.0, strength=1.0, reason="hold_short")

        # Fallback (shouldn't reach here)
        return Signal(direction=0.0, strength=0.0, reason="unknown_state")


def build(params: dict) -> tuple[DonchianATRRegimeStrategy, list[FeatureSpec]]:
    """Build a DonchianATRRegimeStrategy and its FeatureSpecs from config."""
    from src.indicators.impl.donchian import DonchianHigh, DonchianLow
    from src.indicators.impl.atr import ATR
    from src.indicators.impl.transforms import Lag

    N_entry: int = params.get("N_entry", 55)
    N_exit: int = params.get("N_exit", 20)
    ATR_len: int = params.get("ATR_len", 14)
    stop_ATR: float = params.get("stop_ATR", 2.0)

    # Indicators with Lag(1) to avoid lookahead
    entry_high = DonchianHigh(N_entry)
    entry_low = DonchianLow(N_entry)
    exit_high = DonchianHigh(N_exit)
    exit_low = DonchianLow(N_exit)
    atr_ind = ATR(ATR_len)
    lag = Lag(1)

    specs = [
        FeatureSpec(entry_high, [lag]),
        FeatureSpec(entry_low, [lag]),
        FeatureSpec(exit_high, [lag]),
        FeatureSpec(exit_low, [lag]),
        FeatureSpec(atr_ind, [lag]),
    ]

    # Build column names (base_name + transform_name)
    hh_col = f"{entry_high.name}{lag.name}"       # donchian_high_55_lag_1
    ll_col = f"{entry_low.name}{lag.name}"         # donchian_low_55_lag_1
    exit_high_col = f"{exit_high.name}{lag.name}"  # donchian_high_20_lag_1
    exit_low_col = f"{exit_low.name}{lag.name}"    # donchian_low_20_lag_1
    atr_col = f"{atr_ind.name}{lag.name}"          # atr_14_lag_1

    # Regime filter config
    rf_cfg = params.get("regime_filter", {})
    hmm_cfg = rf_cfg.get("hmm", {})
    regime_filter = TrendRangeFilter(
        k=rf_cfg.get("k", 20),
        window_W=rf_cfg.get("window_W", 5000),
        refit_every=rf_cfg.get("refit_every", 50),
        predict_every=rf_cfg.get("predict_every", 10),
        enter_th=rf_cfg.get("enter_th", 0.70),
        exit_th=rf_cfg.get("exit_th", 0.50),
        features=rf_cfg.get("features", ["ER", "vol", "mom"]),
        covariance_type=hmm_cfg.get("covariance_type", "full"),
        n_iter=hmm_cfg.get("n_iter", 200),
        random_state=hmm_cfg.get("random_state", 42),
    )

    warmup = max(N_entry, N_exit, ATR_len, rf_cfg.get("k", 20)) + 1

    strategy = DonchianATRRegimeStrategy(
        hh_col=hh_col,
        ll_col=ll_col,
        exit_high_col=exit_high_col,
        exit_low_col=exit_low_col,
        atr_col=atr_col,
        stop_atr_mult=stop_ATR,
        regime_filter=regime_filter,
        long_only=params.get("long_only", False),
        risk_per_trade_pct=params.get("risk_per_trade_pct", 0.0),
        _warmup_bars=warmup,
    )

    return strategy, specs
