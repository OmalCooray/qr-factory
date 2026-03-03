"""Lorentzian Classification — KNN in Lorentzian space (V2).

Reimplementation of jdehorty's TradingView indicator as a qr-factory strategy.

V2 enhancements (matching original Pine Script):
  - Kernel regression trend filter (Rational Quadratic + Gaussian)
  - Adaptive running min/max normalization for CCI and WaveTrend
  - Ehlers adaptive MA regime filter

Algorithm:
  1. Compute 5 normalized features per bar (RSI, WT, CCI, ADX, RSI-short).
  2. Maintain a rolling window of historical feature vectors with labels.
  3. For each bar, compute Lorentzian distance to historical bars:
       d(x,y) = Σ log(1 + |x_i - y_i|)
  4. Find k-nearest neighbours, sum their labels (+1 bullish, -1 bearish).
  5. Apply filters (kernel, regime, volatility, ADX) to gate signal generation.
  6. Emit Signal on direction changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from src.indicators.core.pipeline import FeatureSpec
from src.strategy.signal import Signal
from .base import StrategyContext, validate_features

_N_FEATURES = 5


# ── Kernel helpers ──────────────────────────────────────────────────────────


def _rq_weights(size: int, lookback: float, rel_weight: float) -> np.ndarray:
    """Precompute Rational Quadratic kernel weights."""
    i = np.arange(size, dtype=np.float64)
    w = np.power(1.0 + i * i / (lookback * lookback * 2.0 * rel_weight), -rel_weight)
    return w


def _gauss_weights(size: int, lookback: float) -> np.ndarray:
    """Precompute Gaussian kernel weights."""
    i = np.arange(size, dtype=np.float64)
    w = np.exp(-i * i / (2.0 * lookback * lookback))
    return w


# ── Regime filter helper ───────────────────────────────────────────────────


class _EhlersRegimeFilter:
    """Ehlers adaptive-MA regime filter (Pine: regime_filter).

    Tracks a Kaufman-like adaptive moving average and its slope.
    Passes when normalized slope decline >= threshold.
    """

    def __init__(self, threshold: float = -0.1) -> None:
        self.threshold = threshold
        self._value1 = 0.0
        self._value2 = 1e-10  # avoid /0 on first bar
        self._klmf = 0.0
        self._prev_klmf = 0.0
        self._ema_slope = 0.0  # EMA(200) of abs curve slope
        self._ema_alpha = 2.0 / (200.0 + 1.0)
        self._bar_count = 0

    def update(self, ohlc4: float, high: float, low: float) -> bool:
        """Update filter state and return True if filter passes."""
        self._bar_count += 1

        # Smoothed momentum and volatility
        if self._bar_count == 1:
            self._value1 = 0.0
            self._value2 = 0.1 * (high - low)
            self._klmf = ohlc4
            self._prev_klmf = ohlc4
            self._ema_slope = 0.0
            return True

        self._value1 = 0.2 * (ohlc4 - self._prev_ohlc4) + 0.8 * self._value1
        self._value2 = 0.1 * (high - low) + 0.8 * self._value2

        # Efficiency ratio → adaptive alpha
        if abs(self._value2) < 1e-10:
            omega = 0.0
        else:
            omega = abs(self._value1 / self._value2)

        omega2 = omega * omega
        alpha = (-omega2 + math.sqrt(omega2 * omega2 + 16.0 * omega2)) / 8.0
        alpha = max(0.0, min(1.0, alpha))

        # Kaufman-like adaptive MA
        self._prev_klmf = self._klmf
        self._klmf = alpha * ohlc4 + (1.0 - alpha) * self._klmf

        # Slope and its EMA(200)
        abs_curve_slope = abs(self._klmf - self._prev_klmf)
        self._ema_slope = (
            self._ema_alpha * abs_curve_slope
            + (1.0 - self._ema_alpha) * self._ema_slope
        )

        if self._ema_slope < 1e-10:
            return True  # no meaningful slope yet

        normalized_decline = (abs_curve_slope - self._ema_slope) / self._ema_slope
        return normalized_decline >= self.threshold

    def pre_update(self, ohlc4: float) -> None:
        """Store ohlc4 for next bar's momentum calc (call before update)."""
        self._prev_ohlc4 = ohlc4


# ── Strategy ────────────────────────────────────────────────────────────────


@dataclass
class LorentzianClassificationStrategy:
    """KNN classification using Lorentzian distance metric (V2)."""

    # Feature column names (set by build())
    rsi_col: str
    wt_col: str
    cci_col: str
    adx_col: str
    rsi_short_col: str
    ema_col: str
    atr_short_col: str
    atr_long_col: str

    # KNN parameters
    neighbors_count: int = 8
    max_bars_back: int = 2000
    lookahead: int = 4

    # Kernel regression parameters
    use_kernel_filter: bool = True
    kernel_lookback: int = 8
    relative_weight: float = 8.0
    kernel_window: int = 64  # bars for kernel estimate

    # Ehlers regime filter
    use_regime_filter: bool = True
    regime_threshold: float = -0.1

    # Other filters
    use_ema_filter: bool = False  # disabled by default, kernel replaces it
    use_volatility_filter: bool = True
    use_adx_filter: bool = False  # disabled by default (Pine default)
    adx_threshold: float = 20.0

    _name: str = "lorentzian_classification"
    _warmup_bars: int = 0

    # ── Internal mutable state ──
    # Ring buffer for KNN
    _feat_buf: np.ndarray = field(default=None, init=False, repr=False)
    _label_buf: np.ndarray = field(default=None, init=False, repr=False)
    _buf_len: int = field(default=0, init=False, repr=False)
    _buf_idx: int = field(default=0, init=False, repr=False)

    # Pending close/feature queues for delayed labelling
    _pend_feats: np.ndarray = field(default=None, init=False, repr=False)
    _pend_close: np.ndarray = field(default=None, init=False, repr=False)
    _pend_len: int = field(default=0, init=False, repr=False)

    # Kernel regression state
    _rq_w: np.ndarray = field(default=None, init=False, repr=False)
    _gauss_w: np.ndarray = field(default=None, init=False, repr=False)
    _close_ring: np.ndarray = field(default=None, init=False, repr=False)
    _close_ring_len: int = field(default=0, init=False, repr=False)
    _close_ring_idx: int = field(default=0, init=False, repr=False)
    _prev_yhat_rq: float = field(default=0.0, init=False, repr=False)

    # Adaptive normalization running min/max
    _wt_min: float = field(default=1e10, init=False, repr=False)
    _wt_max: float = field(default=-1e10, init=False, repr=False)
    _cci_min: float = field(default=1e10, init=False, repr=False)
    _cci_max: float = field(default=-1e10, init=False, repr=False)

    # Ehlers regime filter
    _regime_filter: _EhlersRegimeFilter = field(default=None, init=False, repr=False)

    _last_direction: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        m = self.max_bars_back
        self._feat_buf = np.zeros((m, _N_FEATURES), dtype=np.float64)
        self._label_buf = np.zeros(m, dtype=np.float64)
        cap = self.lookahead + 2
        self._pend_feats = np.zeros((cap, _N_FEATURES), dtype=np.float64)
        self._pend_close = np.zeros(cap, dtype=np.float64)

        # Kernel weights
        kw = self.kernel_window
        self._rq_w = _rq_weights(kw, float(self.kernel_lookback), self.relative_weight)
        self._gauss_w = _gauss_weights(kw, float(self.kernel_lookback))
        self._close_ring = np.zeros(kw, dtype=np.float64)

        # Regime filter
        self._regime_filter = _EhlersRegimeFilter(self.regime_threshold)

    @property
    def name(self) -> str:
        return self._name

    @property
    def required_features(self) -> Sequence[str]:
        return [
            self.rsi_col, self.wt_col, self.cci_col,
            self.adx_col, self.rsi_short_col,
            self.ema_col, self.atr_short_col, self.atr_long_col,
        ]

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars

    def can_trade(self, ctx: StrategyContext) -> bool:
        if ctx.bar_index < self.warmup_bars:
            return False
        if not validate_features(ctx.features, self.required_features):
            return False
        return True

    # ── Ring buffer helpers ─────────────────────────────────────────────

    def _push_labelled(self, feat: np.ndarray, label: float) -> None:
        idx = self._buf_idx
        self._feat_buf[idx] = feat
        self._label_buf[idx] = label
        self._buf_idx = (idx + 1) % self.max_bars_back
        if self._buf_len < self.max_bars_back:
            self._buf_len += 1

    def _push_close(self, close: float) -> None:
        """Push close into the kernel close-price ring buffer."""
        kw = self.kernel_window
        self._close_ring[self._close_ring_idx] = close
        self._close_ring_idx = (self._close_ring_idx + 1) % kw
        if self._close_ring_len < kw:
            self._close_ring_len += 1

    def _kernel_estimate(self, weights: np.ndarray) -> float:
        """Nadaraya-Watson weighted average over close ring buffer."""
        n = self._close_ring_len
        if n == 0:
            return 0.0
        # Extract closes in reverse chronological order
        idx = self._close_ring_idx  # points to next write = one past newest
        buf = self._close_ring
        kw = self.kernel_window
        # Build array: most recent first
        closes = np.empty(n, dtype=np.float64)
        for i in range(n):
            closes[i] = buf[(idx - 1 - i) % kw]
        w = weights[:n]
        return float(np.dot(closes, w) / np.sum(w))

    # ── Adaptive normalization (Pine normalize()) ───────────────────────

    def _adapt_norm_wt(self, val: float) -> float:
        if math.isnan(val):
            return 0.5
        self._wt_min = min(val, self._wt_min)
        self._wt_max = max(val, self._wt_max)
        rng = self._wt_max - self._wt_min
        if rng < 1e-10:
            return 0.5
        return (val - self._wt_min) / rng

    def _adapt_norm_cci(self, val: float) -> float:
        if math.isnan(val):
            return 0.5
        self._cci_min = min(val, self._cci_min)
        self._cci_max = max(val, self._cci_max)
        rng = self._cci_max - self._cci_min
        if rng < 1e-10:
            return 0.5
        return (val - self._cci_min) / rng

    # ── Main per-bar logic ──────────────────────────────────────────────

    def on_bar(self, ctx: StrategyContext) -> Signal:
        if not self.can_trade(ctx):
            return Signal(direction=0.0, strength=0.0, reason="warmup/feature_nan")

        close = float(ctx.bar["close"])
        high = float(ctx.bar["high"])
        low = float(ctx.bar["low"])
        open_ = float(ctx.bar["open"])
        ohlc4 = (open_ + high + low + close) / 4.0

        # ── 1. Regime filter update ──
        self._regime_filter.pre_update(ohlc4)
        regime_ok = True
        if self.use_regime_filter:
            regime_ok = self._regime_filter.update(ohlc4, high, low)

        # ── 2. Kernel regression update ──
        self._push_close(close)
        yhat_rq = self._kernel_estimate(self._rq_w)
        kernel_bullish = yhat_rq > self._prev_yhat_rq
        kernel_bearish = yhat_rq < self._prev_yhat_rq
        self._prev_yhat_rq = yhat_rq

        # ── 3. Read raw indicator values ──
        rsi_val = float(ctx.features[self.rsi_col])
        wt_val = float(ctx.features[self.wt_col])
        cci_val = float(ctx.features[self.cci_col])
        adx_val = float(ctx.features[self.adx_col])
        rsi_short_val = float(ctx.features[self.rsi_short_col])
        ema_val = float(ctx.features[self.ema_col])
        atr_short_val = float(ctx.features[self.atr_short_col])
        atr_long_val = float(ctx.features[self.atr_long_col])

        # ── 4. Normalize features to [0, 1] ──
        # RSI/ADX: fixed rescale (bounded 0-100)
        f1 = rsi_val / 100.0
        f4 = adx_val / 100.0
        f5 = rsi_short_val / 100.0
        # WT/CCI: adaptive running min/max normalization (Pine normalize())
        f2 = self._adapt_norm_wt(wt_val)
        f3 = self._adapt_norm_cci(cci_val)

        feature_vec = np.array([f1, f2, f3, f4, f5], dtype=np.float64)

        # ── 5. Delayed labelling ──
        plen = self._pend_len
        cap = self.lookahead + 2
        if plen < cap:
            self._pend_feats[plen] = feature_vec
            self._pend_close[plen] = close
            self._pend_len = plen + 1
        else:
            self._pend_feats[:-1] = self._pend_feats[1:]
            self._pend_close[:-1] = self._pend_close[1:]
            self._pend_feats[-1] = feature_vec
            self._pend_close[-1] = close
            self._pend_len = cap

        if self._pend_len == cap:
            old_feat = self._pend_feats[0].copy()
            old_close = self._pend_close[0]
            future_close = self._pend_close[self.lookahead]
            label = 1.0 if future_close > old_close else -1.0
            self._push_labelled(old_feat, label)
            self._pend_feats[:-1] = self._pend_feats[1:]
            self._pend_close[:-1] = self._pend_close[1:]
            self._pend_len -= 1

        # Need enough labelled history for KNN
        n = self._buf_len
        if n < self.neighbors_count + 10:
            return Signal(direction=0.0, strength=0.0, reason="building_history")

        # ── 6. Vectorised KNN Classification ──
        feats = self._feat_buf[:n]
        labels = self._label_buf[:n]
        diff = np.abs(feats - feature_vec)
        distances = np.sum(np.log1p(diff), axis=1)

        k = min(self.neighbors_count, n)
        nearest_idx = np.argpartition(distances, k)[:k]
        prediction = float(np.sum(labels[nearest_idx]))

        is_bullish = prediction > 0
        is_bearish = prediction < 0

        # ── 7. Filters (gate entries AND exits — safety net for M5 gold) ──
        kernel_ok = True
        if self.use_kernel_filter:
            if is_bullish:
                kernel_ok = kernel_bullish
            elif is_bearish:
                kernel_ok = kernel_bearish

        ema_ok = True
        if self.use_ema_filter:
            if is_bullish:
                ema_ok = close > ema_val
            elif is_bearish:
                ema_ok = close < ema_val

        volatility_ok = True
        if self.use_volatility_filter:
            volatility_ok = atr_short_val > atr_long_val

        adx_ok = True
        if self.use_adx_filter:
            adx_ok = adx_val > self.adx_threshold

        filters_pass = kernel_ok and regime_ok and ema_ok and volatility_ok and adx_ok

        # ── 8. Signal Generation ──
        if is_bullish and filters_pass:
            direction = 1.0
        elif is_bearish and filters_pass:
            direction = -1.0
        else:
            direction = 0.0

        strength = min(abs(prediction) / self.neighbors_count, 1.0)

        if direction == self._last_direction and direction != 0.0:
            return Signal(direction=direction, strength=strength,
                          reason=f"hold|pred={prediction:.0f}")

        self._last_direction = direction

        reason_parts = [f"pred={prediction:.0f}"]
        if not kernel_ok:
            reason_parts.append("kernel_blocked")
        if not regime_ok:
            reason_parts.append("regime_blocked")
        if not ema_ok:
            reason_parts.append("ema_blocked")
        if not volatility_ok:
            reason_parts.append("vol_blocked")
        if not adx_ok:
            reason_parts.append("adx_blocked")

        return Signal(direction=direction, strength=strength,
                      reason="|".join(reason_parts))


def build(params: dict) -> tuple[LorentzianClassificationStrategy, list[FeatureSpec]]:
    """Build a LorentzianClassificationStrategy and its FeatureSpecs from config."""
    from src.indicators.impl.rsi import RSI
    from src.indicators.impl.cci import CCI
    from src.indicators.impl.wt import WaveTrend
    from src.indicators.impl.adx import ADX
    from src.indicators.impl.ema import EMA
    from src.indicators.impl.atr import ATR
    from src.indicators.impl.transforms import Lag

    rsi_period = params.get("rsi_period", 14)
    rsi_short_period = params.get("rsi_short_period", 9)
    cci_period = params.get("cci_period", 20)
    wt_n1 = params.get("wt_n1", 10)
    wt_n2 = params.get("wt_n2", 11)
    adx_period = params.get("adx_period", 20)
    ema_period = params.get("ema_period", 200)
    atr_short_period = params.get("atr_short_period", 1)
    atr_long_period = params.get("atr_long_period", 10)

    rsi = RSI(rsi_period)
    rsi_short = RSI(rsi_short_period)
    cci = CCI(cci_period)
    wt = WaveTrend(wt_n1, wt_n2)
    adx = ADX(adx_period)
    ema = EMA(ema_period)
    atr_short = ATR(atr_short_period)
    atr_long = ATR(atr_long_period)
    lag = Lag(1)

    specs = [
        FeatureSpec(rsi, [lag]),
        FeatureSpec(rsi_short, [lag]),
        FeatureSpec(cci, [lag]),
        FeatureSpec(wt, [lag]),
        FeatureSpec(adx, [lag]),
        FeatureSpec(ema, [lag]),
        FeatureSpec(atr_short, [lag]),
        FeatureSpec(atr_long, [lag]),
    ]

    rsi_col = f"{rsi.name}{lag.name}"
    rsi_short_col = f"{rsi_short.name}{lag.name}"
    cci_col = f"{cci.name}{lag.name}"
    wt_col = f"{wt.name}{lag.name}"
    adx_col = f"{adx.name}{lag.name}"
    ema_col = f"{ema.name}{lag.name}"
    atr_short_col = f"{atr_short.name}{lag.name}"
    atr_long_col = f"{atr_long.name}{lag.name}"

    # KNN parameters
    neighbors_count = params.get("neighbors_count", 8)
    max_bars_back = params.get("max_bars_back", 2000)
    lookahead = params.get("lookahead", 4)

    # Kernel parameters
    use_kernel_filter = params.get("use_kernel_filter", True)
    kernel_lookback = params.get("kernel_lookback", 8)
    relative_weight = params.get("relative_weight", 8.0)
    kernel_window = params.get("kernel_window", 64)

    # Regime filter
    use_regime_filter = params.get("use_regime_filter", True)
    regime_threshold = params.get("regime_threshold", -0.1)

    # Other filters
    use_ema_filter = params.get("use_ema_filter", False)
    use_volatility_filter = params.get("use_volatility_filter", True)
    use_adx_filter = params.get("use_adx_filter", False)
    adx_threshold = params.get("adx_threshold", 20.0)

    warmup = max(rsi_period, rsi_short_period, cci_period,
                 wt_n1 + wt_n2 + 4, adx_period * 2, ema_period,
                 atr_long_period) + 2

    strategy = LorentzianClassificationStrategy(
        rsi_col=rsi_col,
        wt_col=wt_col,
        cci_col=cci_col,
        adx_col=adx_col,
        rsi_short_col=rsi_short_col,
        ema_col=ema_col,
        atr_short_col=atr_short_col,
        atr_long_col=atr_long_col,
        neighbors_count=neighbors_count,
        max_bars_back=max_bars_back,
        lookahead=lookahead,
        use_kernel_filter=use_kernel_filter,
        kernel_lookback=kernel_lookback,
        relative_weight=relative_weight,
        kernel_window=kernel_window,
        use_regime_filter=use_regime_filter,
        regime_threshold=regime_threshold,
        use_ema_filter=use_ema_filter,
        use_volatility_filter=use_volatility_filter,
        use_adx_filter=use_adx_filter,
        adx_threshold=adx_threshold,
        _warmup_bars=warmup,
    )

    return strategy, specs
