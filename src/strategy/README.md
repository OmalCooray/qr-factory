# Strategy System

**Protocol-based strategy framework with registry pattern for pluggable trading logic.**

## Overview

The `src/strategy/` package implements the strategy layer of the qr-factory backtest engine. Strategies produce **Signal** objects representing the model's directional belief (what the model thinks), which are then converted to **OrderIntent** objects (what to actually execute) by the decision layer in `TradingEngine`.

**Core principle: Predict ≠ Decide**

- **Signal** = Model output (direction + strength + reason)
- **OrderIntent** = Execution action (target position + sizing)

This separation keeps forecasting logic (strategy) decoupled from execution logic (engine). Position sizing, risk gates, and capital allocation happen in the engine, not the strategy.

## Architecture

### Strategy Protocol

All strategies implement the `Strategy` protocol defined in `src/strategy/base.py`:

```python
from typing import Protocol, Sequence
from src.strategy.base import StrategyContext
from src.strategy.signal import Signal

class Strategy(Protocol):
    @property
    def name(self) -> str:
        """Unique strategy identifier (used in run metadata)."""
        ...

    @property
    def required_features(self) -> Sequence[str]:
        """List of required feature column names."""
        ...

    @property
    def warmup_bars(self) -> int:
        """Minimum bars needed before strategy can trade."""
        ...

    def can_trade(self, ctx: StrategyContext) -> bool:
        """Gating logic for warmup / missing features / NaNs."""
        ...

    def on_bar(self, ctx: StrategyContext) -> Signal:
        """Produces a Signal for the current bar."""
        ...
```

**Contract:**
- `name`: Must be unique and deterministic (used in run artifacts)
- `required_features`: Must list all feature names used by `on_bar()`
- `warmup_bars`: Must cover max lookback of required features (e.g., SMA(50) → warmup_bars=50)
- `can_trade()`: Must return `False` during warmup or if features are NaN
- `on_bar()`: Must return a `Signal` object (never `None`)

### StrategyContext

Provides per-bar state to the strategy:

```python
@dataclass(frozen=True)
class StrategyContext:
    ts: pd.Timestamp           # Current bar timestamp
    bar: pd.Series             # OHLCV data (open, high, low, close, volume)
    features: pd.Series        # Computed feature values at this timestamp
    position: float            # Current signed position (+1.0 long, -1.0 short, 0 flat)
    equity: float              # Current account equity
    bar_index: int             # Sequential bar counter (0-indexed)
```

**Usage:**
```python
def on_bar(self, ctx: StrategyContext) -> Signal:
    close_price = ctx.bar["close"]
    sma_10 = ctx.features["sma_10"]
    current_position = ctx.position
    # ... strategy logic
```

### Signal

Represents the model's directional belief:

```python
@dataclass(frozen=True)
class Signal:
    direction: float           # +1.0 long, -1.0 short, 0.0 flat
    strength: float = 1.0      # 0.0-1.0 confidence (for future sizing)
    reason: str = ""           # Human-readable audit trail

    # Factory methods
    @staticmethod
    def long(strength: float = 1.0, reason: str = "") -> Signal:
        return Signal(direction=1.0, strength=strength, reason=reason)

    @staticmethod
    def short(strength: float = 1.0, reason: str = "") -> Signal:
        return Signal(direction=-1.0, strength=strength, reason=reason)

    @staticmethod
    def flat(reason: str = "") -> Signal:
        return Signal(direction=0.0, strength=0.0, reason=reason)
```

**Key design:** `Signal` represents belief, not action. The `TradingEngine` decision layer converts `Signal` → `OrderIntent` based on:
- Current position (avoid redundant signals)
- Risk gates (drawdown halt, daily/monthly limits)
- Position sizing rules (fixed size, probability-scaled, etc.)

### Feature Validation

Helper function for validating required features:

```python
from src.strategy.base import validate_features

def can_trade(self, ctx: StrategyContext) -> bool:
    return validate_features(ctx.features, self.required_features)
```

**Logic:**
- Returns `False` if any required feature is missing or NaN
- Returns `True` if all required features exist and are valid

## Registry Pattern

Strategies are registered in `src/strategy/registry.py` using a builder pattern:

```python
from src.strategy.registry import register

def build_my_strategy(cfg: dict) -> tuple[Strategy, list[FeatureSpec]]:
    """Builder function for MyStrategy.

    Parameters
    ----------
    cfg : dict
        Strategy config block (from YAML, minus 'type' key).

    Returns
    -------
    tuple of (Strategy, list[FeatureSpec])
        Strategy instance + feature specs required by this strategy.
    """
    # Extract params from config
    param1 = cfg.get("param1", default_value)
    param2 = cfg["param2"]  # required param

    # Build feature specs (if any)
    feature_specs = [
        FeatureSpec(indicator=SMA(period=10), transforms=[]),
        FeatureSpec(indicator=EMA(period=20), transforms=[ZScoreRolling(window=50)]),
    ]

    # Build strategy instance
    strategy = MyStrategy(param1=param1, param2=param2)

    return strategy, feature_specs

# Auto-register on import
register("my_strategy", build_my_strategy)
```

**Usage in config:**
```yaml
strategy:
  type: my_strategy       # Maps to registered builder
  param1: 0.5
  param2: 100
```

**Auto-registration:** Built-in strategies are auto-registered on module import (see bottom of `registry.py`).

## Built-in Strategies

### 1. MA Crossover (`ma_crossover`)

Simple moving average crossover strategy.

**Logic:**
- Long when fast MA > slow MA
- Short when fast MA < slow MA

**Config:**
```yaml
strategy:
  type: ma_crossover
  fast_period: 5            # Fast MA period
  slow_period: 10           # Slow MA period
  indicator_type: ema       # ema or sma
```

**File:** `ma_crossover.py`

### 2. ADX Filtered Crossover (`adx_filtered`)

MA crossover with ADX trend filter.

**Logic:**
- Enter MA crossover signals only when ADX > threshold (strong trend)
- Exit when ADX < threshold (weak trend)

**Config:**
```yaml
strategy:
  type: adx_filtered
  fast_period: 5
  slow_period: 10
  adx_period: 14
  adx_threshold: 25.0       # Minimum ADX for trading
  indicator_type: ema
```

**File:** `adx_filtered_crossover.py`

### 3. Donchian + ATR + HMM Regime (`donchian_atr_regime`)

Donchian breakout with ATR-based stops, gated by HMM regime detection.

**Logic:**
- Enter long on Donchian high breakout (only in TREND regime)
- Exit on Donchian low breakout or ATR-based stop
- HMM classifies bars as TREND or RANGE based on Efficiency Ratio, Choppiness, Hurst

**Config:**
```yaml
strategy:
  type: donchian_atr_regime
  donchian_period: 20
  atr_period: 14
  atr_multiplier: 2.0       # ATR stop distance
  regime_lookback: 1000     # HMM feature lookback
  refit_every: 500          # HMM refit frequency
  predict_every: 10         # HMM prediction cache interval
```

**File:** `donchian_atr_regime.py`

### 4. Lorentzian Classification (`lorentzian_classification`)

K-nearest neighbors classification using Lorentzian distance metric.

**Logic:**
- Classify current bar based on historical neighbors (KNN with Lorentzian distance)
- Enter long if majority of neighbors had profitable long moves

**Config:**
```yaml
strategy:
  type: lorentzian_classification
  lookback: 2000
  k: 8                      # Number of neighbors
```

**File:** `lorentzian_classification.py`

### 5. RSI + Bollinger Band Confluence (`rsi_bb_confluence`)

Oversold/overbought confluence strategy.

**Logic:**
- Long when RSI < 30 AND price < BB lower
- Short when RSI > 70 AND price > BB upper

**Config:**
```yaml
strategy:
  type: rsi_bb_confluence
  rsi_period: 14
  bb_period: 20
  bb_std: 2.0
  rsi_oversold: 30
  rsi_overbought: 70
```

**File:** `rsi_bb_confluence.py`

### 6. Bollinger Band Mean Reversion (`bb_mean_reversion`)

Mean reversion strategy based on Bollinger Bands.

**Logic:**
- Long when price touches lower band (expecting reversion to mean)
- Exit when price crosses middle band

**Config:**
```yaml
strategy:
  type: bb_mean_reversion
  bb_period: 20
  bb_std: 2.0
```

**File:** `bb_mean_reversion.py`

### 7. Graph MSS (Market Structure Shift) (`graph_mss`)

Market structure shift detection using zigzag + swing high/low analysis.

**Logic:**
- Detect bullish/bearish market structure shifts
- Enter on confirmed structure shift

**Config:**
```yaml
strategy:
  type: graph_mss
  zigzag_threshold: 0.001   # Zigzag reversal threshold
  lookback: 500
```

**File:** `graph_mss.py`

### 8. ML Probability Threshold (`ml_prob_threshold`)

Machine learning strategy with probability-based entry/exit and optional probability-sized positions.

**Logic:**
- Train model (Logistic/XGBoost/LightGBM) on walk-forward folds
- Enter when P(up) > threshold
- Size position by probability (if probability-sized policy enabled)

**Config (Fixed Threshold):**
```yaml
strategy:
  type: ml_prob_threshold
  p_enter: 0.60
  p_exit: 0.45
  min_hold_bars: 12
  features:
    - type: ema
      period: 5
      transforms:
        - type: zscore
          window: 20
    - type: rsi
      period: 14
```

**Config (Probability-Sized Policy):**
```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.40
    p_exit: 0.30
    p_full: 0.50
    min_size: 0.25
    max_size: 1.0
    min_hold_bars: 5
  features:
    # ... (same as above)
```

**File:** `ml_prob_threshold.py`

## Implementing a Custom Strategy

### Step-by-Step Guide

**Step 1: Implement the Strategy Protocol**

Create a new file in `src/strategy/` (e.g., `my_custom_strategy.py`):

```python
from dataclasses import dataclass
from typing import Sequence
import pandas as pd

from src.strategy.base import StrategyContext, validate_features
from src.strategy.signal import Signal

@dataclass
class MyCustomStrategy:
    """Simple threshold-based spread strategy."""
    threshold: float = 0.02    # Spread threshold (2%)

    @property
    def name(self) -> str:
        return f"my_custom_th{self.threshold}"

    @property
    def required_features(self) -> Sequence[str]:
        return ["sma_10", "sma_50"]

    @property
    def warmup_bars(self) -> int:
        return 50  # Max lookback of required features

    def can_trade(self, ctx: StrategyContext) -> bool:
        # Validate required features exist and are not NaN
        return validate_features(ctx.features, self.required_features)

    def on_bar(self, ctx: StrategyContext) -> Signal:
        if not self.can_trade(ctx):
            return Signal.flat("warmup or NaN features")

        sma10 = ctx.features["sma_10"]
        sma50 = ctx.features["sma_50"]
        spread = (sma10 - sma50) / sma50

        if spread > self.threshold:
            return Signal.long(strength=1.0, reason=f"SMA10 > SMA50 by {spread:.4f}")
        elif spread < -self.threshold:
            return Signal.short(strength=1.0, reason=f"SMA10 < SMA50 by {spread:.4f}")
        else:
            return Signal.flat("neutral zone")
```

**Step 2: Create Builder Function**

Add builder function to the same file:

```python
from src.indicators.core.pipeline import FeatureSpec
from src.indicators.impl.moving_averages import SMA

def build(cfg: dict) -> tuple[MyCustomStrategy, list[FeatureSpec]]:
    """Builder for MyCustomStrategy."""
    threshold = cfg.get("threshold", 0.02)

    # Define required features
    feature_specs = [
        FeatureSpec(indicator=SMA(period=10), transforms=[]),
        FeatureSpec(indicator=SMA(period=50), transforms=[]),
    ]

    strategy = MyCustomStrategy(threshold=threshold)
    return strategy, feature_specs
```

**Step 3: Register the Strategy**

Add to `src/strategy/registry.py`:

```python
from .my_custom_strategy import build as _build_my_custom

register("my_custom", _build_my_custom)
```

**Step 4: Create Config**

Create `configs/my_custom_test.yaml`:

```yaml
symbol: "XAUUSD"
timeframe: "M5"
snapshot_dir: "data/20260217_030324"
starting_capital: 1000.0
output_dir: "runs"

strategy:
  type: my_custom          # Registered name
  threshold: 0.025         # Custom parameter

execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01
```

**Step 5: Test**

```bash
uv run python -m src backtest --config configs/my_custom_test.yaml
```

Check results in `runs/<timestamp>/`.

### Advanced: Extra Artifacts Protocol

Strategies can optionally export extra artifacts (e.g., regime assignments, internal state) by implementing:

```python
def extra_artifacts(self) -> dict[str, pd.DataFrame]:
    """Return dict of {filename: dataframe} for extra artifacts."""
    return {
        "my_internal_state.csv": self._state_history,
    }
```

The runner will write these to the run directory automatically.

**Example:** `DonchianATRRegime` exports `regime.csv` with HMM regime assignments.

## Integration with Engine

### Per-Bar Pipeline

The `TradingEngine` calls strategies in this order:

1. **Execution (at bar open):** Fill previous bar's OrderIntent
2. **Mark-to-Market (at bar close):** Compute unrealized P&L, update equity
3. **Risk check:** RiskManager.update() → RiskAction
4. **Record:** Equity snapshot appended
5. **Model → Signal:** `strategy.on_bar(ctx)` → Signal
6. **Decision:** Signal + RiskAction → OrderIntent (sizing + risk gates)
7. **Store intent:** Pending for next bar's execution

**Key:** Strategies see current bar's close price and features, but OrderIntent is executed at next bar's open price (no lookahead).

### Signal → OrderIntent Conversion

The decision layer (`TradingEngine._decide()`) converts Signal to OrderIntent:

```python
def _decide(signal: Signal, position: float, risk_action: RiskAction) -> OrderIntent:
    if risk_action.halted or risk_action.flatten:
        # Risk override: flatten position
        return OrderIntent(target_position=0.0, reason=risk_action.reason)

    # Convert signal direction to target position
    target = signal.direction  # +1.0 long, -1.0 short, 0.0 flat

    # Avoid redundant signals (already at target position)
    if target == position:
        return OrderIntent(target_position=position, reason="hold")

    # Size by signal strength (future: probability-scaled sizing)
    sized_target = target * signal.strength

    return OrderIntent(target_position=sized_target, reason=signal.reason)
```

**Future enhancement:** Probability-scaled sizing will use `signal.strength` to interpolate position size.

## Config Examples

### Simple Rule-Based

```yaml
strategy:
  type: ma_crossover
  fast_period: 5
  slow_period: 10
  indicator_type: ema
```

### Rule-Based with Filter

```yaml
strategy:
  type: adx_filtered
  fast_period: 5
  slow_period: 10
  adx_period: 14
  adx_threshold: 25.0
  indicator_type: ema
```

### HMM Regime-Gated

```yaml
strategy:
  type: donchian_atr_regime
  donchian_period: 20
  atr_period: 14
  atr_multiplier: 2.0
  regime_lookback: 1000
  refit_every: 500
  predict_every: 10
```

### ML with Probability Sizing

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.40
    p_exit: 0.30
    p_full: 0.50
    min_size: 0.25
    max_size: 1.0
    min_hold_bars: 5
  features:
    - type: ema
      period: 5
      transforms:
        - type: zscore
          window: 20
    - type: rsi
      period: 14
```

## Key Files

| File | Description |
|------|-------------|
| `base.py` | Strategy protocol, StrategyContext, validate_features() |
| `signal.py` | Signal dataclass + factory methods |
| `registry.py` | Strategy registry + build_strategy() |
| `ma_crossover.py` | Simple MA crossover strategy |
| `adx_filtered_crossover.py` | MA crossover with ADX filter |
| `donchian_atr_regime.py` | Donchian + ATR + HMM regime |
| `lorentzian_classification.py` | KNN with Lorentzian distance |
| `rsi_bb_confluence.py` | RSI + Bollinger Band confluence |
| `bb_mean_reversion.py` | Bollinger Band mean reversion |
| `graph_mss.py` | Market structure shift detection |
| `ml_prob_threshold.py` | ML strategy with probability threshold |

## See Also

- [ML Pipeline](../ml/README.md) — How ML strategies integrate with walk-forward cross-validation
- [Backtest Engine](../engine/README.md) — How TradingEngine orchestrates strategy execution
- [Feature Engineering](../indicators/README.md) — Building features for strategies
- [Risk Management](../risk/README.md) — How RiskAction overrides strategy signals
- [Execution Layer](../execution/README.md) — How OrderIntent becomes Fill
