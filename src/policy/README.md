# Position Sizing Policies

**Uncertainty-aware decision layer: probability → signal strength → position size.**

## Overview

The `src/policy/` package implements the policy layer for ML strategies. Policies convert calibrated probabilities into trading signals with appropriate position sizing.

**Core workflow:**
```
Calibrated Probability → Policy.decide() → Signal(direction, strength, reason)
```

**Key principle:** Position sizing should reflect model uncertainty. High-confidence predictions get larger positions; marginal predictions get smaller positions.

## Policy Protocol

```python
from typing import Protocol
from src.strategy.signal import Signal

class Policy(Protocol):
    def decide(
        self,
        p_pred: float,      # Predicted probability (calibrated)
        position: float,    # Current position (-1.0 to +1.0)
        bars_held: int,     # Bars since position opened
    ) -> Signal:
        """Convert probability + state → Signal."""
        ...
```

**Returns:** `Signal(direction, strength, reason)`

**Integration:** ML strategies (e.g., `MLProbThresholdStrategy`) call `policy.decide()` in `on_bar()`.

## Built-in Policies

### 1. Fixed Threshold Policy

Simple binary threshold with minimum hold period.

**Logic:**
- Enter long if `p > p_enter`
- Exit if `p < p_exit`
- Hold for at least `min_hold_bars` (no exit before min hold)

**Position size:** Fixed at 1.0 (full position). Sizing happens in `TradingEngine`, not policy.

**Config:**
```yaml
strategy:
  type: ml_prob_threshold
  p_enter: 0.60             # Enter threshold
  p_exit: 0.45              # Exit threshold
  min_hold_bars: 12         # Minimum hold period
```

**Implementation:**
```python
@dataclass
class FixedThresholdPolicy:
    p_enter: float = 0.60
    p_exit: float = 0.45
    min_hold_bars: int = 12

    def decide(self, p_pred: float, position: float, bars_held: int) -> Signal:
        # Hold minimum bars
        if position != 0 and bars_held < self.min_hold_bars:
            return Signal.flat(f"Hold min {self.min_hold_bars} bars (held {bars_held})")

        # Entry logic
        if position == 0 and p_pred > self.p_enter:
            return Signal.long(strength=1.0, reason=f"P(up)={p_pred:.3f} > {self.p_enter}")

        # Exit logic
        if position != 0 and p_pred < self.p_exit:
            return Signal.flat(reason=f"P(up)={p_pred:.3f} < {self.p_exit}")

        # Hold
        return Signal.flat(reason="neutral zone")
```

**Pros:** Simple, interpretable
**Cons:** Binary all-or-nothing (ignores model confidence)

### 2. Probability-Sized Policy

**Scales position size linearly by model confidence.**

**Logic:**
- Enter if `p > p_enter`
- Exit if `p < p_exit`
- Position size scales linearly between `p_enter` (min_size) and `p_full` (max_size)

**Config:**
```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.40           # Entry threshold
    p_exit: 0.30            # Exit threshold
    p_full: 0.50            # Full position threshold
    min_size: 0.25          # Minimum position size (at p_enter)
    max_size: 1.0           # Maximum position size (at p_full)
    min_hold_bars: 5        # Minimum hold period
```

**Sizing formula:**
```python
if p >= p_full:
    size = max_size         # Full position
elif p >= p_enter:
    # Linear interpolation
    size = min_size + (max_size - min_size) * (p - p_enter) / (p_full - p_enter)
else:
    size = 0.0              # Flat
```

**Example:**
- `p_enter = 0.40`, `p_full = 0.50`, `min_size = 0.25`, `max_size = 1.0`

| P(up) | Position Size | Interpretation |
|-------|---------------|----------------|
| 0.35  | 0.00          | Below entry threshold |
| 0.40  | 0.25          | Minimum position (cautious entry) |
| 0.43  | 0.475         | Interpolated (moderate confidence) |
| 0.45  | 0.625         | Interpolated |
| 0.50  | 1.00          | Full position (high confidence) |
| 0.55  | 1.00          | Full position (capped at max_size) |

**Implementation:**
```python
@dataclass
class ProbabilitySizedPolicy:
    p_enter: float = 0.40
    p_exit: float = 0.30
    p_full: float = 0.50
    min_size: float = 0.25
    max_size: float = 1.0
    min_hold_bars: int = 5

    def decide(self, p_pred: float, position: float, bars_held: int) -> Signal:
        # Hold minimum bars
        if position != 0 and bars_held < self.min_hold_bars:
            return Signal.flat(f"Hold min bars (held {bars_held})")

        # Exit logic
        if position != 0 and p_pred < self.p_exit:
            return Signal.flat(reason=f"Exit: p={p_pred:.3f} < {self.p_exit}")

        # Entry/sizing logic
        if p_pred >= self.p_full:
            size = self.max_size
        elif p_pred >= self.p_enter:
            # Linear interpolation
            size = self.min_size + (self.max_size - self.min_size) * \
                   (p_pred - self.p_enter) / (self.p_full - self.p_enter)
        else:
            size = 0.0

        if size > 0:
            return Signal.long(strength=size, reason=f"P(up)={p_pred:.3f}, size={size:.2f}")
        else:
            return Signal.flat(reason="below entry threshold")
```

**Pros:**
- Uncertainty-aware (larger positions for high confidence)
- Smoother capital allocation (no binary jumps)
- Reduces risk of marginal trades (small size for marginal probabilities)

**Cons:**
- More parameters to tune (`p_enter`, `p_full`, `min_size`, `max_size`)

## Config Examples

### Fixed Threshold (Conservative)

```yaml
strategy:
  type: ml_prob_threshold
  p_enter: 0.65             # High entry threshold (selective)
  p_exit: 0.50              # Exit at neutral
  min_hold_bars: 20         # Long hold period
  features: [...]
```

### Fixed Threshold (Aggressive)

```yaml
strategy:
  type: ml_prob_threshold
  p_enter: 0.55             # Lower entry threshold (more trades)
  p_exit: 0.40              # Hold through noise
  min_hold_bars: 5          # Short hold period
  features: [...]
```

### Probability-Sized (Balanced)

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.40           # Enter cautiously
    p_exit: 0.30            # Exit early
    p_full: 0.50            # Full position at 50%
    min_size: 0.25          # 25% min position
    max_size: 1.0           # 100% max position
    min_hold_bars: 5
  features: [...]
```

### Probability-Sized (Conservative)

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.45           # Higher entry bar
    p_exit: 0.35
    p_full: 0.60            # Require 60% for full position
    min_size: 0.15          # Very small min position
    max_size: 0.75          # Cap at 75% max position (risk control)
    min_hold_bars: 10
  features: [...]
```

## Integration with Strategies

### MLProbThresholdStrategy

```python
from src.ml.policy import FixedThresholdPolicy, ProbabilitySizedPolicy

class MLProbThresholdStrategy:
    def __init__(
        self,
        policy: FixedThresholdPolicy | ProbabilitySizedPolicy,
        features: list[str],
        model: Predictor,
    ):
        self.policy = policy
        self.features = features
        self.model = model
        self.bars_held = 0

    def on_bar(self, ctx: StrategyContext) -> Signal:
        # Get features
        X = ctx.features[self.features].values.reshape(1, -1)

        # Predict probability
        p_pred = self.model.predict_proba(X)[0]

        # Policy decides signal + sizing
        signal = self.policy.decide(p_pred, ctx.position, self.bars_held)

        # Update hold counter
        if ctx.position != 0:
            self.bars_held += 1
        else:
            self.bars_held = 0

        return signal
```

## Extending: Adding a New Policy

**Step 1:** Implement the Policy protocol in `src/ml/policy.py`:

```python
@dataclass
class MyCustomPolicy:
    threshold: float = 0.55

    def decide(self, p_pred: float, position: float, bars_held: int) -> Signal:
        # Custom logic here
        if position == 0 and p_pred > self.threshold:
            return Signal.long(strength=1.0, reason=f"Custom: p={p_pred:.3f}")
        elif position != 0 and p_pred < self.threshold:
            return Signal.flat(reason="Custom exit")
        return Signal.flat(reason="hold")
```

**Step 2:** Wire into strategy builder:

```python
# src/strategy/ml_prob_threshold.py
def build(cfg: dict) -> tuple[Strategy, list[FeatureSpec]]:
    policy_cfg = cfg.get("policy", {})
    policy_type = policy_cfg.get("type", "fixed_threshold")

    if policy_type == "fixed_threshold":
        policy = FixedThresholdPolicy(...)
    elif policy_type == "probability_sized":
        policy = ProbabilitySizedPolicy(...)
    elif policy_type == "my_custom":
        policy = MyCustomPolicy(**policy_cfg)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    strategy = MLProbThresholdStrategy(policy=policy, ...)
    return strategy, feature_specs
```

**Step 3:** Use in config:

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: my_custom
    threshold: 0.58
  features: [...]
```

## Key Files

| File | Description |
|------|-------------|
| `base.py` | Policy protocol definition |
| `fixed_threshold.py` | FixedThresholdPolicy implementation |
| `probability_sized.py` | ProbabilitySizedPolicy implementation |

## See Also

- [ML Pipeline](../ml/README.md) — How calibrated probabilities are produced
- [Strategy System](../strategy/README.md) — How ML strategies use policies
- [Backtest Engine](../engine/README.md) — How signal strength affects position sizing
