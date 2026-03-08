# Regime Detection

**HMM-based trend/range classification for regime-gated trading strategies.**

## Overview

The `src/regime/` package implements market regime detection using Hidden Markov Models (HMM) and other statistical methods. The primary use case is **regime-gated strategies** that only trade in favorable market conditions (e.g., breakout strategies that only enter in trending regimes).

**Core principle:** Markets alternate between **TREND** (directional movement) and **RANGE** (sideways chop) regimes. Regime-aware strategies avoid trading during unfavorable conditions, reducing whipsaws and drawdowns.

**Key output:** `RegimeLabel` with:
- `regime`: "TREND" or "RANGE"
- `confidence`: 0.0–1.0 (probability of assigned regime)
- `raw_value`: Underlying numeric (e.g., Hurst exponent, HMM posterior probability)

## Architecture

### RegimeDetector Protocol

All regime detectors implement the `RegimeDetector` protocol:

```python
from typing import Protocol
from src.regime.protocol import RegimeLabel

class RegimeDetector(Protocol):
    name: str               # Detector identifier
    warmup_bars: int        # Minimum bars before detection is valid

    def update(
        self,
        close: float,
        high: float,
        low: float,
        bar_index: int,
    ) -> RegimeLabel:
        """Consume one bar and return a regime label."""
        ...
```

**Contract:**
- `update()` is called per bar in sequential order
- Returns `RegimeLabel` every bar (even during warmup)
- During warmup, confidence should be 0.0 or regime should be neutral

### RegimeLabel

```python
@dataclass(frozen=True)
class RegimeLabel:
    regime: str        # "TREND" or "RANGE"
    confidence: float  # 0.0–1.0 (probability of assigned regime)
    raw_value: float   # Underlying numeric (detector-specific)
```

**Example:**
```python
RegimeLabel(regime="TREND", confidence=0.85, raw_value=0.72)
# Interpretation: 85% confident this is a TREND regime, HMM posterior = 0.72
```

## HMM-Based Detection

The primary implementation is `TrendRangeFilter` using a 2-state Gaussian HMM trained on three features:

1. **Efficiency Ratio (ER)**: `|net_change| / sum_abs_change` (1.0 = perfect trend, 0.0 = random walk)
2. **Choppiness Index (CHOP)**: `log(ATR_sum / (high_max - low_min)) / log(period)` (high = choppy, low = trending)
3. **Hurst Exponent (H)**: Rescaled range analysis (0.5 = random walk, >0.5 = trending, <0.5 = mean-reverting)

### Feature Engineering Pipeline

**Bars → Raw Features → Z-Score Normalization → HMM**

```python
# 1. Compute raw features
ER = |close[t] - close[t-period]| / sum(|close[i] - close[i-1]|)
CHOP = log(sum(TR)) / (max(high) - min(low)) / log(period)
H = rescaled_range_analysis(close, period)

# 2. Z-score normalize over rolling window
ER_z = (ER - rolling_mean(ER)) / rolling_std(ER)
CHOP_z = (CHOP - rolling_mean(CHOP)) / rolling_std(CHOP)
H_z = (H - rolling_mean(H)) / rolling_std(H)

# 3. HMM training on [ER_z, CHOP_z, H_z] matrix
hmm.fit(X)  # X shape: (n_bars, 3)
```

### State Identification

After training, the HMM produces 2 states (0 and 1). To determine which state corresponds to TREND:

**Heuristic:** The state with higher mean Efficiency Ratio is the TREND state.

```python
trend_state = argmax(hmm.means_[:, ER_index])
```

**Logic:** Trending markets have higher ER (larger net change relative to path length).

### Refit & Prediction Caching

**Refit every N bars:** HMM is retrained every `refit_every` bars to adapt to changing market dynamics.

**Prediction caching:** HMM predictions are cached for `predict_every` bars to avoid recomputation.

**Why cache?**
- HMM forward-backward algorithm is expensive on large datasets
- Regime assignments are relatively stable (don't need per-bar updates)
- `predict_every=10` (default) provides good balance between freshness and performance

**Performance tip:** For 635k bars, `refit_every=500` and `predict_every=10` are practical. Lower values (e.g., `refit_every=50`) cause excessive retraining overhead.

### TrendRangeFilter Implementation

```python
from src.regime.filter import TrendRangeFilter

detector = TrendRangeFilter(
    lookback=1000,          # Feature computation window
    refit_every=500,        # HMM refit frequency
    predict_every=10,       # Prediction cache interval
    random_state=42,        # Random seed
)

# Per-bar update
label = detector.update(close=2045.3, high=2046.1, low=2044.8, bar_index=1234)

# label.regime: "TREND" or "RANGE"
# label.confidence: 0.0–1.0 (HMM posterior probability)
# label.raw_value: HMM posterior for trend state
```

**Internal flow:**

1. **Bar 0–999 (warmup):** Build feature buffer, return neutral label
2. **Bar 1000 (first refit):** Compute features, z-score normalize, fit HMM, predict first batch
3. **Bar 1001–1009:** Return cached predictions
4. **Bar 1010:** Predict next batch (no refit)
5. **Bar 1500:** Refit HMM (500 bars since last fit), predict new batch
6. Repeat...

## Integration with Strategies

### DonchianATRRegime Example

Donchian breakout strategy that only enters in TREND regimes:

```python
from src.regime.filter import TrendRangeFilter
from src.regime.protocol import RegimeLabel

class DonchianATRRegime:
    def __init__(
        self,
        donchian_period: int = 20,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        regime_lookback: int = 1000,
        refit_every: int = 500,
        predict_every: int = 10,
    ):
        self.regime_filter = TrendRangeFilter(
            lookback=regime_lookback,
            refit_every=refit_every,
            predict_every=predict_every,
        )
        # ... (Donchian + ATR logic)

    def on_bar(self, ctx: StrategyContext) -> Signal:
        # Get regime label
        regime_label = self.regime_filter.update(
            close=ctx.bar["close"],
            high=ctx.bar["high"],
            low=ctx.bar["low"],
            bar_index=ctx.bar_index,
        )

        # Only trade in TREND regime
        if regime_label.regime != "TREND":
            return Signal.flat(f"Range regime (conf={regime_label.confidence:.2f})")

        # Donchian breakout logic (only executed in TREND)
        donchian_high = ctx.features["donchian_high_20"]
        if ctx.bar["close"] > donchian_high:
            return Signal.long(
                strength=1.0,
                reason=f"Donchian breakout in TREND (conf={regime_label.confidence:.2f})"
            )

        # ... (exit logic)
```

**Key advantage:** Avoids whipsaws in range-bound markets. Breakout strategies suffer from false signals in chop; regime gating filters them out.

## Extra Artifacts: regime.csv

Strategies using regime detection can export regime assignments via the `extra_artifacts()` protocol:

```python
def extra_artifacts(self) -> dict[str, pd.DataFrame]:
    """Export regime history for robustness analysis."""
    return {"regime.csv": self._regime_history}
```

**regime.csv format:**

```csv
time,regime,confidence,raw_value
2024-01-15 09:00:00,TREND,0.85,0.72
2024-01-15 09:05:00,TREND,0.82,0.68
2024-01-15 09:10:00,RANGE,0.65,-0.23
...
```

**Usage:**
- Robustness testing (`regime_summary.csv` slices performance by regime)
- Post-hoc analysis (regime transitions, confidence distributions)
- Visualization (plot regime assignments over price)

See [Robustness Testing](../robustness/README.md) for regime-stratified performance analysis.

## Other Regime Detectors

### Choppiness Index (Simple Threshold)

```python
from src.regime.choppiness import ChoppinessDetector

detector = ChoppinessDetector(
    period=14,
    threshold=61.8,  # Typical threshold (range: 38.2–61.8 = chop)
)

label = detector.update(close, high, low, bar_index)
# regime = "RANGE" if CHOP > 61.8 else "TREND"
```

**Pros:** Simple, interpretable
**Cons:** Fixed threshold may not adapt to changing volatility

### Hurst Exponent (Simple Threshold)

```python
from src.regime.hurst import HurstDetector

detector = HurstDetector(
    period=100,
    threshold=0.5,  # H > 0.5 = trending, H < 0.5 = mean-reverting
)

label = detector.update(close, high, low, bar_index)
# regime = "TREND" if H > 0.5 else "RANGE"
```

**Pros:** Directly measures long-term memory
**Cons:** Computationally expensive, sensitive to period choice

### Gaussian Mixture Model (GMM)

```python
from src.regime.gmm import GMMRegimeDetector

detector = GMMRegimeDetector(
    lookback=1000,
    refit_every=500,
)
```

**Pros:** No hidden states assumption (unlike HMM)
**Cons:** Less principled for time series (HMM models temporal transitions)

### K-Means Clustering

```python
from src.regime.kmeans import KMeansRegimeDetector

detector = KMeansRegimeDetector(
    lookback=1000,
    refit_every=500,
)
```

**Pros:** Fast, simple
**Cons:** Ignores temporal structure

### PELT (Change Point Detection)

```python
from src.regime.pelt import PELTDetector

detector = PELTDetector(
    lookback=1000,
    penalty=10.0,  # Change point detection penalty
)
```

**Pros:** Detects structural breaks
**Cons:** Binary output (change vs no change), not regime classification

**Note:** HMM is recommended for most use cases due to principled temporal modeling and probabilistic confidence estimates.

## Config Examples

### HMM Regime-Gated Strategy

```yaml
strategy:
  type: donchian_atr_regime
  donchian_period: 20
  atr_period: 14
  atr_multiplier: 2.0
  regime_lookback: 1000     # HMM feature window
  refit_every: 500          # HMM refit frequency
  predict_every: 10         # HMM prediction cache interval
```

### Choppiness Detector

```yaml
strategy:
  type: my_custom_regime_strategy
  regime_detector: choppiness
  chop_period: 14
  chop_threshold: 61.8
```

### Hurst Detector

```yaml
strategy:
  type: my_custom_regime_strategy
  regime_detector: hurst
  hurst_period: 100
  hurst_threshold: 0.5
```

## Extending: Adding a New Detector

**Step 1:** Implement the `RegimeDetector` protocol in `src/regime/my_detector.py`:

```python
from dataclasses import dataclass
from collections import deque
from src.regime.protocol import RegimeLabel, RegimeDetector

@dataclass
class MyCustomDetector:
    period: int = 50
    threshold: float = 0.5

    def __post_init__(self):
        self._buffer = deque(maxlen=self.period)

    @property
    def name(self) -> str:
        return f"my_detector_{self.period}"

    @property
    def warmup_bars(self) -> int:
        return self.period

    def update(
        self,
        close: float,
        high: float,
        low: float,
        bar_index: int,
    ) -> RegimeLabel:
        self._buffer.append(close)

        if len(self._buffer) < self.period:
            # Warmup period
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)

        # Custom regime logic
        raw_value = my_custom_metric(self._buffer)
        regime = "TREND" if raw_value > self.threshold else "RANGE"
        confidence = abs(raw_value - self.threshold) / self.threshold

        return RegimeLabel(regime=regime, confidence=confidence, raw_value=raw_value)
```

**Step 2:** Use in strategy:

```python
from src.regime.my_detector import MyCustomDetector

class MyStrategy:
    def __init__(self):
        self.regime_detector = MyCustomDetector(period=50, threshold=0.5)

    def on_bar(self, ctx: StrategyContext) -> Signal:
        regime = self.regime_detector.update(
            close=ctx.bar["close"],
            high=ctx.bar["high"],
            low=ctx.bar["low"],
            bar_index=ctx.bar_index,
        )

        if regime.regime != "TREND":
            return Signal.flat("Not trending")

        # ... strategy logic
```

## Performance Notes

### HMM Tuning for Large Datasets

For 635k bars:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `lookback` | 1000 | Enough history for stable feature computation (not too long to avoid stale data) |
| `refit_every` | 500 | Balance between adaptation and computation cost |
| `predict_every` | 10 | Cache predictions to avoid redundant forward-backward passes |

**Too frequent refit:** `refit_every=50` causes excessive retraining (635k / 50 = 12,700 HMM fits!)

**Too rare refit:** `refit_every=5000` risks stale model that doesn't adapt to regime shifts

### Fresh Model per Refit

**Critical implementation detail:**

```python
def fit(self, X: np.ndarray) -> None:
    # Create fresh GaussianHMM each time
    self._model = GaussianHMM(
        n_components=self._model.n_components,
        covariance_type=self._model.covariance_type,
        n_iter=self._model.n_iter,
        random_state=self._model.random_state,
    )
    self._model.fit(X)
```

**Why?** Reusing the same HMM instance across refits can lead to degenerate covariance matrices (numerical instability). Fresh model per refit avoids this.

## Key Files

| File | Description |
|------|-------------|
| `protocol.py` | RegimeDetector protocol, RegimeLabel dataclass |
| `filter.py` | TrendRangeFilter (HMM-based detector) |
| `hmm_model.py` | HMMTrendRangeModel wrapper (2-state Gaussian HMM) |
| `adapter_hmm.py` | HMM adapter logic (state identification, posterior mapping) |
| `features.py` | Feature computation (ER, CHOP, Hurst) |
| `choppiness.py` | Simple Choppiness Index detector |
| `hurst.py` | Simple Hurst Exponent detector |
| `gmm.py` | Gaussian Mixture Model detector |
| `kmeans.py` | K-Means clustering detector |
| `pelt.py` | PELT change point detector |

## See Also

- [Strategy System](../strategy/README.md) — How strategies integrate regime detection
- [Robustness Testing](../robustness/README.md) — Regime-stratified performance analysis
- [Feature Engineering](../indicators/README.md) — Feature computation for HMM
- [Backtest Engine](../engine/README.md) — How regime records are exported
