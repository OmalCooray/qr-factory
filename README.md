# qr-factory

**Adaptive, Uncertainty-Aware, Execution-Aware Intraday Trading Framework for XAUUSD (Gold)**

qr-factory is a local-first quantitative research engine designed for systematic development and evaluation of intraday trading strategies. Built as the foundation for an MSc research project (University of Moratuwa — MSc Data Science & AI), it covers the full pipeline from MT5 data ingestion through probabilistic forecasting, calibration, and execution-aware backtesting.

**Core Principles:**
- **Determinism**: Same seed + config + data = identical results, always
- **Predict ≠ Decide**: Forecasting (model beliefs) and trading decisions (execution actions) are separate concerns
- **Artifacts-First**: Every run leaves a complete audit trail (config snapshot, git hash, metrics, predictions)

**Key Features:**
- ✓ MT5 data loading with validation and versioning
- ✓ Event-driven deterministic backtesting engine
- ✓ Rule-based strategies (MA crossover, ADX filtering, Donchian breakout)
- ✓ ML strategies (Logistic Regression, XGBoost, LightGBM)
- ✓ Walk-forward cross-validation with expanding/sliding windows
- ✓ Platt scaling calibration for probability estimates
- ✓ Probability-sized position policies (scale position by model confidence)
- ✓ Risk management (global drawdown halt, daily/monthly loss limits)
- ✓ Robustness testing with promotion gates (cost stress, latency, parameter sweeps)
- ✓ MLflow integration for experiment tracking
- ✓ Git-based reproducibility (commit hash + dirty state captured)

---

## North-Star Pipeline

The full research pipeline this codebase is converging toward:

```
Data → Features → Forecaster → Calibration → Policy → Execution → Risk/Metrics
```

| Stage | Description | Package | Status |
|-------|-------------|---------|--------|
| **Data** | MT5 OHLCV ingestion, validation, bar replay | `src/loader/`, `src/replay/` | ✓ Done |
| **Features** | OHLCV-derived indicators, time-of-day encoding | `src/indicators/` | ✓ Done (basic) |
| **Forecaster** | Probabilistic models (Logistic/XGBoost/LightGBM) | `src/strategy/`, `src/ml/` | ✓ Rule + ML baselines |
| **Calibration** | Platt scaling for calibrated probabilities | `src/ml/calibration.py` | ✓ Done |
| **Policy** | Uncertainty-aware trade filtering & position sizing | `src/ml/policy.py` | ✓ Fixed + Prob-sized |
| **Execution** | OrderIntent → Fill with spread/slippage | `src/execution/` | ✓ OrderIntent/Fill done |
| **Risk** | Drawdown tracking, daily/monthly limits | `src/risk/` | ✓ Done |
| **Metrics/Tracking** | Run artifacts, MLflow, robustness reports | `src/engine/`, `src/robustness/` | ✓ Done |

**Key Design:** Strategies produce `Signal` (direction + strength + reason) representing the model's belief. The `TradingEngine` converts Signal → `OrderIntent` (target position + sizing) based on risk gates and capital allocation. This separation keeps forecasting and execution concerns decoupled.

For architecture deep-dive, see [`CLAUDE.md`](CLAUDE.md).

---

## Quick Start

### Prerequisites
- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (dependency manager)
- **MetaTrader 5** Terminal (installed & logged in)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd qr-factory

# Sync dependencies
uv sync
```

### Step 1: Load Data

Fetch M5 XAUUSD data from your MT5 terminal:

```bash
uv run loader
```

This creates a timestamped snapshot directory under `data/` (e.g., `data/20260217_030324/`) containing per-year CSV files and metadata.

### Step 2: Run Your First Backtest

Execute a simple rule-based strategy:

```bash
uv run python -m src backtest --config configs/w4_rule_baseline.yaml
```

### Step 3: View Results

Artifacts are saved to `runs/<YYYYMMDD_HHMMSS>/`:

```
runs/20260308_143022/
├── config.yaml          # Config snapshot (for reproducibility)
├── metrics.json         # Headline metrics + git hash
├── equity.csv           # Per-bar equity curve
├── trades.csv           # Completed round-trip trades
├── plots/
│   ├── close_price.png
│   └── equity_curve.png
└── README.md            # Human-readable run summary
```

**Next Steps:**
- See [Configuration Guide](#configuration-guide) to understand config structure
- See [Implementing a Custom Strategy](#implementing-a-custom-strategy) to build your own

---

## Configuration Guide

Backtests are configured via YAML files in the `configs/` directory. All configs share a common schema.

### Required Keys

| Key | Type | Description |
|-----|------|-------------|
| `symbol` | str | Market symbol (e.g., `"XAUUSD"`) |
| `timeframe` | str | Bar interval (e.g., `"M5"`) |
| `snapshot_dir` | str | Path to data snapshot (e.g., `"data/20260217_030324"`) |
| `starting_capital` | float | Initial equity in account currency |
| `output_dir` | str | Output directory for run artifacts (usually `"runs"`) |
| `strategy` | dict | Strategy configuration (see below) |
| `execution` | dict | Execution costs (spread, slippage, point value) |
| `validation` | dict | Walk-forward cross-validation settings (optional) |

### Strategy Configuration

#### Rule-Based Strategy

```yaml
strategy:
  type: ma_crossover           # Strategy type (registered in src/strategy/registry.py)
  fast_period: 5               # Fast MA period
  slow_period: 10              # Slow MA period
  indicator_type: ema          # Indicator type (ema or sma)
```

#### ML Strategy with Calibration

```yaml
strategy:
  type: ml_prob_threshold      # ML strategy with probability threshold
  p_enter: 0.60                # Entry threshold (long if P(up) > 0.60)
  p_exit: 0.45                 # Exit threshold (flatten if P(up) < 0.45)
  min_hold_bars: 12            # Minimum bars to hold position
  features:                    # List of feature specs
    - type: ema
      period: 5
      transforms:
        - type: zscore
          window: 20
    - type: rsi
      period: 14
      transforms:
        - type: zscore
          window: 20

# ML-specific blocks
label:
  type: return_sign_cost_aware # Label generation function
  horizon: 12                  # Forward-looking horizon in bars
  price_col: close             # Price column to use
  cost_buffer: 0.0005          # Transaction cost buffer (spread + slippage)
  direction: up                # Direction to predict (up or down)

model:
  type: logistic               # Model type (logistic, xgboost, lightgbm)
  C: 1.0                       # Regularization strength
  max_iter: 200                # Max iterations
  class_weight: balanced       # Class weighting
  random_state: 0              # Random seed

calibration:
  method: platt                # Calibration method (platt or isotonic)
  cal_fraction: 0.2            # Fraction of train set for calibration
```

#### Probability-Sized Policy

Instead of fixed position sizing, scale position by model confidence:

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.40              # Enter if P(up) > 0.40
    p_exit: 0.30               # Exit if P(up) < 0.30
    p_full: 0.50               # Full position (1.0) if P(up) > 0.50
    min_size: 0.25             # Minimum position size (at p_enter)
    max_size: 1.0              # Maximum position size (at p_full)
    min_hold_bars: 5           # Minimum bars to hold
  features: [...]              # Feature list as above
```

Position size scales linearly between `p_enter` and `p_full`:
- P(up) = 0.40 → size = 0.25 (min_size)
- P(up) = 0.45 → size = 0.625 (interpolated)
- P(up) ≥ 0.50 → size = 1.0 (max_size)

### Execution Configuration

```yaml
execution:
  spread_points: 40            # Bid-ask spread in points (1 point = point_value)
  slippage_points: 10          # Execution slippage in points
  point_value: 0.01            # Value of 1 point in account currency
```

For XAUUSD M5, typical values:
- Spread: 30-50 points (0.3-0.5 USD/oz)
- Slippage: 5-15 points (0.05-0.15 USD/oz)
- Point value: 0.01 (1 point = 1 cent)

### Walk-Forward Validation

```yaml
validation:
  type: walk_forward
  mode: expanding              # 'expanding' or 'sliding'
  train_size: 100000           # Bars per training fold
  test_size: 50000             # Bars per test fold
  step: 50000                  # Step size for walk-forward
  drop_last: true              # Drop incomplete final fold
```

**Expanding mode**: Training window grows over time (anchored walk-forward)
**Sliding mode**: Training window slides (fixed-size rolling window)

### Risk Management

```yaml
risk:
  global_dd_halt_pct: 10.0     # Halt trading if drawdown exceeds 10%
  daily_loss_limit: 500.0      # Pause trading if daily loss > 500 (account currency)
  monthly_loss_limit: 2000.0   # Pause trading if monthly loss > 2000
```

### Example Configurations

**1. Simple Rule-Based (configs/w4_rule_baseline.yaml)**
```yaml
symbol: "XAUUSD"
timeframe: "M5"
snapshot_dir: "data/20260217_030324"
starting_capital: 1000.0
output_dir: "runs"

strategy:
  type: ma_crossover
  fast_period: 5
  slow_period: 10
  indicator_type: ema

execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01

validation:
  type: walk_forward
  mode: expanding
  train_size: 100000
  test_size: 50000
  step: 50000
  drop_last: true
```

**2. ML with Platt Scaling (configs/w6_logistic_calibrated.yaml)**

See full config at [`configs/w6_logistic_calibrated.yaml`](configs/w6_logistic_calibrated.yaml).

**3. Robustness Campaign (configs/w8_robustness_probsized.yaml)**

Campaign manifest for stress-testing across cost variations, latency proxies, and parameter sweeps. See [`configs/w8_robustness_probsized.yaml`](configs/w8_robustness_probsized.yaml).

For all available examples, see the [`configs/`](configs/) directory.

---

## Implementing a Custom Strategy

Strategies implement the `Strategy` protocol defined in [`src/strategy/base.py`](src/strategy/base.py).

### Strategy Protocol

```python
from typing import Protocol, Sequence
from src.strategy.base import StrategyContext
from src.strategy.signal import Signal

class Strategy(Protocol):
    @property
    def name(self) -> str:
        """Unique strategy identifier"""
        ...

    @property
    def required_features(self) -> Sequence[str]:
        """List of required feature column names"""
        ...

    @property
    def warmup_bars(self) -> int:
        """Minimum bars needed before strategy can trade"""
        ...

    def can_trade(self, ctx: StrategyContext) -> bool:
        """Gating logic for warmup / missing features"""
        ...

    def on_bar(self, ctx: StrategyContext) -> Signal:
        """Produces a Signal for the current bar"""
        ...
```

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

### Signal

Represents the model's directional belief (NOT an execution instruction):

```python
@dataclass(frozen=True)
class Signal:
    direction: float           # +1.0 long, -1.0 short, 0.0 flat
    strength: float            # 0.0-1.0 confidence (for future sizing)
    reason: str                # Human-readable audit trail

# Factory methods
Signal.long(strength=1.0, reason="...")
Signal.short(strength=1.0, reason="...")
Signal.flat(reason="...")
```

The `TradingEngine` converts `Signal` → `OrderIntent` based on risk gates and position sizing rules.

### Step-by-Step Guide

**1. Implement the Protocol**

Create a new file in `src/strategy/` (e.g., `src/strategy/my_strategy.py`):

```python
from dataclasses import dataclass
from typing import Sequence
import pandas as pd

from src.strategy.base import StrategyContext, validate_features
from src.strategy.signal import Signal

@dataclass
class MyStrategy:
    """Simple threshold-based SMA spread strategy."""
    threshold: float = 0.02    # Spread threshold (2%)

    @property
    def name(self) -> str:
        return f"my_strategy_th{self.threshold}"

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

**2. Register the Strategy**

Add to `src/strategy/registry.py`:

```python
from src.strategy.my_strategy import MyStrategy

def build_my_strategy(cfg: dict) -> Strategy:
    return MyStrategy(threshold=cfg.get("threshold", 0.02))

register("my_strategy", build_my_strategy)
```

**3. Create a Config**

Create `configs/my_strategy_test.yaml`:

```yaml
symbol: "XAUUSD"
timeframe: "M5"
snapshot_dir: "data/20260217_030324"
starting_capital: 1000.0
output_dir: "runs"

strategy:
  type: my_strategy
  threshold: 0.025
  features:
    - type: sma
      period: 10
    - type: sma
      period: 50

execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01
```

**4. Test It**

```bash
uv run python -m src backtest --config configs/my_strategy_test.yaml
```

Check results in `runs/<timestamp>/`.

For more examples, see existing strategies:
- [`src/strategy/ma_crossover.py`](src/strategy/ma_crossover.py) — Simple MA crossover
- [`src/strategy/adx_filtered.py`](src/strategy/adx_filtered.py) — MA crossover with ADX filter
- [`src/strategy/donchian_atr_regime.py`](src/strategy/donchian_atr_regime.py) — HMM regime-gated Donchian breakout

---

## Features & Indicators

Features are built using the **FeaturePipeline** system, which chains **Indicators** and **Transforms**.

### Built-in Indicators

| Indicator | Description | Parameters |
|-----------|-------------|------------|
| `SMA` | Simple Moving Average | `period` |
| `EMA` | Exponential Moving Average | `period` |
| `ATR` | Average True Range | `period` |
| `RSI` | Relative Strength Index | `period` |
| `ADX` | Average Directional Index | `period` |
| `BBUpper` | Bollinger Band Upper | `period`, `num_std` |
| `BBLower` | Bollinger Band Lower | `period`, `num_std` |
| `CCI` | Commodity Channel Index | `period` |
| `DonchianHigh` | Donchian Channel High | `period` |
| `DonchianLow` | Donchian Channel Low | `period` |

### Built-in Transforms

| Transform | Description | Parameters |
|-----------|-------------|------------|
| `Diff` | First difference | `periods` |
| `Lag` | Lag by k periods | `k` |
| `ZScoreRolling` | Rolling z-score normalization | `window` |

### Config Example (Feature Composition)

```yaml
strategy:
  features:
    # Raw EMA(5)
    - type: ema
      period: 5

    # EMA(5) → Z-Score(20)
    - type: ema
      period: 5
      transforms:
        - type: zscore
          window: 20

    # RSI(14) → Diff(1) → Z-Score(20)
    - type: rsi
      period: 14
      transforms:
        - type: diff
          periods: 1
        - type: zscore
          window: 20
```

### Adding Custom Indicators

See [`src/indicators/README.md`](src/indicators/README.md) for detailed guide on implementing custom indicators and transforms.

---

## ML Pipeline & Calibration

The ML pipeline enables walk-forward cross-validation with probabilistic forecasting, calibration, and uncertainty-aware policies.

### Walk-Forward Workflow

1. **Label Generation**: Generate binary labels from future returns (cost-aware)
2. **Model Training**: Train a classifier (Logistic/XGBoost/LightGBM) per fold
3. **Calibration**: Apply Platt scaling to calibrate probabilities
4. **Policy**: Convert calibrated probabilities to trading signals
5. **Artifacts**: Save predictions, fold metrics, and evaluation summary

### Label Configuration

```yaml
label:
  type: return_sign_cost_aware  # Cost-aware binary labeling
  horizon: 12                   # Forward-looking horizon (bars)
  price_col: close              # Price column for return calculation
  cost_buffer: 0.0005           # Transaction cost buffer (50 bps)
  direction: up                 # Direction to predict (up or down)
```

The `return_sign_cost_aware` labeler creates binary labels:
- Label = 1 (profitable long) if `future_return > cost_buffer`
- Label = 0 (unprofitable long) otherwise

This ensures labels account for realistic transaction costs (spread + slippage).

### Model Types

| Type | Description | Key Parameters |
|------|-------------|----------------|
| `logistic` | Logistic Regression | `C`, `max_iter`, `class_weight` |
| `xgboost` | XGBoost Classifier | `n_estimators`, `max_depth`, `learning_rate` |
| `lightgbm` | LightGBM Classifier | `n_estimators`, `max_depth`, `learning_rate` |

Example (XGBoost):
```yaml
model:
  type: xgboost
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  random_state: 0
```

### Calibration

Platt scaling fits a logistic regression on held-out calibration data to map raw model outputs to well-calibrated probabilities.

```yaml
calibration:
  method: platt          # platt or isotonic
  cal_fraction: 0.2      # 20% of training data reserved for calibration
```

**Why calibrate?**
- Raw classifier outputs (especially from tree models) are often poorly calibrated
- Calibration ensures P(y=1|X) ≈ 0.7 means ~70% empirical positive rate
- Essential for probability-sized policies (position size scales with probability)

### Policy Layer

#### Fixed Threshold Policy

```yaml
strategy:
  type: ml_prob_threshold
  p_enter: 0.60          # Enter long if P(up) > 0.60
  p_exit: 0.45           # Exit if P(up) < 0.45
  min_hold_bars: 12      # Minimum hold period
```

#### Probability-Sized Policy

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.40        # Enter if P(up) > 0.40
    p_exit: 0.30         # Exit if P(up) < 0.30
    p_full: 0.50         # Full position (1.0) if P(up) > 0.50
    min_size: 0.25       # Minimum position size
    max_size: 1.0        # Maximum position size
    min_hold_bars: 5     # Minimum hold period
```

Position size scales linearly between `p_enter` (min_size) and `p_full` (max_size).

### ML Artifacts

ML runs produce additional artifacts in the run directory:

- `predictions.csv`: Per-bar predictions (probability, label, position) for all test folds
- `fold_metrics.json`: Per-fold training and test metrics (accuracy, F1, ROC-AUC, etc.)
- `evaluation.json`: Aggregated metrics across all folds

Example configs:
- [`configs/w6_logistic_calibrated.yaml`](configs/w6_logistic_calibrated.yaml) — Logistic regression with Platt scaling
- [`configs/w7_policy_probsized.yaml`](configs/w7_policy_probsized.yaml) — Probability-sized policy

---

## Robustness Testing & Promotion Gates

The **robustness campaign harness** stress-tests strategies across cost variations, latency proxies, and parameter sweeps. A **promotion gate** determines if a baseline is production-ready.

### Campaign Concept

A robustness campaign runs a baseline config plus multiple stress scenarios:
1. **Baseline**: The candidate strategy (e.g., `w7_policy_probsized.yaml`)
2. **Cost Stress**: 0.5x, 1.5x, 2.0x spread/slippage multipliers
3. **Latency Proxy**: 1-bar execution delay (simulates network latency)
4. **Parameter Sweeps**: Grid search over key hyperparameters (p_enter, p_full, min_hold_bars)

### Campaign Configuration

Example: [`configs/w8_robustness_probsized.yaml`](configs/w8_robustness_probsized.yaml)

```yaml
campaign_id: w8_probsized
baseline_config: configs/w7_policy_probsized.yaml
description: "Robustness campaign for probability-sized policy baseline"

scenarios:
  # Cost stress (spread/slippage multipliers)
  - family: cost_stress
    name: 1.5x_cost
    description: "1.5x baseline friction"
    overrides:
      execution:
        spread_points: 60
        slippage_points: 15

  # Latency proxy (1-bar delay)
  - family: latency_proxy
    name: 1bar_delay
    description: "1-bar execution delay"
    overrides:
      execution:
        latency_bars: 1

  # Parameter sweep (p_enter sensitivity)
  - family: parameter_sweep
    name_prefix: p_enter
    description: "Entry threshold sensitivity"
    grid:
      strategy.policy.p_enter: [0.35, 0.45]
```

### Running a Campaign

**Step 1: Dry-run (validate config)**

```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml --dry-run
```

This validates the manifest and prints the scenario matrix without running backtests.

**Step 2: Execute campaign**

```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml
```

Runs baseline + all scenarios and saves results to `runs/robustness/<campaign_id>_<timestamp>/`.

**Step 3: Finalize and evaluate**

```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml --finalize runs/robustness/w8_probsized_20260308_143022
```

Generates:
- `regime_summary.csv`: Performance by market regime (trend/range)
- `promotion_report.md`: Pass/fail verdict based on promotion gate thresholds
- `robustness_report.md`: Comprehensive stress test analysis

### Promotion Gate Logic

The promotion gate evaluates:
1. **Baseline Minimum Thresholds**: Sharpe > 0.5, Win Rate > 40%, Max DD < 15%
2. **Cost Stress Tolerance**: 1.5x cost scenario must have Sharpe > 0.3
3. **Parameter Stability**: Sweeps must not deviate >30% in Sharpe/WinRate
4. **Regime Consistency**: Both trend and range regimes must be profitable

**Output**: `promotion_report.md` with PASS/FAIL verdict and detailed diagnostics.

---

## Risk Management

The `RiskManager` enforces three levels of protection:

### Global Drawdown Halt

If equity drawdown from peak exceeds threshold, **flatten all positions and halt trading**.

```yaml
risk:
  global_dd_halt_pct: 10.0   # Halt if DD > 10%
```

Once halted, no new trades are allowed for the remainder of the backtest.

### Daily Loss Limit

If realized P&L for the current day exceeds threshold, **pause trading until next day**.

```yaml
risk:
  daily_loss_limit: 500.0    # Pause if daily loss > 500
```

### Monthly Loss Limit

If realized P&L for the current month exceeds threshold, **pause trading until next month**.

```yaml
risk:
  monthly_loss_limit: 2000.0 # Pause if monthly loss > 2000
```

### Execution Order

Risk checks run **after MTM but before strategy signal**:

```
1. Execute pending orders (at bar open)
2. Mark-to-market (at bar close)
3. Risk check → RiskAction (flatten / halt / pass)
4. Record equity
5. Strategy signal (if not halted)
6. Decision layer (Signal → OrderIntent)
```

This ensures risk limits are checked before new trades can be generated.

---

## Testing

The test suite covers data validation, feature determinism, strategy logic, risk guards, ML pipeline integrity, and end-to-end reproducibility.

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_feature_pipeline.py

# Run by keyword
uv run pytest -k determinism
```

### Test Coverage

| Area | Coverage |
|------|----------|
| **Data Validation** | OHLC integrity, timestamp monotonicity, duplicate removal |
| **Feature Determinism** | Reproducible computation, column sorting, NaN handling |
| **Strategy Logic** | Unit tests for each strategy implementation |
| **Risk Guards** | Drawdown halt, daily/monthly limits, circuit breakers |
| **ML Pipeline** | Walk-forward splits, calibration correctness, label generation |
| **End-to-End Determinism** | Same config + seed → identical equity curve |

**All tests must pass** before commits. See [`tests/README.md`](tests/README.md) for detailed test documentation.

---

## Artifacts & Reproducibility

Every run generates a complete audit trail for reproducibility.

### Run Directory Structure

```
runs/<YYYYMMDD_HHMMSS>/
├── config.yaml               # Config snapshot (exact state at run time)
├── metrics.json              # Headline metrics + git metadata
├── equity.csv                # Per-bar equity curve
├── trades.csv                # Completed round-trip trades
├── fold_metrics.json         # Walk-forward per-fold metrics (ML only)
├── predictions.csv           # ML predictions on test folds (ML only)
├── evaluation.json           # Aggregated ML metrics (ML only)
├── regime.csv                # HMM regime assignments (if applicable)
├── README.md                 # Human-readable run summary
├── plots/
│   ├── close_price.png
│   └── equity_curve.png
└── DATA_REF.json             # Pointer to data snapshot used
```

### Git Hash Tracking

`metrics.json` captures git state for full reproducibility:

```json
{
  "net_pnl": 1234.56,
  "sharpe_ratio": 1.23,
  "max_drawdown_pct": 8.5,
  "git_commit": "a3f2c9d",
  "git_dirty": false,
  "run_timestamp": "2026-03-08T14:30:22Z"
}
```

### Reproducing a Run

1. **Checkout the git commit**: `git checkout <commit_hash>`
2. **Use the exact config**: `runs/<timestamp>/config.yaml`
3. **Re-run**: `uv run python -m src backtest --config runs/<timestamp>/config.yaml`
4. **Verify**: Output should match (guaranteed by determinism)

**Determinism guarantee**: Same seed + same config + same data → identical equity curve, trade list, and metrics.

---

## Comparison Tools

### Canonical Trio Comparison

Run controlled ablation study across rule-based, logistic, and LightGBM baselines:

```bash
uv run python scripts/run_canonical_trio.py
```

Output: `runs/comparisons/<timestamp>/` with side-by-side report.

This runs:
1. `configs/w4_rule_baseline.yaml` (EMA crossover)
2. `configs/w6_logistic_calibrated.yaml` (Logistic + Platt)
3. `configs/exp_ml_lgbm_baseline.yaml` (LightGBM + Platt)

Use this for controlled before/after comparisons when changing infrastructure.

### Manual Comparison

```bash
uv run python scripts/compare_runs.py runs/20260308_143022 runs/20260308_154533
```

Generates diff report highlighting metric deltas.

See [`runs/comparisons/README.md`](runs/comparisons/README.md) for canonical evidence archive.

---

## MLflow Integration

Optional experiment tracking via MLflow.

### Enable MLflow

```bash
# Install tracking dependencies
uv sync --extra tracking

# Set environment variable
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Run backtest (MLflow auto-enabled if tracking URI is set)
uv run python -m src backtest --config configs/w6_logistic_calibrated.yaml
```

### View UI

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Navigate to `http://localhost:5000` to view runs.

### Logged Artifacts

- Config (as JSON)
- Metrics (net P&L, Sharpe, max DD, win rate, etc.)
- Plots (equity curve, close price)
- Predictions (if ML run)
- Run README

---

## Project Structure

```
qr-factory/
├── configs/                  # YAML configuration files
│   ├── w4_rule_baseline.yaml
│   ├── w6_logistic_calibrated.yaml
│   └── w8_robustness_probsized.yaml
├── data/                     # Market data snapshots (see data/README.md)
├── runs/                     # Backtest artifacts
├── scripts/                  # Utility scripts
│   ├── run_ev_sim.py
│   ├── run_canonical_trio.py
│   ├── run_robustness_campaign.py
│   └── compare_runs.py
├── src/                      # Source code (see src/README.md)
│   ├── loader/               # MT5 data pipeline
│   ├── replay/               # Bar iterator and validation
│   ├── indicators/           # Feature engineering (see src/indicators/README.md)
│   ├── strategy/             # Strategy implementations
│   ├── ml/                   # ML pipeline (models, calibration, policy)
│   ├── engine/               # Backtest engine and runner
│   ├── execution/            # OrderIntent and Fill
│   ├── risk/                 # Risk management
│   ├── reporting/            # Metrics and plots
│   ├── robustness/           # Robustness campaign harness
│   └── ev/                   # Expected value simulator
├── tests/                    # Test suite (see tests/README.md)
├── CLAUDE.md                 # Engineering standards and architecture
├── README.md                 # This file
└── pyproject.toml            # Dependencies and scripts
```

**Key Documentation:**
- [`CLAUDE.md`](CLAUDE.md) — Engineering standards, architecture, operating manual
- [`src/README.md`](src/README.md) — Source code module overview
- [`src/indicators/README.md`](src/indicators/README.md) — Feature engineering guide
- [`tests/README.md`](tests/README.md) — Test suite documentation
- [`data/README.md`](data/README.md) — Data management guide

---

## 📚 Documentation Navigator

### Core Architecture
- [**Engineering Standards & Operating Manual**](CLAUDE.md) — Required reading for contributors
- [**Source Code Overview**](src/README.md) — Module architecture and data flow

### Pipeline Components
- [**ML Pipeline**](src/ml/README.md) — Features, labels, models, calibration
- [**Strategy System**](src/strategy/README.md) — Strategy protocol, registry, implementations
- [**Backtest Engine**](src/engine/README.md) — 8-stage pipeline, per-bar processing
- [**Feature Engineering**](src/indicators/README.md) — Indicators, transforms, custom features
- [**Regime Detection**](src/regime/README.md) — HMM-based trend/range classification
- [**Position Sizing Policies**](src/policy/README.md) — Fixed threshold, probability-sized
- [**Walk-Forward Validation**](src/validation/README.md) — Leakage-safe time series splitter

### Infrastructure
- [**Data Management**](data/README.md) — MT5 data pipeline, snapshots
- [**MT5 Loader**](src/loader/README.md) — Data ingestion & versioning
- [**Bar Replay**](src/replay/README.md) — Validation, deduplication, iteration
- [**Execution Layer**](src/execution/README.md) — OrderIntent, Fill, transaction costs
- [**Risk Management**](src/risk/README.md) — Drawdown halt, daily/monthly limits
- [**Metrics & Reporting**](src/reporting/README.md) — P&L, Sharpe, plots

### Research Workflows
- [**Robustness Testing**](src/robustness/README.md) — Campaign harness, promotion gates
- [**Evaluation Suite**](src/eval/README.md) — Canonical trio, run comparisons
- [**Utility Scripts**](scripts/README.md) — Helper tools & automation
- [**Test Suite**](tests/README.md) — Coverage, determinism checks

### Configuration
- [**Config Examples & Schema**](configs/README.md) — YAML reference, preset configs

---

## Command Reference

### Data Management

```bash
# Load M5 data from MT5
uv run loader
```

### Backtesting

```bash
# Run single backtest
uv run python -m src backtest --config configs/w4_rule_baseline.yaml

# Run with specific seed
uv run python -m src backtest --config configs/w6_logistic_calibrated.yaml --seed 42
```

### Comparison & Analysis

```bash
# Run canonical trio comparison
uv run python scripts/run_canonical_trio.py

# Compare two runs manually
uv run python scripts/compare_runs.py runs/20260308_143022 runs/20260308_154533
```

### Robustness Campaigns

```bash
# Dry-run (validate manifest only)
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml --dry-run

# Execute campaign
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml

# Finalize and generate reports
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml --finalize runs/robustness/w8_probsized_20260308_143022
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_feature_pipeline.py

# Run by keyword
uv run pytest -k determinism

# Run with verbose output
uv run pytest -v
```

### MLflow

```bash
# Enable tracking (after uv sync --extra tracking)
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# View UI
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## FAQ / Troubleshooting

### Q: "MetaTrader5 module not found"

**A:** Ensure MT5 Terminal is installed and you're on Windows. The MT5 Python API only works on Windows.

```bash
# Verify installation
uv run python -c "import MetaTrader5; print(MetaTrader5.__version__)"
```

### Q: Backtest results differ between runs

**A:** Check for:
1. Same data snapshot (verify `snapshot_dir` in config)
2. Same random seed (if using ML models)
3. Same config parameters
4. Clean git state (no uncommitted changes)

If all match, results should be identical (determinism guarantee).

### Q: Walk-forward validation returns zero folds

**A:** Check that `train_size + test_size <= total_bars`. The walk-forward splitter requires enough data for at least one fold.

```yaml
validation:
  train_size: 100000    # Reduce if dataset is small
  test_size: 50000
```

### Q: ML run produces no predictions.csv

**A:** Check that:
1. `validation.type` is set to `walk_forward` (required for ML)
2. At least one test fold completed successfully
3. No exceptions in log output

### Q: How to add a new indicator?

**A:** See [`src/indicators/README.md`](src/indicators/README.md) for detailed guide. Summary:

1. Implement `Indicator` protocol in `src/indicators/impl/`
2. Add to `src/indicators/__init__.py` exports
3. Register in feature builder (if needed)

---

## Contributing & Development

### Engineering Standards

See [`CLAUDE.md`](CLAUDE.md) for comprehensive engineering standards and operating manual.

**Key Principles:**
- **Minimal patch first** — Always prefer the smallest safe fix
- **Preserve determinism** — Same seed + config + data = same results
- **Explicit exceptions** — Use `if` + `raise`, not `assert` (production code must work under `-O`)
- **Typed structures** — Use dataclasses, TypedDict, or Protocols for critical data structures
- **No premature abstraction** — Only add abstraction when it removes real coupling

### Before Making Changes

1. **Read existing code** — Understand current behavior
2. **Identify minimal patch** — What's the smallest safe change?
3. **Run targeted tests** — Test what you changed (`pytest tests/test_<module>.py`)
4. **Run full regression** — Ensure no breakage (`pytest`)
5. **Update docs** — If public API or config schema changed

### Escalation

If you encounter:
- Ambiguous requirements → Ask before proceeding
- Test failures you can't diagnose → Report and stop
- Changes that risk determinism → Halt and request approval

---

## License & Acknowledgments

This project is part of an MSc research thesis at the **University of Moratuwa** (MSc Data Science & AI).

**Title**: "Adaptive, Uncertainty-Aware, Execution-Aware Intraday Trading Framework for XAUUSD (Gold)"

For questions or collaboration inquiries, please open an issue on the repository.
