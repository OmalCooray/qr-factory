# Configuration Examples & Schema

**YAML configuration reference for backtests and robustness campaigns.**

## Overview

The `configs/` directory contains YAML configuration files for backtests, robustness campaigns, and evaluation experiments. All configs share a common schema defined by the runner pipeline.

## Config Categories

### 1. Weekly Deliverables (w4–w8)

Progressive research milestones:

| Config | Description | Week |
|--------|-------------|------|
| `w4_rule_baseline.yaml` | EMA crossover baseline (no ML) | Week 4 |
| `w6_logistic_calibrated.yaml` | Logistic + Platt scaling | Week 6 |
| `w7_policy_probsized.yaml` | Probability-sized policy | Week 7 |
| `w8_robustness_probsized.yaml` | Robustness campaign manifest | Week 8 |

### 2. Experiment Configs (exp_*)

Ablation studies and baseline comparisons:

| Config | Description |
|--------|-------------|
| `exp_ema_50_200.yaml` | Long-period EMA crossover (50/200) |
| `exp_ml_logistic_baseline.yaml` | Logistic regression baseline |
| `exp_ml_xgboost_baseline.yaml` | XGBoost baseline |
| `exp_ml_lgbm_baseline.yaml` | LightGBM baseline |

### 3. Robustness Campaigns

Stress testing manifests:

| Config | Description |
|--------|-------------|
| `w8_robustness_probsized.yaml` | Full robustness campaign (cost stress, latency, sweeps) |
| `w8_robustness_conservative_risk.yaml` | Conservative risk limits campaign |

## Common Schema

### Required Keys

```yaml
symbol: "XAUUSD"
timeframe: "M5"
snapshot_dir: "data/20260217_030324"
starting_capital: 1000.0
output_dir: "runs"
strategy: {...}                     # Strategy block (see below)
execution: {...}                    # Execution costs
```

### Optional Keys

```yaml
validation: {...}                   # Walk-forward cross-validation
label: {...}                        # Label generation (ML only)
model: {...}                        # Model config (ML only)
calibration: {...}                  # Platt scaling (ML only)
risk: {...}                         # Risk management thresholds
```

## Strategy Block

### Rule-Based Strategy

```yaml
strategy:
  type: ma_crossover
  fast_period: 5
  slow_period: 10
  indicator_type: ema
```

### ADX Filtered Crossover

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

### ML with Fixed Threshold

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

### ML with Probability-Sized Policy

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

## Execution Block

```yaml
execution:
  spread_points: 40         # Bid-ask spread (points)
  slippage_points: 10       # Execution slippage (points)
  point_value: 0.01         # Value of 1 point in account currency
  latency_bars: 0           # Optional: execution delay (bars)
```

## Validation Block (Walk-Forward)

```yaml
validation:
  type: walk_forward
  mode: expanding           # expanding or rolling
  train_size: 100000        # Bars per training fold
  test_size: 50000          # Bars per test fold
  step: 50000               # Advance between folds
  drop_last: true           # Drop incomplete final fold
```

## Label Block (ML Only)

```yaml
label:
  type: return_sign_cost_aware
  horizon: 12               # Forward-looking horizon (bars)
  price_col: close          # Price column for returns
  cost_buffer: 0.0005       # Transaction cost buffer (50 bps)
  direction: up             # Direction to predict (up/down/both)
```

## Model Block (ML Only)

### Logistic Regression

```yaml
model:
  type: logistic
  C: 1.0
  max_iter: 200
  class_weight: balanced
  scale: false
  random_state: 0
```

### XGBoost

```yaml
model:
  type: xgboost
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 0
```

### LightGBM

```yaml
model:
  type: lightgbm
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  num_leaves: 31
  feature_fraction: 0.8
  random_state: 0
```

## Calibration Block (ML Only)

```yaml
calibration:
  method: platt             # platt or isotonic
  cal_fraction: 0.2         # Fraction of train for calibration
```

## Risk Block

```yaml
risk:
  global_dd_halt_pct: 10.0      # Global drawdown halt threshold (%)
  daily_loss_limit: 500.0       # Daily loss limit (account currency)
  monthly_loss_limit: 2000.0    # Monthly loss limit (account currency)
```

## Robustness Campaign Schema

```yaml
campaign_id: "w8_probsized"
baseline_config: "configs/w7_policy_probsized.yaml"
description: "Robustness campaign for probability-sized policy baseline"

scenarios:
  # Cost stress
  - family: cost_stress
    name: 1.5x_cost
    description: "1.5x baseline friction"
    overrides:
      execution:
        spread_points: 60
        slippage_points: 15

  # Latency proxy
  - family: latency_proxy
    name: 1bar_delay
    description: "1-bar execution delay"
    overrides:
      execution:
        latency_bars: 1

  # Parameter sweep
  - family: parameter_sweep
    name_prefix: p_enter
    description: "Entry threshold sensitivity"
    grid:
      strategy.policy.p_enter: [0.35, 0.40, 0.45]
```

## Example Configs

### Minimal Rule-Based

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
```

### Full ML Walk-Forward

```yaml
symbol: "XAUUSD"
timeframe: "M5"
snapshot_dir: "data/20260217_030324"
starting_capital: 1000.0
output_dir: "runs"

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

label:
  type: return_sign_cost_aware
  horizon: 12
  price_col: close
  cost_buffer: 0.0005
  direction: up

model:
  type: lightgbm
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  num_leaves: 31
  feature_fraction: 0.8
  random_state: 0

calibration:
  method: platt
  cal_fraction: 0.2

validation:
  type: walk_forward
  mode: expanding
  train_size: 100000
  test_size: 50000
  step: 50000
  drop_last: true

execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01

risk:
  global_dd_halt_pct: 10.0
  daily_loss_limit: 500.0
  monthly_loss_limit: 2000.0
```

## See Also

- [Strategy System](../src/strategy/README.md) — Strategy configuration reference
- [ML Pipeline](../src/ml/README.md) — ML block details
- [Robustness Testing](../src/robustness/README.md) — Campaign manifest schema
- [Backtest Engine](../src/engine/README.md) — How configs are loaded and validated
