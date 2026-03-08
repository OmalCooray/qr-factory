# ML Pipeline

**Walk-forward machine learning integration with probabilistic forecasting, calibration, and policy-driven decision-making.**

## Overview

The `src/ml/` package implements a walk-forward machine learning pipeline for intraday trading. It bridges raw OHLCV data, feature engineering, and probabilistic model training with the backtest engine. The pipeline enforces leakage-safe train/calibration/test splits, applies Platt scaling for probability calibration, and converts calibrated forecasts into trading signals via configurable policies.

**Core workflow:**
```
Features (X) + Labels (y) → ForecastFrame → Train → Calibrate → Test → Predictions
```

The pipeline is designed for **uncertainty-aware trading**: models output calibrated probabilities, not binary decisions, enabling position sizing and trade filtering based on model confidence.

## Architecture

### Key Components

| Module | Purpose |
|--------|---------|
| **`frame.py`** | Builds ForecastFrame (features + labels) with `X_*` prefix convention |
| **`predictor.py`** | Protocol + implementations (Logistic, XGBoost, LightGBM) |
| **`calibration.py`** | Platt scaling for probability calibration |
| **`policy.py`** | Base and implementations (Fixed Threshold, Probability-Sized) |

### ForecastFrame

A **ForecastFrame** is a standardized `pd.DataFrame` with:
- **`X_*`** columns: Features (prefixed for clarity)
- **`y`**: Binary label (1 = profitable move, 0 = unprofitable)
- **`ret_fwd`**: Realized forward return used for labeling
- **`w`**: Sample weight (planned, not yet implemented)

**Contract:**
- Index aligned to bars (minus trailing rows when labels are present)
- Feature columns are sorted alphabetically (determinism)
- Rows with NaN features or labels are dropped before training

**Example:**
```
                    X_ema_5  X_ema_10_zscore  X_rsi_14  y  ret_fwd
2024-01-15 09:00:00  2045.3           1.234     68.5    1   0.0012
2024-01-15 09:05:00  2045.8           1.189     67.2    0  -0.0003
...
```

### Label Generation

Labels are generated from **forward-looking returns** with transaction cost awareness.

#### `return_sign` (simple threshold)

```yaml
label:
  type: return_sign
  horizon: 12               # Forward-looking horizon (bars)
  price_col: close          # Price column for return calculation
  threshold: 0.0            # Label = 1 if ret_fwd > threshold
```

**Logic:** `y = 1 if (price[t+horizon] / price[t] - 1) > threshold else 0`

#### `return_sign_cost_aware` (transaction cost buffering)

```yaml
label:
  type: return_sign_cost_aware
  horizon: 12
  price_col: close
  cost_buffer: 0.0005       # Transaction cost buffer (e.g., 50 bps)
  direction: up             # Direction to predict (up / down / both)
```

**Logic (for `direction: up`):**
```python
ret_fwd = price[t+horizon] / price[t] - 1
y = 1 if ret_fwd > cost_buffer else 0
```

**Why cost-aware?** A raw `ret_fwd > 0` label ignores transaction costs (spread + slippage). A model trained on such labels will predict many marginal moves that become unprofitable after costs. The `cost_buffer` parameter ensures labels only mark moves that exceed realistic transaction costs.

**Example:** For XAUUSD M5 with 40 points spread + 10 points slippage:
```python
cost_buffer = (40 + 10) * 0.01 / typical_price ≈ 0.0002 (20 bps)
```

### Predictor Protocol

All models implement the `Predictor` protocol:

```python
class Predictor(Protocol):
    def fit(self, train_ff: pd.DataFrame) -> None:
        """Train on ForecastFrame with X_* features and y label."""
        ...

    def predict(self, test_ff: pd.DataFrame) -> pd.Series:
        """Return predicted probabilities (P(y=1)) aligned to input index."""
        ...
```

**Naming convention:** Predictions are returned as a `pd.Series` named `"yhat"` with index matching the test ForecastFrame.

### Built-in Predictors

| Type | Class | Description | Scaling |
|------|-------|-------------|---------|
| **`logistic`** | `SklearnLogisticPredictor` | Logistic Regression | Optional (via `scale` param) |
| **`xgboost`** | `XGBoostPredictor` | XGBoost Classifier | Not needed (tree-based) |
| **`lightgbm`** | `LightGBMPredictor` | LightGBM Classifier | Not needed (tree-based) |

**Config examples:**

```yaml
# Logistic Regression
model:
  type: logistic
  C: 1.0                    # Regularization strength (inverse)
  max_iter: 200             # Max iterations for solver
  class_weight: balanced    # Class weighting strategy
  scale: false              # Feature scaling (StandardScaler)
  random_state: 0           # Random seed

# XGBoost
model:
  type: xgboost
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 0

# LightGBM
model:
  type: lightgbm
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  num_leaves: 31
  feature_fraction: 0.8
  random_state: 0
```

## Platt Calibration

**Why calibrate?**

Raw classifier outputs (especially from tree models like XGBoost/LightGBM) are often **poorly calibrated**. A raw score of 0.7 does not mean "70% empirical positive rate". Calibration maps raw scores to well-calibrated probabilities where `P(y=1|X) ≈ 0.7` means ~70% of samples with that score are actually positive.

**Platt scaling** fits a logistic regression on held-out calibration data:
```
calibrated_prob = sigmoid(A * raw_score + B)
```

### Walk-Forward Calibration Workflow

Each fold is split into **train** → **calibration** → **test**:

1. **Train split** (e.g., 80% of fold's training data):
   - Train base model (Logistic/XGBoost/LightGBM)
2. **Calibration split** (e.g., 20% of fold's training data):
   - Get raw predictions from base model
   - Fit Platt scaler (logistic regression on raw scores → true labels)
3. **Test split** (entire test fold):
   - Get raw predictions from base model
   - Apply Platt scaler to calibrate probabilities
   - Return calibrated probabilities to strategy

**Config:**
```yaml
calibration:
  method: platt             # Calibration method (platt or isotonic)
  cal_fraction: 0.2         # Fraction of training data for calibration
```

**Implementation:** See `PlattCalibrator` in `src/ml/calibration.py`.

**Critical for probability-sized policies:** Position sizing logic (see Policy section) scales position by probability. If probabilities are poorly calibrated, position sizes will be systematically biased.

## Policy Layer

The **policy layer** converts calibrated probabilities into trading signals (Signal objects with direction + strength).

### Policy Protocol

```python
class Policy(Protocol):
    def decide(self, p_pred: float, position: float, bars_held: int) -> Signal:
        """Convert probability + current state → Signal."""
        ...
```

### Fixed Threshold Policy

Simple threshold-based entry/exit:

```yaml
strategy:
  type: ml_prob_threshold
  p_enter: 0.60             # Enter long if P(up) > 0.60
  p_exit: 0.45              # Exit if P(up) < 0.45
  min_hold_bars: 12         # Minimum bars to hold position
```

**Logic:**
- Flat + `p > p_enter` → Long (strength = 1.0)
- Long + `p < p_exit` → Flat
- Long + `bars_held < min_hold_bars` → Hold (no signal)

**Position size:** Fixed at 1.0 (full position). Sizing happens in `TradingEngine`, not policy.

### Probability-Sized Policy

**Scales position size by model confidence** (linear interpolation):

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: probability_sized
    p_enter: 0.40           # Enter if P(up) > 0.40
    p_exit: 0.30            # Exit if P(up) < 0.30
    p_full: 0.50            # Full position (max_size) if P(up) >= 0.50
    min_size: 0.25          # Minimum position size (at p_enter)
    max_size: 1.0           # Maximum position size (at p_full)
    min_hold_bars: 5        # Minimum hold period
```

**Sizing logic:**
```python
if p > p_full:
    size = max_size         # Full position
elif p > p_enter:
    # Linear interpolation between min_size and max_size
    size = min_size + (max_size - min_size) * (p - p_enter) / (p_full - p_enter)
else:
    size = 0.0              # Flat
```

**Example:**
- `p = 0.40` → `size = 0.25` (min_size, cautious entry)
- `p = 0.45` → `size = 0.625` (interpolated)
- `p = 0.50` → `size = 1.0` (max_size, full conviction)
- `p = 0.35` → `size = 0.0` (flat, below threshold)

**Key advantage:** Position size reflects model uncertainty. High-confidence predictions get larger positions; marginal predictions get smaller positions. This is **execution-aware decision-making** — the policy layer respects uncertainty.

## Integration with Backtest Engine

### Walk-Forward Integration

The ML pipeline is invoked by `runner.py` when `validation.type == "walk_forward"`:

1. **Time series splitter** generates folds (train/test index pairs)
2. **Per fold:**
   - Build ForecastFrame (features + labels) for train and test slices
   - Split train into train/calibration (per `cal_fraction`)
   - Train model on train split
   - Calibrate on calibration split
   - Predict on test split → `predictions.csv`
   - Run backtest replay on test bars with ML strategy
   - Collect fold metrics (P&L, Sharpe, Brier, etc.)
3. **Aggregate:**
   - Write `fold_metrics.json` (per-fold results)
   - Write `evaluation.json` (cross-fold aggregated metrics)

### Strategy Integration

ML strategies (e.g., `MLProbThresholdStrategy`) implement the `Strategy` protocol and use trained models via the policy layer:

**Simplified flow:**
```python
class MLProbThresholdStrategy:
    def on_bar(self, ctx: StrategyContext) -> Signal:
        # Get features for current bar
        features = ctx.features[self.feature_names]

        # Predict probability (calibrated)
        p_pred = self.model.predict_proba(features)

        # Policy decides signal based on probability
        signal = self.policy.decide(p_pred, ctx.position, self.bars_held)

        return signal
```

**Critical:** The model object is **trained per fold** and injected into the strategy instance before the test fold replay. See `runner.py` for implementation.

## Artifacts

ML runs produce additional artifacts in the run directory:

### `predictions.csv`

Per-bar predictions for all test folds:

```csv
timestamp,fold_id,y,yhat,position,ret_fwd
2024-01-15 09:00:00,0,1,0.623,1.0,0.0012
2024-01-15 09:05:00,0,0,0.548,1.0,-0.0003
...
```

**Columns:**
- `timestamp`: Bar timestamp (test fold only)
- `fold_id`: Which walk-forward fold (0-indexed)
- `y`: True label (binary)
- `yhat`: Predicted probability (calibrated if calibration enabled)
- `position`: Current position at this bar
- `ret_fwd`: Realized forward return

**Purpose:** Enable post-hoc analysis (calibration curves, threshold tuning, regime performance).

### `fold_metrics.json`

Per-fold training and test metrics:

```json
[
  {
    "fold_id": 0,
    "train_start": 0,
    "train_end": 100000,
    "test_start": 100000,
    "test_end": 150000,
    "test_bars": 50000,
    "n_trades": 234,
    "gross_pnl": 1234.56,
    "net_pnl": 987.65,
    "max_drawdown_pct": 5.2,
    "win_rate": 0.52,
    "brier": 0.234,
    "log_loss": 0.678,
    "n_train_rows": 98567,
    "n_test_rows": 49234,
    "cal_n_train": 78853,
    "cal_n_cal": 19714
  },
  ...
]
```

**Key metrics:**
- **P&L metrics:** `net_pnl`, `gross_pnl`, `max_drawdown_pct`, `win_rate`
- **ML metrics:** `brier` (Brier score), `log_loss` (cross-entropy loss)
- **Sample counts:** `n_train_rows`, `n_test_rows`, `cal_n_train`, `cal_n_cal`

### `evaluation.json`

Aggregated metrics across all folds:

```json
{
  "total_folds": 5,
  "total_net_pnl": 4321.23,
  "mean_sharpe": 1.23,
  "mean_brier": 0.245,
  "mean_log_loss": 0.687,
  "std_net_pnl": 234.56,
  "mean_win_rate": 0.51,
  "mean_max_dd_pct": 6.1
}
```

**Purpose:** Quick summary for comparing different model configs or feature sets.

## Config Examples

### Minimal ML Config (Logistic Regression)

```yaml
symbol: "XAUUSD"
timeframe: "M5"
snapshot_dir: "data/20260217_030324"
starting_capital: 1000.0
output_dir: "runs"

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

label:
  type: return_sign_cost_aware
  horizon: 12
  price_col: close
  cost_buffer: 0.0005
  direction: up

model:
  type: logistic
  C: 1.0
  max_iter: 200
  class_weight: balanced
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

execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01
```

### Probability-Sized Policy (LightGBM)

```yaml
# ... (same as above, except:)

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
    # ... (same feature list)

model:
  type: lightgbm
  n_estimators: 100
  max_depth: 5
  learning_rate: 0.1
  num_leaves: 31
  feature_fraction: 0.8
  random_state: 0
```

## Extending the ML Pipeline

### Adding a New Predictor

**Step 1:** Implement the `Predictor` protocol in `src/ml/predictor.py`:

```python
class MyCustomPredictor:
    def __init__(self, param1: float, param2: int, random_state: int = 0) -> None:
        self._param1 = param1
        self._param2 = param2
        self._x_cols: list[str] = []
        # Initialize your model here

    def fit(self, train_ff: pd.DataFrame) -> None:
        self._x_cols = sorted(c for c in train_ff.columns if c.startswith("X_"))
        X = train_ff[self._x_cols].values
        y = train_ff["y"].values.astype(int)
        # Train your model here
        self._model.fit(X, y)

    def predict(self, test_ff: pd.DataFrame) -> pd.Series:
        X = test_ff[self._x_cols].values
        proba = self._model.predict_proba(X)[:, 1]  # P(y=1)
        return pd.Series(proba, index=test_ff.index, name="yhat")
```

**Step 2:** Register in `src/ml/predictor.py` (or a registry module):

```python
PREDICTOR_REGISTRY = {
    "logistic": lambda cfg: SklearnLogisticPredictor(**cfg),
    "xgboost": lambda cfg: XGBoostPredictor(**cfg),
    "lightgbm": lambda cfg: LightGBMPredictor(**cfg),
    "my_custom": lambda cfg: MyCustomPredictor(**cfg),  # Add here
}
```

**Step 3:** Use in config:

```yaml
model:
  type: my_custom
  param1: 0.5
  param2: 100
  random_state: 0
```

### Adding a New Policy

**Step 1:** Implement `Policy` protocol in `src/ml/policy.py`:

```python
@dataclass
class MyCustomPolicy:
    threshold: float = 0.5

    def decide(self, p_pred: float, position: float, bars_held: int) -> Signal:
        # Custom logic here
        if position == 0 and p_pred > self.threshold:
            return Signal.long(strength=1.0, reason=f"p={p_pred:.3f} > {self.threshold}")
        elif position != 0 and p_pred < self.threshold:
            return Signal.flat(reason=f"p={p_pred:.3f} < {self.threshold}")
        return Signal.flat(reason="hold")
```

**Step 2:** Wire into `MLProbThresholdStrategy` (or create a new strategy class).

**Step 3:** Use in config:

```yaml
strategy:
  type: ml_prob_threshold
  policy:
    type: my_custom
    threshold: 0.55
  features: [...]
```

## Key Files

| File | Description |
|------|-------------|
| `frame.py` | ForecastFrame builder, label generation (`return_sign`, `return_sign_cost_aware`) |
| `predictor.py` | Predictor protocol + implementations (Logistic, XGBoost, LightGBM) |
| `calibration.py` | Platt scaling calibrator |
| `policy.py` | Policy protocol + implementations (FixedThreshold, ProbabilitySized) |

## See Also

- [Strategy System](../strategy/README.md) — How strategies integrate with ML models
- [Backtest Engine](../engine/README.md) — How runner.py orchestrates walk-forward ML pipeline
- [Walk-Forward Validation](../validation/README.md) — Leakage-safe time series splitter
- [Feature Engineering](../indicators/README.md) — Building features for ML models
- [Position Sizing Policies](../policy/README.md) — Detailed policy layer documentation (if separate)
