# Walk-Forward Validation

**Leakage-safe time series cross-validation splitter for ML backtesting.**

## Overview

The `src/validation/` package implements a walk-forward time series splitter that generates non-overlapping train/test folds for machine learning backtesting. It enforces strict temporal ordering to prevent future-leakage: `train_indices.max() < test_indices.min()` for every fold.

**Key guarantee:** Training data is always in the past relative to test data (no lookahead bias).

## TimeSeriesSplitter

### Core Concept

Walk-forward validation splits a time series into multiple (train, test) folds:

```
[-------- Train 1 --------][-- Test 1 --]
                           [-------- Train 2 --------][-- Test 2 --]
                                                      [-------- Train 3 --------][-- Test 3 --]
```

**Two modes:**

1. **Expanding**: Training window grows from index 0 (anchored walk-forward)
2. **Rolling**: Training window slides forward (fixed-size rolling window)

### Expanding Mode

Training window starts at index 0 and expands over time:

```
Fold 0:  [Train: 0–99999    ] [Test: 100000–149999]
Fold 1:  [Train: 0–149999   ] [Test: 150000–199999]
Fold 2:  [Train: 0–199999   ] [Test: 200000–249999]
```

**Use case:** Markets have persistent trends/regimes. Larger training windows capture more historical patterns.

**Config:**
```yaml
validation:
  type: walk_forward
  mode: expanding
  train_size: 100000      # Initial training window
  test_size: 50000        # Test window size (fixed)
  step: 50000             # Advance by 50k bars per fold
  drop_last: true         # Drop incomplete final fold
```

**Formula:**
```python
train_start = 0
train_end = train_size + (fold_id * step)
test_start = train_end
test_end = test_start + test_size
```

### Rolling Mode

Training window slides forward (fixed size):

```
Fold 0:  [Train: 0–99999       ] [Test: 100000–149999]
Fold 1:  [Train: 50000–149999  ] [Test: 150000–199999]
Fold 2:  [Train: 100000–199999 ] [Test: 200000–249999]
```

**Use case:** Markets exhibit regime changes. Recent data is more relevant than distant past.

**Config:**
```yaml
validation:
  type: walk_forward
  mode: rolling
  train_size: 100000      # Fixed training window (slides)
  test_size: 50000        # Test window size (fixed)
  step: 50000             # Advance by 50k bars per fold
  drop_last: true
```

**Formula:**
```python
train_start = max(0, fold_id * step)
train_end = train_start + train_size
test_start = train_end
test_end = test_start + test_size
```

## Parameters

### `mode`

**Type:** `"expanding"` or `"rolling"`

**Description:** Whether training window expands or slides.

### `train_size`

**Type:** `int`

**Description:**
- **Expanding mode:** Initial training window size (grows over time)
- **Rolling mode:** Fixed training window size (slides over time)

**Typical values:** 50,000–200,000 bars (depends on dataset size and stationarity assumptions)

### `test_size`

**Type:** `int`

**Description:** Size of each test fold (fixed across all folds).

**Typical values:** 20,000–100,000 bars (20–50% of `train_size`)

**Research note:** Larger test folds provide more robust out-of-sample metrics but reduce number of folds.

### `step`

**Type:** `int` (optional, defaults to `test_size`)

**Description:** How many bars to advance between folds.

**Constraint:** `step >= test_size` (enforced to prevent overlapping test windows).

**Common values:**
- `step = test_size` (non-overlapping test folds, standard walk-forward)
- `step = test_size / 2` (overlapping test folds, **not recommended** for research)

**Why no overlap?** Overlapping test windows violate statistical independence assumptions. For rigorous backtesting, test folds must be disjoint.

### `drop_last`

**Type:** `bool` (default: `True`)

**Description:** Whether to drop incomplete final fold.

**Behavior:**
- `drop_last=True`: Skip final fold if `test_size` would extend beyond dataset end
- `drop_last=False`: Allow partial final fold (test length in `[1, test_size-1]`)

**Recommendation:** Use `True` for consistent fold sizes (easier metric aggregation).

## Usage

### Standalone

```python
from src.validation.time_series_splitter import TimeSeriesSplitter
import numpy as np

# Create splitter
splitter = TimeSeriesSplitter(
    mode="expanding",
    train_size=100000,
    test_size=50000,
    step=50000,
    drop_last=True,
)

# Generate folds
n_samples = 635_000  # Total bars in dataset
for fold_id, (train_idx, test_idx) in enumerate(splitter.split(n_samples)):
    print(f"Fold {fold_id}:")
    print(f"  Train: {train_idx[0]:6d} – {train_idx[-1]:6d} ({len(train_idx):6d} bars)")
    print(f"  Test:  {test_idx[0]:6d} – {test_idx[-1]:6d} ({len(test_idx):6d} bars)")
    print(f"  Leakage check: {train_idx.max()} < {test_idx.min()} = {train_idx.max() < test_idx.min()}")
```

**Output:**
```
Fold 0:
  Train:      0 –  99999 (100000 bars)
  Test:  100000 – 149999 ( 50000 bars)
  Leakage check: 99999 < 100000 = True

Fold 1:
  Train:      0 – 149999 (150000 bars)
  Test:  150000 – 199999 ( 50000 bars)
  Leakage check: 149999 < 150000 = True

...
```

### Integration with runner.py

The splitter is called by `runner.py` when `validation.type == "walk_forward"`:

```python
# Stage 6: Replay (Walk-Forward Mode)

# Load config
val_cfg = cfg["validation"]
splitter = TimeSeriesSplitter(
    mode=val_cfg["mode"],
    train_size=val_cfg["train_size"],
    test_size=val_cfg["test_size"],
    step=val_cfg.get("step", val_cfg["test_size"]),
    drop_last=val_cfg.get("drop_last", True),
)

# Generate folds
fold_results = []
for fold_id, (train_idx, test_idx) in enumerate(splitter.split(len(bars))):
    # Slice data
    train_bars = bars.iloc[train_idx]
    test_bars = bars.iloc[test_idx]

    # Build ForecastFrame (features + labels)
    train_ff = build_forecast_frame(train_bars, train_features, label_cfg)
    test_ff = build_forecast_frame(test_bars, test_features, label_cfg=None)

    # Train model (if ML strategy)
    model.fit(train_ff)

    # Backtest on test fold
    engine = TradingEngine(strategy, risk_manager, starting_capital=prev_fold_ending_equity)
    for bar_index, bar_row in BarIterator(test_bars):
        engine.process_bar(bar_row, features_row, bar_index)

    # Collect fold metrics
    fold_results.append(FoldResult(...))

# Aggregate fold metrics
write_fold_metrics(fold_results, run_dir)
```

## Leakage Prevention

### Enforced Guarantees

1. **Temporal ordering:** `train_indices.max() < test_indices.min()`
2. **Non-overlapping test windows:** `step >= test_size`
3. **No data duplication:** Indices are numpy arrays (no repeats)

**Assertion in runner.py:**

```python
if train_idx.max() >= test_idx.min():
    raise RuntimeError(
        f"Fold {fold_id} leakage detected: train.max() = {train_idx.max()}, "
        f"test.min() = {test_idx.min()}"
    )
```

**Purpose:** Fail-fast if splitter logic is violated (e.g., due to code changes).

### Common Pitfalls (Avoided)

**❌ Shuffle:** Never shuffle time series data (destroys temporal structure).

**❌ K-Fold CV:** Standard k-fold randomly assigns samples to folds (causes leakage in time series).

**❌ Rolling with `step < test_size`:** Creates overlapping test windows (inflates performance metrics).

**✓ Walk-Forward:** Only correct approach for time series backtesting.

## Config Examples

### Minimal Walk-Forward

```yaml
validation:
  type: walk_forward
  mode: expanding
  train_size: 100000
  test_size: 50000
```

**Behavior:**
- `step` defaults to `test_size` (50,000)
- `drop_last` defaults to `True`
- Expanding window (training grows from 100k → 150k → 200k → ...)

### Rolling Window with Custom Step

```yaml
validation:
  type: walk_forward
  mode: rolling
  train_size: 100000
  test_size: 50000
  step: 25000          # Advance by 25k per fold (faster iteration)
  drop_last: false     # Allow partial final fold
```

**Note:** `step=25000 < test_size=50000` is **invalid** (raises `ValueError` at construction).

### Large Folds for Robust Metrics

```yaml
validation:
  type: walk_forward
  mode: expanding
  train_size: 200000
  test_size: 100000
  step: 100000
  drop_last: true
```

**Trade-off:** Fewer folds (less cross-validation) but more robust per-fold metrics (larger test samples).

## Metrics Aggregation

Walk-forward runs produce per-fold metrics. Aggregation strategies:

### Mean Metrics

```python
mean_sharpe = np.mean([fold.sharpe_ratio for fold in fold_results])
mean_win_rate = np.mean([fold.win_rate for fold in fold_results])
```

**Use case:** Quick summary for model comparison.

### Weighted Mean (by test bars)

```python
total_bars = sum(fold.test_bars for fold in fold_results)
weighted_sharpe = sum(fold.sharpe_ratio * fold.test_bars for fold in fold_results) / total_bars
```

**Use case:** Folds with more data get higher weight (more reliable).

### Cumulative P&L

```python
total_net_pnl = sum(fold.net_pnl for fold in fold_results)
```

**Use case:** Simulates continuous trading (capital carries forward between folds).

**Note:** Runner already computes cumulative P&L via capital carryover (`fold[i+1].starting_capital = fold[i].ending_equity`).

### Standard Deviation

```python
std_sharpe = np.std([fold.sharpe_ratio for fold in fold_results])
```

**Use case:** Measures consistency across folds. High std = unstable performance.

## Number of Folds Calculation

### Formula

```python
def n_splits(n_samples: int) -> int:
    if mode == "expanding":
        available_test_bars = n_samples - train_size
    else:  # rolling
        available_test_bars = n_samples - train_size

    max_folds = available_test_bars // step

    if drop_last:
        # Only count folds where test window is fully within bounds
        return max([i for i in range(max_folds) if test_end(i) <= n_samples])
    else:
        # Allow partial final fold
        return max_folds
```

**Example:**
```python
splitter = TimeSeriesSplitter(mode="expanding", train_size=100000, test_size=50000, step=50000)
n_folds = splitter.n_splits(635000)  # Returns 10 folds
```

## Common Errors

### "n_samples too small for even one fold"

**Error:**
```
ValueError: n_samples (120000) must be >= train_size + test_size (150000)
when drop_last=True to guarantee at least one full test fold
```

**Fix:** Reduce `train_size` or `test_size`, or use more data.

### "step < test_size not allowed"

**Error:**
```
ValueError: step (25000) < test_size (50000): overlapping test windows
disallowed for research-grade walk-forward
```

**Fix:** Set `step >= test_size`.

### "No folds generated"

**Cause:** `n_samples` is just barely larger than `train_size + test_size`, but `step` is too large.

**Example:**
```python
splitter = TimeSeriesSplitter(mode="expanding", train_size=100000, test_size=50000, step=100000)
splitter.n_splits(160000)  # Returns 0 folds (not enough data for 2nd fold)
```

**Fix:** Reduce `step` or use more data.

## Key Files

| File | Description |
|------|-------------|
| `time_series_splitter.py` | TimeSeriesSplitter class (expanding/rolling walk-forward) |

## See Also

- [Backtest Engine](../engine/README.md) — How runner.py integrates the splitter
- [ML Pipeline](../ml/README.md) — How ForecastFrame is built per fold
- [Robustness Testing](../robustness/README.md) — Campaign-based stress testing
