# Backtest Engine

**8-stage deterministic backtesting pipeline with walk-forward support and comprehensive artifact generation.**

## Overview

The `src/engine/` package implements the core backtesting infrastructure for qr-factory. It consists of two main components:

1. **TradingEngine (`core.py`)**: Mode-agnostic per-bar processing core (execution → MTM → risk → signal → decision)
2. **Runner (`runner.py`)**: Orchestrates the 8-stage backtest pipeline (config → data → features → replay → metrics → artifacts)

Both backtest and future live mode will use the same `TradingEngine` core, ensuring consistent behavior across modes.

## Architecture

### 8-Stage Pipeline (runner.py)

The `run_backtest()` function orchestrates the following stages:

```
1. Configuration  →  2. Run Setup  →  3. Data  →  4. Validation
         ↓
5. Features  →  6. Replay  →  7. Metrics  →  8. Artifacts
```

| Stage | Function | Description |
|-------|----------|-------------|
| **1. Configuration** | `_load_config()` | Parse YAML, build strategy, extract risk/cost configs |
| **2. Run Setup** | `_create_run_dir()` | Create timestamped run directory + subdirectories |
| **3. Data** | `_load_market_data()` | Load and concatenate CSVs from snapshot directory |
| **4. Validation** | `validate_bars()` | Integrity checks (timestamps, OHLC, spread sanity) |
| **5. Features** | `FeaturePipeline.compute()` | Compute indicators + transforms for all bars |
| **6. Replay** | `_run_replay()` | Process bars through TradingEngine (or walk-forward folds) |
| **7. Metrics** | `_compute_metrics()` | Calculate P&L, Sharpe, Calmar, drawdown, win rate |
| **8. Artifacts** | `_write_artifacts()` | Write equity.csv, trades.csv, metrics.json, plots, README |

### Per-Bar Processing (core.py)

Each bar is processed through the `TradingEngine.process_bar()` pipeline:

```
1. Execution (Open)     → Fill pending OrderIntent from previous bar
2. MTM (Close)          → Compute unrealized P&L, update equity
3. Risk Check           → RiskManager.update() → RiskAction
4. Record Equity        → Append equity snapshot to history
5. Strategy Signal      → strategy.on_bar(ctx) → Signal
6. Decision Layer       → Signal + RiskAction → OrderIntent
7. Queue Intent         → Store for next bar's execution phase
```

**Key timing:**
- **Execution** happens at current bar's **open** price
- **Signal generation** happens at current bar's **close** price (after MTM)
- **OrderIntent** is executed at **next bar's open** price (1-bar delay, no lookahead)

**Latency simulation:** Optional `latency_bars` parameter adds extra execution delay (e.g., `latency_bars=1` → 2-bar total delay).

## Stage 1: Configuration

### `_load_config()`

Parses YAML config and builds strategy + feature specs.

**Returns:**
```python
(cfg_path, cfg_dict, strategy, feature_specs, risk_config, cost_config)
```

**Config validation:**
- Required keys: `symbol`, `timeframe`, `snapshot_dir`, `output_dir`, `starting_capital`, `strategy`
- Missing keys → `ValueError`

**Strategy building:**
```python
strategy, feature_specs = build_strategy(cfg["strategy"])
```

Uses registry pattern to instantiate strategy based on `type` key.

**Risk config:**
```python
risk_config = RiskConfig(
    max_drawdown_pct=cfg.get("max_drawdown_pct", None),
    daily_dd_limit=cfg.get("daily_drawdown_pct", None),
    monthly_dd_limit=cfg.get("monthly_drawdown_pct", None),
)
```

**Cost config:**
```python
cost_config = CostConfig(
    spread_points=exec_cfg.get("spread_points", 0.0),
    slippage_points=exec_cfg.get("slippage_points", 0.0),
    point_value=exec_cfg.get("point_value", 1.0),
)
```

## Stage 2: Run Setup

### `_create_run_dir()`

Creates timestamped run directory:

```
runs/<YYYYMMDD_HHMMSS>/
├── plots/
└── data_snapshot/
```

**Run ID:** `datetime.now(UTC).strftime("%Y%m%d_%H%M%S")`

**Returns:** `(run_id, run_dir)`

## Stage 3: Data Loading

### `_load_market_data()`

Loads all CSVs from snapshot directory:

```python
snapshot_dir = "data/20260217_030324"
bars, csv_files = _load_market_data(snapshot_dir)
```

**Processing:**
1. Find all `*.csv` files in snapshot directory
2. Read each CSV with `pd.read_csv(parse_dates=["time"])`
3. Handle volume column aliasing (`tick_volume` or `real_volume` → `volume`)
4. Concatenate all CSVs into single DataFrame
5. Sort by `time` (ascending)

**Returns:**
```python
(bars: pd.DataFrame, csv_files: list[Path])
```

**Expected columns:** `time`, `open`, `high`, `low`, `close`, `volume`

## Stage 4: Validation & Preparation

### `validate_bars()`

Fail-fast integrity checks (from `src/replay/validation.py`):

1. **Timestamp checks:**
   - Timestamps are `pd.Timestamp` (not string)
   - Timestamps are timezone-aware (UTC)
   - Timestamps are monotonically increasing
   - No duplicate timestamps

2. **OHLC checks:**
   - No NaNs in `open`, `high`, `low`, `close`
   - `high >= max(open, close)`
   - `low <= min(open, close)`

3. **Spread sanity:**
   - `(high - low) / close < sanity_threshold` (default 0.1 = 10%)

**Raises:** `ValueError` if any check fails.

### `BarIterator`

Canonicalizes timestamps and yields bars:

```python
from src.replay.bar_iterator import BarIterator

for bar_index, bar_row in BarIterator(bars):
    # bar_row is a pd.Series with time, open, high, low, close, volume
    process_bar(bar_row, ...)
```

**Guarantees:**
- Timestamps converted to `datetime64[ns, UTC]`
- Duplicates removed (keeps first)
- Sorted by timestamp (ascending)

## Stage 5: Features

### `FeaturePipeline.compute()`

Computes all required features for the strategy:

```python
from src.indicators.core.pipeline import FeaturePipeline

pipeline = FeaturePipeline(feature_specs)
features = pipeline.compute(bars)
```

**Returns:** `pd.DataFrame` with feature columns (index aligned to `bars`)

**Column naming:** Output columns are sorted alphabetically for determinism.

**NaN handling:** Warmup rows (where indicators are NaN) are preserved in the DataFrame. Strategies must check `can_trade()` before using features.

See [Feature Engineering](../indicators/README.md) for details.

## Stage 6: Replay

Two modes:

### 6a. Single-Fold Backtest (No Validation)

**When:** `validation` block is absent or `validation.type != "walk_forward"`

**Flow:**
1. Create `TradingEngine(strategy, risk_manager, starting_capital, ...)`
2. Loop through bars via `BarIterator`
3. Call `engine.process_bar(bar_row, features_row, bar_index)`
4. Collect `engine.equity_records` and `engine.all_fills`

### 6b. Walk-Forward Cross-Validation

**When:** `validation.type == "walk_forward"`

**Flow:**
1. Create `TimeSeriesSplitter` with train/test sizes and step
2. Generate folds: `[(train_indices, test_indices), ...]`
3. **Per fold:**
   - Build ForecastFrame (features + labels) for train and test slices
   - Train model (if ML strategy)
   - Calibrate model (if calibration enabled)
   - Inject model into strategy instance
   - Run backtest replay on test slice only
   - Collect fold metrics (P&L, Sharpe, Brier, etc.)
   - **Carry capital forward:** `fold[i+1].starting_capital = fold[i].ending_equity`
4. Aggregate fold metrics → `fold_metrics.json`, `evaluation.json`
5. Concatenate predictions across folds → `predictions.csv`

**FoldResult Schema:**

```python
@dataclass
class FoldResult:
    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    test_bars: int
    starting_capital: float
    n_trades: int
    gross_pnl: float
    net_pnl: float
    max_drawdown_pct: float
    ending_equity: float
    win_rate: float
    # ML-specific (optional)
    brier: float | None = None
    log_loss: float | None = None
    n_train_rows: int | None = None
    n_test_rows: int | None = None
    cal_n_train: int | None = None
    cal_n_cal: int | None = None
```

**Capital carryover:** Each fold starts with the ending equity of the previous fold (simulates continuous trading).

See [Walk-Forward Validation](../validation/README.md) for splitter details.

## Stage 7: Metrics

### `_compute_metrics()`

Calculates headline metrics from equity and trades:

**P&L Metrics:**
- `net_pnl`: Final equity - starting capital (after transaction costs)
- `gross_pnl`: Sum of gross P&L per trade (ignores costs)
- `total_fees`: Sum of transaction costs (spread + slippage)

**Risk-Adjusted Returns:**
- `sharpe_ratio`: Annualized Sharpe (bar-to-bar equity returns)
- `calmar_ratio`: Net P&L / Max Drawdown (absolute)
- `max_drawdown_pct`: Worst peak-to-trough equity drawdown (%)

**Trade Statistics:**
- `n_trades`: Number of completed round-trip trades
- `win_rate`: Fraction of trades with `pnl > 0`
- `avg_win`: Average P&L of winning trades
- `avg_loss`: Average P&L of losing trades (negative)

**Reproducibility:**
- `git_commit`: Git commit hash (or "dirty" if uncommitted changes)
- `git_dirty`: Boolean flag for uncommitted changes
- `run_timestamp`: ISO timestamp of run start

**Example output:**
```json
{
  "net_pnl": 1234.56,
  "gross_pnl": 1456.78,
  "sharpe_ratio": 1.23,
  "calmar_ratio": 0.45,
  "max_drawdown_pct": 8.5,
  "n_trades": 234,
  "win_rate": 0.52,
  "git_commit": "a3f2c9d",
  "git_dirty": false,
  "run_timestamp": "2026-03-08T14:30:22Z"
}
```

## Stage 8: Artifacts

### `_write_artifacts()`

Writes all outputs to run directory:

#### `equity.csv`

Per-bar equity curve:

```csv
time,equity,equity_gross
2024-01-15 09:00:00,1000.0,1000.0
2024-01-15 09:05:00,1012.3,1015.6
...
```

**Columns:**
- `time`: Bar timestamp
- `equity`: Net equity (after transaction costs)
- `equity_gross`: Gross equity (ignores costs)

#### `trades.csv`

Completed round-trip trades:

```csv
entry_time,exit_time,side,qty,entry_price,exit_price,gross_pnl,fees,pnl,reason
2024-01-15 09:00:00,2024-01-15 10:30:00,long,1.0,2045.3,2047.8,2.5,0.5,2.0,MA crossover
...
```

**Columns:**
- `entry_time`, `exit_time`: Trade timestamps
- `side`: `long` or `short`
- `qty`: Position size (always positive)
- `entry_price`, `exit_price`: Execution prices (cost-adjusted)
- `gross_pnl`: P&L before transaction costs
- `fees`: Transaction costs (spread + slippage)
- `pnl`: Net P&L (gross_pnl - fees)
- `reason`: Strategy-provided reason string

#### `metrics.json`

Headline metrics + git metadata (see Stage 7 above).

#### `config.yaml`

Exact snapshot of input config (for reproducibility).

#### `DATA_REF.json`

Pointer to data snapshot used:

```json
{
  "snapshot_dir": "data/20260217_030324",
  "csv_files": [
    "data/20260217_030324/XAUUSD_M5_2023.csv",
    "data/20260217_030324/XAUUSD_M5_2024.csv"
  ]
}
```

#### `README.md`

Human-readable run summary (generated from template).

#### `plots/`

Matplotlib plots:
- `close_price.png`: Close price over time
- `equity_curve.png`: Equity curve (net + gross)
- `fold_pnl.png`: Per-fold P&L bar chart (walk-forward only)
- `gross_vs_net_equity.png`: Gross vs net equity overlay (walk-forward only)

**Backend:** Non-interactive Agg (for headless environments).

#### Walk-Forward Artifacts (ML only)

- `predictions.csv`: Per-bar predictions on test folds
- `fold_metrics.json`: Per-fold results (see FoldResult schema above)
- `evaluation.json`: Aggregated cross-fold metrics

#### Extra Artifacts (Strategy-Specific)

Strategies can export extra artifacts via `extra_artifacts()` protocol:

```python
def extra_artifacts(self) -> dict[str, pd.DataFrame]:
    return {"regime.csv": self._regime_history}
```

Example: `DonchianATRRegime` exports `regime.csv` with HMM regime assignments.

## Per-Bar Processing Deep Dive

### TradingEngine.process_bar()

**Signature:**
```python
def process_bar(
    self,
    bar_row: pd.Series,
    features_row: pd.Series,
    bar_index: int,
    is_last: bool = False,
) -> None
```

**Steps:**

#### 1. Execution (at Open)

```python
if len(self._intent_queue) > self._latency_bars:
    ready_intent = self._intent_queue.popleft()
    target_position = ready_intent.target_position
else:
    target_position = self._position  # No ready intent, hold current
```

**Execution flow:**
- Pop ready intent from queue (accounting for latency delay)
- Call `_execute_order()` to compute fills
- Update `self._position`, `self._entry_price`, `self._equity`
- Append fills to `self.all_fills`

**Cost adjustment:**
```python
if going_long:
    entry_price = mid_price + (spread_points / 2 + slippage_points) * point_value
elif going_short:
    entry_price = mid_price - (spread_points / 2 + slippage_points) * point_value
```

**Fill object:**
```python
Fill(
    entry_ts=entry_timestamp,
    exit_ts=current_timestamp,
    side="long" or "short",
    qty=abs(position),
    entry_price=cost_adjusted_entry,
    exit_price=cost_adjusted_exit,
    gross_pnl=gross_pnl,
    fees=transaction_costs,
    pnl=net_pnl,
    reason=strategy_reason,
)
```

#### 2. Mark-to-Market (at Close)

```python
current_close = bar_row["close"]
if self._position > 0:
    unrealized_pnl = (current_close - self._entry_price) * abs(self._position)
elif self._position < 0:
    unrealized_pnl = (self._entry_price - current_close) * abs(self._position)

current_equity = self._equity + unrealized_pnl
```

**Purpose:** Update equity with unrealized P&L from open positions.

#### 3. Risk Check

```python
risk_action = self.risk_manager.update(bar_ts, current_equity)
```

**RiskAction fields:**
- `flatten`: Boolean (force close all positions)
- `halted`: Boolean (stop trading for remainder of run)
- `reason`: String (e.g., "Global DD halt: 10.2% > 10.0%")

**Risk overrides strategy:** If `risk_action.halted` or `risk_action.flatten`, the decision layer ignores strategy signal and flattens position.

See [Risk Management](../risk/README.md) for details.

#### 4. Record Equity

```python
self.equity_records.append({
    "time": bar_ts,
    "equity": current_equity,
    "equity_gross": current_equity_gross,
})
```

**Purpose:** Capture equity snapshot for `equity.csv` and drawdown calculations.

#### 5. Strategy Signal

```python
ctx = StrategyContext(
    ts=bar_ts,
    bar=bar_row,
    features=features_row,
    position=self._position,
    equity=current_equity,
    bar_index=bar_index,
)
signal = self.strategy.on_bar(ctx)
```

**Signal output:** `Signal(direction, strength, reason)`

**Warmup handling:** If `strategy.can_trade(ctx) == False`, strategy should return `Signal.flat("warmup")`.

#### 6. Decision Layer

```python
intent = self._decide(signal, risk_action)
```

**Conversion logic:**
```python
if risk_action.halted or risk_action.flatten:
    return OrderIntent(target_position=0.0, reason=risk_action.reason)

target = signal.direction * self.position_size  # Scale by position_size

if target == self._position:
    return OrderIntent(target_position=self._position, reason="hold")

return OrderIntent(target_position=target, reason=signal.reason)
```

**Key:** Position sizing happens here, not in strategy.

#### 7. Queue Intent

```python
self._intent_queue.append(intent)
```

**Queue behavior:**
- `latency_bars=0` (default): Intent at bar T executes at bar T+1
- `latency_bars=1`: Intent at bar T executes at bar T+2
- Queue depth = `latency_bars + 1`

**Purpose:** Simulates network latency / decision-to-execution delay.

## Determinism Guarantees

The engine enforces strict determinism:

1. **Timestamp canonicalization:** UTC timezone, nanosecond precision
2. **Feature column sorting:** Alphabetically sorted (no insertion-order randomness)
3. **Duplicate removal:** Consistent (keeps first)
4. **Execution order:** Fixed per-bar pipeline (no parallelism, no threading)
5. **Random seed:** Explicit seeding for ML models (via config)

**Verification:** `pytest tests/test_determinism.py` runs identical configs twice and asserts equity curves match.

## Config Examples

### Minimal (Single-Fold Backtest)

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

### Walk-Forward ML

```yaml
# ... (same as above, plus:)

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

label:
  type: return_sign_cost_aware
  horizon: 12
  cost_buffer: 0.0005

model:
  type: logistic
  C: 1.0
  class_weight: balanced

calibration:
  method: platt
  cal_fraction: 0.2

validation:
  type: walk_forward
  mode: expanding
  train_size: 100000
  test_size: 50000
  step: 50000
```

### Latency Simulation

```yaml
# ... (same as above, plus:)

execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01
  latency_bars: 1        # 2-bar total delay (1 standard + 1 latency)
```

## Key Files

| File | Description |
|------|-------------|
| `core.py` | TradingEngine class (per-bar processing pipeline) |
| `runner.py` | `run_backtest()` orchestrator (8-stage pipeline) |
| `git_info.py` | Git metadata capture (`get_git_info()`) |

## See Also

- [Strategy System](../strategy/README.md) — How strategies produce signals
- [ML Pipeline](../ml/README.md) — How walk-forward ML integration works
- [Risk Management](../risk/README.md) — How RiskAction overrides signals
- [Execution Layer](../execution/README.md) — OrderIntent → Fill conversion
- [Walk-Forward Validation](../validation/README.md) — Time series splitter
- [Feature Engineering](../indicators/README.md) — FeaturePipeline
