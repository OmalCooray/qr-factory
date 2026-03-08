# Source Code (`src`)

**qr-factory source code architecture and module overview.**

## Overview

The `src/` directory contains the core Python packages for qr-factory, organized around the north-star pipeline:

```
Data → Features → Forecaster → Calibration → Policy → Execution → Risk/Metrics
```

## North-Star Pipeline Mapping

| Stage | Package | Status | Description |
|-------|---------|--------|-------------|
| **Data** | `loader`, `replay` | ✓ Done | MT5 ingestion, validation, bar iteration |
| **Features** | `indicators` | ✓ Done | OHLCV-derived indicators, transforms |
| **Forecaster** | `strategy`, `ml` | ✓ Done | Rule-based + ML (Logistic/XGBoost/LightGBM) |
| **Calibration** | `ml/calibration.py` | ✓ Done | Platt scaling for probability calibration |
| **Policy** | `policy`, `ml/policy.py` | ✓ Done | Fixed threshold, probability-sized |
| **Execution** | `execution` | ✓ Done | OrderIntent → Fill with transaction costs |
| **Risk** | `risk` | ✓ Done | Drawdown halt, daily/monthly limits |
| **Metrics** | `reporting`, `engine` | ✓ Done | P&L, Sharpe, Calmar, plots |

## Module Quick Reference

### Core Infrastructure

#### `loader/` — MT5 Data Pipeline
[**README**](loader/README.md)

- **Purpose:** Fetch OHLCV data from MetaTrader 5
- **Key files:** `_loader.py`, `data_source.py`, `data_writer.py`
- **Output:** Timestamped snapshot directories (`data/<YYYYMMDD_HHMMSS>/`)
- **Usage:** `uv run loader`

**Status:** ✓ Production-ready

---

#### `replay/` — Bar Validation & Iteration
[**README**](replay/README.md)

- **Purpose:** Validate data integrity and provide deterministic bar iteration
- **Key files:** `bar_iterator.py`, `validation.py`
- **Features:** Timestamp canonicalization, duplicate removal, OHLC integrity checks
- **Integration:** Called by runner.py in Stage 4 (Validation)

**Status:** ✓ Production-ready

---

#### `indicators/` — Feature Engineering
[**README**](indicators/README.md)

- **Purpose:** Compute OHLCV-derived features (SMA, EMA, RSI, ADX, etc.) with transforms
- **Key files:** `core/pipeline.py`, `impl/moving_averages.py`, `impl/oscillators.py`
- **Protocol:** `Indicator` + `Transform` → `FeaturePipeline`
- **Features:** Alphabetically sorted columns (determinism), NaN handling, composability

**Status:** ✓ Production-ready

---

### Strategy & ML

#### `strategy/` — Strategy System
[**README**](strategy/README.md)

- **Purpose:** Strategy protocol, registry, and implementations
- **Key files:** `base.py`, `signal.py`, `registry.py`, 8 strategy implementations
- **Protocol:** `Strategy` → `on_bar(ctx) -> Signal`
- **Implementations:** MA crossover, ADX filtered, Donchian+regime, Lorentzian, RSI+BB, ML
- **Signal vs OrderIntent:** Strategies produce beliefs (Signal), engine decides actions (OrderIntent)

**Status:** ✓ Production-ready (8 implementations)

---

#### `ml/` — ML Pipeline
[**README**](ml/README.md)

- **Purpose:** Walk-forward ML with probabilistic forecasting, calibration, and policy-driven decisions
- **Key files:** `frame.py`, `predictor.py`, `calibration.py`, `policy.py`
- **Models:** Logistic, XGBoost, LightGBM
- **Features:** ForecastFrame, Platt scaling, probability-sized positions
- **Artifacts:** `predictions.csv`, `fold_metrics.json`, `evaluation.json`

**Status:** ✓ Production-ready

---

#### `policy/` — Position Sizing Policies
[**README**](policy/README.md)

- **Purpose:** Convert calibrated probabilities to trading signals
- **Key files:** `base.py`, `fixed_threshold.py`, `probability_sized.py`
- **Implementations:** Fixed threshold, probability-sized (linear scaling)
- **Integration:** Used by `MLProbThresholdStrategy`

**Status:** ✓ Production-ready

---

### Execution & Risk

#### `execution/` — Execution Layer
[**README**](execution/README.md)

- **Purpose:** OrderIntent → Fill with transaction cost modeling
- **Key files:** `models.py` (OrderIntent, Fill), `costs.py` (CostConfig)
- **Features:** Spread + slippage, 1-bar delay, latency simulation
- **Integration:** Execution engine in `TradingEngine.process_bar()`

**Status:** ✓ Production-ready

---

#### `risk/` — Risk Management
[**README**](risk/README.md)

- **Purpose:** Three-tier protection (global DD halt, daily/monthly limits)
- **Key files:** `manager.py`, `drawdown_tracker.py`, `action.py`
- **Features:** Risk override, calendar-day/month boundary tracking
- **Integration:** Risk check between MTM and strategy signal in per-bar pipeline

**Status:** ✓ Production-ready

---

### Backtest Engine

#### `engine/` — Backtest Engine
[**README**](engine/README.md)

- **Purpose:** Orchestrate 8-stage backtest pipeline + per-bar processing
- **Key files:** `runner.py` (8-stage pipeline), `core.py` (TradingEngine)
- **Pipeline:** Config → Data → Validation → Features → Replay → Metrics → Artifacts
- **Per-bar:** Execution → MTM → Risk → Signal → Decision → OrderIntent
- **Walk-forward:** Fold splitter, capital carryover, fold aggregation

**Status:** ✓ Production-ready

---

#### `validation/` — Walk-Forward Splitter
[**README**](validation/README.md)

- **Purpose:** Leakage-safe time series cross-validation
- **Key files:** `time_series_splitter.py`
- **Modes:** Expanding (anchored), rolling (sliding window)
- **Guarantees:** `train.max() < test.min()`, non-overlapping test folds

**Status:** ✓ Production-ready

---

### Research Workflows

#### `robustness/` — Robustness Testing
[**README**](robustness/README.md)

- **Purpose:** Campaign-based stress testing + promotion gates
- **Key files:** `spec.py`, `campaign.py`, `expand.py`, `promotion.py`, `regimes.py`, `summarize.py`
- **Features:** Cost stress, latency proxy, parameter sweeps, regime-stratified performance
- **Artifacts:** `regime_summary.csv`, `promotion_report.md`, `robustness_report.md`

**Status:** ✓ Production-ready

---

#### `regime/` — Regime Detection
[**README**](regime/README.md)

- **Purpose:** HMM-based trend/range classification for regime-gated strategies
- **Key files:** `protocol.py`, `filter.py`, `hmm_model.py`
- **Models:** HMM (primary), Choppiness, Hurst, GMM, KMeans, PELT
- **Features:** ER, CHOP, Hurst → z-scored → 2-state Gaussian HMM
- **Integration:** `DonchianATRRegime` strategy, regime.csv artifacts

**Status:** ✓ Production-ready

---

#### `eval/` — Evaluation Suite
[**README**](eval/README.md)

- **Purpose:** Canonical trio comparison, run comparison tools
- **Key files:** `cli.py`
- **Features:** Rule-based vs Logistic vs LightGBM comparison
- **Usage:** `scripts/run_canonical_trio.py`, `scripts/compare_runs.py`

**Status:** ✓ Production-ready

---

### Reporting

#### `reporting/` — Metrics & Visualization
[**README**](reporting/README.md)

- **Purpose:** Compute metrics and generate plots
- **Key files:** `metrics.py`, `plots.py`
- **Metrics:** Net P&L, Sharpe, Calmar, drawdown, win rate, etc.
- **Plots:** Equity curve, close price, fold P&L, gross vs net
- **Backend:** Non-interactive Agg (headless)

**Status:** ✓ Production-ready

---

### Utilities

#### `ev/` — Expected Value Simulator
**Purpose:** Monte-Carlo EV simulation for trade expectancy concepts
**Status:** ✓ Week 1 deliverable
**Usage:** `uv run python scripts/run_ev_sim.py`

---

## Data Flow Diagram

```
MT5 Terminal
    ↓
[Loader] → data/<snapshot>/
    ↓
[BarIterator + validate_bars()] → Clean bars
    ↓
[FeaturePipeline] → Features (X_*)
    ↓
[TimeSeriesSplitter] → Train/Test folds (walk-forward)
    ↓
[build_forecast_frame()] → ForecastFrame (X_* + y)
    ↓
[Predictor.fit() / Calibrator] → Trained model
    ↓
[Strategy.on_bar()] → Signal (direction + strength)
    ↓
[TradingEngine._decide()] → OrderIntent (target position)
    ↓
[_execute_order()] → Fill (completed trade)
    ↓
[RiskManager.update()] → RiskAction (halt/flatten/pass)
    ↓
[compute_metrics()] → Metrics (Sharpe, DD, etc.)
    ↓
[write_artifacts()] → runs/<timestamp>/
```

## Running Tests / Scripts

All scripts should be run via `uv` to ensure dependency isolation:

```bash
# Run the Data Loader
uv run loader

# Run a Backtest
uv run python -m src backtest --config configs/w4_rule_baseline.yaml

# Run EV Simulator
uv run python scripts/run_ev_sim.py --config configs/ev_sim_base.yaml

# Run Robustness Campaign
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml

# Run Canonical Trio
uv run python scripts/run_canonical_trio.py

# Run All Tests
uv run pytest

# Run Specific Test File
uv run pytest tests/test_feature_pipeline.py
```

## Key Design Patterns

### Protocol-Based Pluggability

Core interfaces are defined as `Protocol` classes (duck typing):
- `DataSource`, `DataWriter` (loader)
- `Indicator`, `Transform` (features)
- `Strategy` (strategies)
- `Predictor` (ML models)
- `Policy` (position sizing)
- `RegimeDetector` (regime detection)

**Benefit:** New implementations don't require modifying core code.

### Registry Pattern

Strategies, models, and features are registered in registries:
- `strategy/registry.py`: Maps `type` strings to strategy builders
- Auto-registration on module import (no manual registration needed)

**Benefit:** Config-driven instantiation (YAML `type` → object).

### Signal/Intent Separation

- **Signal**: What the model believes (direction + strength + reason)
- **OrderIntent**: What to execute (target position)

**Benefit:** Decouples forecasting (strategy) from execution (engine). Position sizing and risk gates happen in the engine, not the strategy.

### Frozen Dataclasses

All value objects are immutable:
- `Signal`, `OrderIntent`, `Fill`, `RiskAction`, `StrategyContext`
- Config objects: `RiskConfig`, `CostConfig`, `FeatureSpec`

**Benefit:** Prevents accidental mutation, easier debugging.

### Determinism Enforcement

- Timestamps canonicalized to `datetime64[ns, UTC]`
- Feature columns sorted alphabetically
- Duplicate bars removed (keeps first)
- Explicit random seeds for ML models

**Benefit:** Same seed + config + data = identical results (critical for research).

## See Also

- [**CLAUDE.md**](../CLAUDE.md) — Engineering standards and operating manual
- [**Main README**](../README.md) — Project overview and quick start
- [**tests/README.md**](../tests/README.md) — Test suite documentation
- [**data/README.md**](../data/README.md) — Data snapshot structure
- [**configs/README.md**](../configs/README.md) — Configuration examples
