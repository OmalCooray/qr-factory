# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

qr-factory is a local-first quantitative research factory for **adaptive, uncertainty-aware, execution-aware intraday trading of XAUUSD (gold)**. Built as the engine behind an MSc research project (University of Moratuwa — MSc Data Science & AI).

The system covers data ingestion from MetaTrader 5, event-driven bar replay, feature/indicator pipelines, risk management, and deterministic backtesting. It is being extended toward probabilistic forecasting, conformal calibration, and execution-aware decision policies. Python 3.12+, managed with `uv`.

**Core principles:** determinism (same seed + config + data = same result), simplicity (explicit event loops, no hidden magic), artifacts (every run leaves a trace with config snapshot and git hash), predict ≠ decide (forecasting and trading decisions are separate concerns).

## North-Star Architecture

The full research pipeline this codebase is converging toward:

```
Data → Features → Forecaster → Calibration → Policy → Execution → Risk/Metrics
```

| Pipeline Stage | Description | Package | Status |
|---|---|---|---|
| **Data** | MT5 OHLCV ingestion, validation, bar replay | `src/loader/`, `src/replay/` | Done |
| **Features** | OHLCV-derived indicators, time-of-day encoding | `src/indicators/` | Done (basic), needs time-of-day features |
| **Forecaster** | Probabilistic models (LSTM/GRU/TCN/Transformer with distributional heads) | `src/strategy/` → will become `src/forecaster/` | Rule-based baselines done; ML models not yet started |
| **Calibration** | Weighted conformal prediction for calibrated prediction intervals under non-stationarity | Not yet created → `src/calibration/` | Not started |
| **Policy** | Uncertainty-aware trade filtering, position sizing, order type selection | Currently `TradingEngine._decide()` → will become `src/policy/` | Minimal; needs uncertainty-driven logic |
| **Execution** | OrderIntent → Fill with transaction cost model (spread + slippage) | `src/execution/` | OrderIntent/Fill done; cost model not started |
| **Risk** | Drawdown tracking, daily/monthly limits, circuit breakers | `src/risk/` | Done |
| **Metrics/Tracking** | Run artifacts, MLflow integration, PBO overfitting checks | `src/engine/`, `src/reporting/` | Basic artifacts done; MLflow/PBO not started |

### Key Research Contributions (from PG Dip report)

1. **Distributional forecasting** — models output predictive distributions (quantiles, density params), not point forecasts
2. **Conformal calibration under non-stationarity** — rolling weighted conformal prediction maintains coverage across sessions/regimes
3. **Execution-aware decisions** — calibrated uncertainty drives trade filtering, sizing, and order type selection under realistic transaction costs
4. **Robust evaluation** — walk-forward backtesting with PBO overfitting controls, session-stratified reporting

## Commands

```bash
# Install / sync dependencies
uv sync

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_validate_bars.py

# Run tests by keyword
uv run pytest -k "determinism"

# Load data from MT5
uv run loader

# Run a backtest
uv run python -m src backtest --config configs/exp_ema_50_200.yaml

# Run EV simulator
uv run python scripts/run_ev_sim.py
```

No linter or formatter is configured yet. No build step required.

## Architecture (Current State)

### Entry Points

- `src/__main__.py` — unified CLI dispatcher (`loader`, `backtest` subcommands)
- `src/loader/_loader.py:main` — registered as `loader` script in pyproject.toml
- `scripts/run_ev_sim.py` — standalone EV simulator CLI

### Module Layout

- **`src/loader/`** — MT5 data pipeline. `Loader` orchestrates `DataSource` → `DataWriter` → `VersionLog`. Uses atomic file writes (temp + rename). Outputs per-year CSVs to timestamped snapshot dirs under `data/`.
- **`src/replay/`** — `BarIterator` validates, deduplicates, and sorts bars. `validate_bars()` does fail-fast integrity checks (timestamps, OHLC NaNs, spread sanity).
- **`src/indicators/`** — Protocol-based feature engineering. `FeaturePipeline` chains `FeatureSpec`s (each = base `Indicator` + list of `Transform`s). Output columns are sorted alphabetically for determinism. Implementations: SMA, EMA, ADX.
- **`src/strategy/`** — `Strategy` protocol defines `on_bar(ctx) -> Signal`. `Signal` is the model's directional belief (direction + strength + reason). The decision layer in `TradingEngine` converts Signal → OrderIntent. Implementations: `MACrossoverStrategy`, `ADXFilteredCrossoverStrategy`. Registry pattern maps YAML `type` strings to builders.
- **`src/engine/core.py`** — `TradingEngine`: mode-agnostic per-bar processing core. Orchestrates execution → MTM → risk → model/signal → decision → order intent. Both backtest and future live mode use this.
- **`src/engine/runner.py`** — `run_backtest()` orchestrates 8 pipeline stages: config → run setup → data → validation → features → replay → metrics → artifacts. Helper functions: `_load_config()`, `_create_run_dir()`, `_load_market_data()`, `_run_replay()`, `_compute_metrics()`, `_write_artifacts()`.
- **`src/execution/`** — Value objects: `OrderIntent` (what to execute) and `Fill` (completed trade).
- **`src/risk/`** — `RiskManager` orchestrates global DD halt + daily/monthly pause logic. `DrawdownTracker` for peak/trough math. `RiskAction` returned to engine per bar.
- **`src/ev/`** — Monte-Carlo expected value simulator. Configured via YAML scenarios.
- **`src/reporting/`** — matplotlib plotting (close price, equity curve). Uses non-interactive Agg backend.

### Per-Bar Pipeline (TradingEngine.process_bar)

Each bar processes in this order:
1. **Execution (at Open):** fill previous bar's OrderIntent, calculate realized P&L
2. **Mark-to-Market (at Close):** compute unrealized P&L, current equity
3. **Risk check:** RiskManager.update() → RiskAction (flatten / halt / pass)
4. **Record:** equity snapshot appended
5. **Model → Signal:** strategy.on_bar(ctx) produces Signal (direction + strength)
6. **Decision:** Signal + RiskAction → OrderIntent (sizing + risk gate)
7. **Store intent:** pending for next bar's execution phase

### Key Data Types

```
Signal(direction, strength, reason)     — model output (what the model believes)
OrderIntent(target_position, reason)    — execution instruction (what to actually do)
Fill(entry_ts, exit_ts, side, qty, ...) — completed round-trip trade
RiskAction(flatten, halted, reason)     — risk layer instruction
StrategyContext(ts, bar, features, position, equity, bar_index) — per-bar state
```

### Key Design Patterns

- **Protocol-based pluggability:** `DataSource`, `DataWriter`, `Indicator`, `Transform`, `Strategy` are all `Protocol` classes. New implementations don't require modifying core code.
- **Frozen dataclasses:** configs and value objects (`LoaderConfig`, `SMA`, `StrategyContext`, `Signal`, `OrderIntent`, `Fill`, `RiskAction`) are `@dataclass(frozen=True)`.
- **Signal/Intent separation:** strategies produce Signals (beliefs), TradingEngine converts to OrderIntents (actions). Position sizing lives in the engine, not the strategy.
- **YAML configs:** all major tools read YAML configs from `configs/`. Configs are copied into run artifacts for reproducibility.
- **Git metadata capture:** `engine/git_info.py` records commit hash and dirty state alongside run metrics.
- **Strategy registry:** `strategy/registry.py` maps type strings to builder functions. Auto-registers built-in strategies on import.

### Data Layout

- `data/<YYYYMMDD_HHMMSS>/` — snapshot dirs with `<symbol>_M5_<year>.csv` files and `META.json`
- `runs/<YYYYMMDD_HHMMSS>/` — backtest artifacts: `equity.csv`, `trades.csv`, `metrics.json`, `config.yaml`, plots, `DATA_REF.json`
- `configs/` — YAML configuration files for backtest and EV simulator

### Determinism Guarantees

- Timestamps canonicalized to `datetime64[ns, UTC]`; duplicates removed; monotonic-increasing enforced
- Feature columns sorted alphabetically (no insertion-order randomness)
- NumPy RNG seeded explicitly for EV simulations
- Git state captured in metrics for reproducibility tracking
- Drawdown cross-validated: numpy recomputation vs DrawdownTracker assertion in every run

## Planned Evolution

As the research progresses, the architecture will evolve:

1. **`src/strategy/` → `src/forecaster/`** — when ML models (LSTM/GRU/TCN/Transformer) are added, `Signal` becomes `Forecast` carrying full distributional output (quantiles, density params)
2. **`src/calibration/` (new)** — weighted conformal prediction layer between forecaster and policy
3. **`src/policy/` (new)** — extracted from `TradingEngine._decide()` when uncertainty-driven sizing logic is needed
4. **`src/execution/cost_model.py` (new)** — spread + slippage model for execution-aware backtesting

Each package is created when it gains real logic, not before.

## Engineering Operating Manual

**This section defines strict engineering rules for all future work in this repository. Following these rules is MANDATORY.**

### Core Engineering Principles

1. **Minimal patch first** — Always prefer the smallest safe patch that fixes the issue. No big rewrites. No speculative cleanup.

2. **Preserve determinism and reproducibility** — Same seed + config + data MUST produce same results. Do not break this under any circumstances.

3. **Preserve artifact contracts** — Do not change artifact schemas (equity.csv, trades.csv, metrics.json, fold_metrics.json, predictions.csv, regime.csv, etc.) unless a bug fix explicitly requires it. Changing schemas breaks result comparisons and reproducibility.

4. **Explicit exceptions, not production asserts** — Runtime correctness checks must use explicit `if` + `raise ValueError/RuntimeError`, never `assert`. Production code must work under optimized Python (`-O` flag strips asserts).

5. **Typed structures over raw dicts** — Use typed dataclasses, TypedDict, or Protocols for critical structures (fold results, config validation, metrics aggregation). Avoid error-prone ad-hoc dict plumbing with string keys.

6. **Centralize shared paths/constants** — Use `src/_paths.py:REPO_ROOT` for repository root. Do not duplicate path resolution logic across modules.

7. **Explicit contracts, no invisible coupling** — Avoid `hasattr(...)` duck typing, magic attributes, or implicit side channels. Use Protocols with explicit optional methods (e.g., `Strategy.extra_artifacts()`).

8. **Library code raises exceptions** — Library functions (in `src/`) must raise exceptions (ValueError, FileNotFoundError, RuntimeError). Only CLI entry points (`src/__main__.py`, `src/eval/cli.py`, scripts/) may call `sys.exit()`.

9. **Protocols and composition preferred** — Favor Protocol-based design. Avoid unnecessary inheritance or ABC hierarchies.

10. **No abstraction without removal of real coupling** — Do not add abstraction layers unless they directly remove a current defect or reduce real coupling.

### Research-Factory Specific Rules

1. **Predict ≠ Decide must remain explicit** — Forecasting (model beliefs) and trading decisions (execution actions) are separate concerns. Do not merge `Forecaster` into `Policy` or `Policy` into `TradingEngine`.

2. **No leakage-tolerant shortcuts** — Walk-forward integrity is non-negotiable. Train/test splits must be strictly enforced. Features must be causal (no future lookahead).

3. **Walk-forward integrity cannot be weakened** — Do not bypass leakage checks. Do not merge train and test slices. Do not allow test data to influence training.

4. **Costs/risk logic must remain explicit and testable** — Transaction costs (spread, slippage), risk management (drawdown, position limits), and position sizing must be explicit, configurable, and testable.

5. **Robustness/promotion artifacts are part of the research contract** — Campaign manifests, regime slicing, promotion gate logic, and robustness reports are core outputs, not optional extras.

6. **Every meaningful change must preserve or improve reproducibility evidence** — Git hash, config snapshot, and data reference must be captured for every run.

### Required Workflow for Non-Trivial Changes

For each task beyond simple one-line fixes:

1. **Inspect existing code path first** — Read the relevant modules. Understand current behavior before proposing changes.

2. **Identify smallest safe patch** — What is the minimal change that addresses the issue without destabilizing other code?

3. **State assumptions** — What invariants are you assuming? What behavior must be preserved?

4. **Implement** — Make the minimal change. Do not refactor unrelated code.

5. **Run targeted tests** — Test the specific functionality you changed. Use `pytest tests/test_<module>.py` or `pytest -k <keyword>`.

6. **Run broader regression checks** — After targeted tests pass, run `uv run pytest` to ensure no regressions.

7. **Update docs/config/tests if contract changed** — If you changed a public API, artifact schema, or config key, update relevant documentation and tests.

### Required "Do Not Do" List

**NEVER do these things without explicit user approval:**

- No broad rewrites of working code
- No speculative architecture cleanup
- No model-zoo expansion during engineering hardening tasks
- No hidden behavior changes (all changes must be visible in tests or artifacts)
- No silent fallbacks for critical correctness paths (fail-fast instead)
- No duplicate sources of truth for paths/config semantics
- No bypassing tests after refactors ("I'll test it later" is not allowed)
- No breaking the walk-forward leakage checks
- No changing artifact schemas without documenting the change

### Required Output Standard

When you complete a task in this repository, you MUST report:

1. **Objective** — What were you asked to do?
2. **Files changed** — List each file and summarize changes.
3. **Tests run** — Which tests did you run? Did they pass?
4. **Risks** — Any residual risks or known issues?
5. **Artifact behavior preserved or changed?** — Did output schemas or artifact formats change?
6. **Tracker/docs update needed?** — Should CLAUDE.md, README, or other docs be updated?

### Escalation Path

If you encounter:

- **Ambiguous requirements** — Ask the user for clarification before proceeding.
- **Conflicts between rules** — Ask the user which rule takes precedence.
- **Proposed changes that risk determinism** — Halt and ask for approval.
- **Test failures you cannot diagnose** — Report the failure and stop.

### Examples of Good vs. Bad Patches

**GOOD** — Replace production assert with explicit exception:
```python
# Before
assert len(folds) > 0, "Splitter produced no folds"

# After
if len(folds) == 0:
    raise ValueError("Splitter produced no folds — check validation config")
```

**BAD** — Introducing unnecessary abstraction:
```python
# Before (simple, clear)
fold_result = {"fold_id": fold_id, "net_pnl": net_pnl, ...}

# After (over-engineered for no benefit)
class FoldResultBuilder:
    def __init__(self): ...
    def with_fold_id(self, fold_id): ...
    def with_net_pnl(self, net_pnl): ...
    def build(self): ...
```

**GOOD** — Typed structure for high-risk aggregation:
```python
@dataclass
class FoldResult:
    fold_id: int
    net_pnl: float
    gross_pnl: float
    # ... explicit typed fields
```

**BAD** — Rewriting working code for style reasons without functional improvement.

### Summary

This repository is a **research engine** where reproducibility, correctness, and determinism are paramount. Every change must respect these constraints. When in doubt, choose the minimal patch. When uncertain, ask before proceeding. When you break tests, stop and fix them before continuing.
