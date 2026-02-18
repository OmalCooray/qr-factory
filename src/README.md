# Source Code (`src`)

This directory contains the core Python packages for the **qr-factory** project.

## Modules

### `loader`
**Data Pipeline & Managment**
- Extracts M5/M1 data from MetaTrader 5 terminal.
- Validates schema and ensures atomic writes.
- Maintains `data/DATA_VERSION.md` registry.
- **Usage**: `uv run loader`

### `ev`
**Expected Value Simulator** (Week 1 Deliverable)
- Standalone Monte-Carlo simulator for trade expectancy concepts.
- Demonstrates relationship between win rate, expectancy, and variance.
- **Usage**: `uv run python scripts/run_ev_sim.py`

### `replay`
**Deterministic Replay**
- `BarIterator`: Yields bars one-by-one for event-driven backtesting.
- Ensures cross-run reproducibility via canonical timestamp parsing.

### `engine`
**Execution Engine**
- `Runner`: Orchestrates the backtest loop (replay data → strategy → metrics).
- Generates artifacts in `runs/<run_id>/`.

### `reporting`
**Metrics & Visualisation**
- Generates equity curves and performance plots.

## Running Tests / Scripts

All scripts should be run via `uv` to ensure dependency isolation:

```powershell
# Run the EV simulator
uv run python scripts/run_ev_sim.py --config configs/week1_base.yaml

# Run the Data Loader
uv run loader

# Run the Backtest Engine (Runner)
uv run python -m src backtest --config configs/week1_base.yaml
```
