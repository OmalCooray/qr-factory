# Quantitative Research Factory (qr-factory)

**qr-factory** is a local-first quantitative research environment designed for:
1.  **Robust Data Engineering**: Reliable extraction from MT5, versioned storage, and schema validation.
2.  **Deterministic Backtesting**: Event-driven replay with guaranteed reproducibility (seed-based).
3.  **Concept Verification**: Standalone simulators (e.g., EV simulation) to validate foundational trading concepts.

## 🚀 Getting Started

### Prerequisites
- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** (Dependency manager)
- **MetaTrader 5** Terminal (installed & logged in)

### Installation

```powershell
# Clone the repo
git clone <repo-url>
cd qr-factory

# Sync dependencies
uv sync
```

## 🛠 Usage

### 1. Data Loading
Fetch data from your MT5 terminal into `data/`:

```powershell
uv run loader
```

### 2. EV Simulator (Week 1)
Run the Monte-Carlo Expected Value simulator to visualise trade expectancy:

```powershell
uv run python scripts/run_ev_sim.py
```

Check results in `notes/ev_sim_results.json` and plots in `notes/plots/`.

### 3. Backtest Runner
Execute a deterministic replay simulation:

```powershell
uv run python -m src backtest --config configs/week1_base.yaml
```

Artifacts are generated in `runs/<timestamp>_<hash>/`.

## 📂 Project Structure

- **`src/`**: Source code modules ([Read more](src/README.md)).
  - `loader`: Data pipeline.
  - `ev`: EV simulator.
  - `engine`: Backtest runner.
- **`configs/`**: YAML configuration files.
- **`data/`**: Processed market data storage.
- **`notes/`**: Research notes, learning deliverables, and ad-hoc plots.
- **`runs/`**: Backtest run artifacts (config, equity, metrics).
- **`scripts/`**: Utility scripts (e.g., `run_ev_sim.py`).

## 🧪 Philosophy
- **Determinism**: The same seed + same config + same data = exact same result, always.
- **Simplicity**: No hidden magic. Explicit event loops.
- **Artifacts**: Every run leaves a trace (config snapshot, git hash, logs).
