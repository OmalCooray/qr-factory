# Metrics & Reporting

**Performance metrics computation and visualization.**

## Overview

The `src/reporting/` package computes backtest metrics and generates plots. It's called in Stage 7 (Metrics) and Stage 8 (Artifacts) of the runner pipeline.

## Metrics Computation

### `compute_metrics()`

Computes headline metrics from equity and trades:

```python
from src.reporting.metrics import compute_metrics

metrics = compute_metrics(
    equity_records=[...],
    trades=[...],
    starting_capital=1000.0,
)
```

**Returns:**
```python
{
    "net_pnl": 1234.56,
    "gross_pnl": 1456.78,
    "total_fees": 222.22,
    "sharpe_ratio": 1.23,
    "calmar_ratio": 0.45,
    "max_drawdown_pct": 8.5,
    "n_trades": 234,
    "win_rate": 0.52,
    "avg_win": 15.6,
    "avg_loss": -12.3,
}
```

### Metrics Definitions

**P&L Metrics:**
- `net_pnl`: Final equity - starting capital (after costs)
- `gross_pnl`: Sum of gross P&L per trade (ignores costs)
- `total_fees`: Sum of transaction costs

**Risk-Adjusted Returns:**
- `sharpe_ratio`: Annualized Sharpe (bar-to-bar equity returns)
- `calmar_ratio`: Net P&L / Max Drawdown (absolute)

**Drawdown:**
- `max_drawdown_pct`: Worst peak-to-trough equity drawdown (%)

**Trade Statistics:**
- `n_trades`: Number of completed round-trip trades
- `win_rate`: Fraction of trades with `pnl > 0`
- `avg_win`: Average P&L of winning trades
- `avg_loss`: Average P&L of losing trades (negative)

## Plotting

### `plot_equity()`

Generates equity curve plot:

```python
from src.reporting.plots import plot_equity

plot_equity(
    equity_records=[...],
    output_path="runs/20260308_143022/plots/equity_curve.png",
)
```

**Output:**
- Dual-line plot: Net equity (after costs) vs Gross equity (no costs)
- Shows transaction cost impact visually

### `plot_close_price()`

Generates close price plot:

```python
from src.reporting.plots import plot_close_price

plot_close_price(
    bars=bars,
    output_path="runs/20260308_143022/plots/close_price.png",
)
```

**Output:**
- Line plot of close price over time

### `plot_fold_pnl()` (Walk-Forward Only)

Generates per-fold P&L bar chart:

```python
from src.reporting.plots import plot_fold_pnl

plot_fold_pnl(
    fold_results=[...],
    output_path="runs/20260308_143022/plots/fold_pnl.png",
)
```

**Output:**
- Bar chart showing net P&L per fold
- Useful for detecting fold-specific failures

### `plot_gross_vs_net_equity()` (Walk-Forward Only)

Generates gross vs net equity overlay:

```python
from src.reporting.plots import plot_gross_vs_net_equity

plot_gross_vs_net_equity(
    fold_results=[...],
    output_path="runs/20260308_143022/plots/gross_vs_net_equity.png",
)
```

**Output:**
- Dual-line plot showing cumulative gross vs net equity across folds

## Backend Configuration

**Non-interactive Agg backend** (for headless environments):

```python
import matplotlib
matplotlib.use("Agg")  # No display required
```

**Configured in:** `src/reporting/plots.py` (top of file)

## Key Files

| File | Description |
|------|-------------|
| `metrics.py` | compute_metrics() |
| `plots.py` | plot_equity(), plot_close_price(), plot_fold_pnl(), plot_gross_vs_net_equity() |

## See Also

- [Backtest Engine](../engine/README.md) — How metrics are computed in Stage 7
