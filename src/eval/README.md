# Evaluation Suite

**Canonical trio comparison and run comparison tools for controlled ablation studies.**

## Overview

The `src/eval/` package provides tools for comparing backtest runs and conducting controlled ablation studies. The **canonical trio** is a standard set of three baseline strategies used for infrastructure validation and model comparison.

## Canonical Trio

Three baseline strategies for controlled comparison:

1. **Rule-Based**: EMA crossover (no ML, no calibration)
2. **Logistic**: Logistic regression with Platt scaling
3. **LightGBM**: LightGBM with Platt scaling

**Purpose:** Controlled ablation to isolate ML contribution vs feature engineering.

**Config files:**
- `configs/w4_rule_baseline.yaml` (EMA crossover)
- `configs/w6_logistic_calibrated.yaml` (Logistic + Platt)
- `configs/exp_ml_lgbm_baseline.yaml` (LightGBM + Platt)

## Running Canonical Trio

```bash
uv run python scripts/run_canonical_trio.py
```

**Output:**
```
runs/comparisons/<timestamp>/
├── rule_baseline/          # EMA crossover run
├── logistic_calibrated/    # Logistic + Platt run
├── lgbm_baseline/          # LightGBM + Platt run
└── comparison_report.md    # Side-by-side metric comparison
```

### comparison_report.md

```markdown
# Canonical Trio Comparison

**Generated:** 2026-03-08T14:45:00Z

## Summary

| Strategy | Net PnL | Sharpe | Max DD% | Win Rate | Trades |
|----------|---------|--------|---------|----------|--------|
| Rule-Based | 1234.56 | 0.89 | 12.3% | 0.48 | 156 |
| Logistic | 1567.89 | 1.15 | 9.2% | 0.51 | 234 |
| LightGBM | 1789.01 | 1.34 | 8.1% | 0.53 | 267 |

## Insights

- LightGBM outperforms rule-based by +45% net PnL
- Calibration improves Sharpe by +30% (Logistic vs Rule)
- Trade count increases with ML (more frequent signals)
```

**Use case:** Before/after validation when changing infrastructure (e.g., modifying feature pipeline, changing execution logic).

## Manual Run Comparison

Compare two specific runs:

```bash
uv run python scripts/compare_runs.py runs/20260308_143022 runs/20260308_154533
```

**Output:**
```
=== Run Comparison ===

Run A: runs/20260308_143022
Run B: runs/20260308_154533

Metric           | Run A      | Run B      | Delta      | % Change
-----------------|------------|------------|------------|----------
Net PnL          | 1234.56    | 1567.89    | +333.33    | +27.0%
Sharpe Ratio     | 1.23       | 1.45       | +0.22      | +17.9%
Max Drawdown %   | 8.5%       | 6.2%       | -2.3%      | -27.1%
Win Rate         | 0.52       | 0.55       | +0.03      | +5.8%
Trades           | 234        | 267        | +33        | +14.1%
```

**Use case:** A/B testing different hyperparameters or strategies.

## Key Files

| File | Description |
|------|-------------|
| `cli.py` | CLI entry points for comparison tools |

## See Also

- [Robustness Testing](../robustness/README.md) — Campaign-based stress testing
- [Backtest Engine](../engine/README.md) — How runs are executed
