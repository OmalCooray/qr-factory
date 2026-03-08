# Robustness Testing & Promotion Gates

**Campaign-based stress testing harness for evaluating strategy robustness across cost variations, latency proxies, and parameter sweeps.**

## Overview

The `src/robustness/` package implements a systematic stress-testing framework for trading strategies. It runs a baseline strategy configuration against multiple adverse scenarios (cost stress, latency delays, parameter perturbations) and evaluates whether the strategy meets promotion gate criteria for forward testing or production deployment.

**Core workflow:**
```
1. Define campaign (baseline + stress scenarios)
2. Execute campaign (run baseline + all scenarios)
3. Normalize results (load run artifacts into structured format)
4. Evaluate promotion gate (deterministic pass/fail based on thresholds)
5. Generate reports (robustness report + promotion verdict)
```

**Key outputs:**
- `regime_summary.csv`: Performance by market regime (trend/range)
- `promotion_report.md`: Pass/fail verdict with detailed diagnostics
- `robustness_report.md`: Comprehensive stress test analysis

## Architecture

### Campaign Concept

A **robustness campaign** consists of:

1. **Baseline**: The candidate strategy config (e.g., `w7_policy_probsized.yaml`)
2. **Cost Stress Scenarios**: Multipliers on spread/slippage (0.5x, 1.5x, 2.0x)
3. **Latency Proxy Scenarios**: Execution delay simulation (1-bar, 2-bar delay)
4. **Parameter Sweep Scenarios**: Grid search over key hyperparameters (p_enter, p_full, min_hold_bars)

Each scenario is a **deep merge** of overrides onto the baseline config.

**Example scenario families:**

| Family | Purpose | Example Scenarios |
|--------|---------|-------------------|
| **cost_stress** | Test sensitivity to transaction costs | 1.5x, 2.0x spread/slippage |
| **latency_proxy** | Test robustness to execution delays | 1-bar delay, 2-bar delay |
| **parameter_sweep** | Test sensitivity to hyperparameters | p_enter ∈ {0.35, 0.40, 0.45} |

### Data Structures

#### RobustnessScenario

```python
@dataclass(frozen=True)
class RobustnessScenario:
    scenario_id: str        # Unique ID (e.g., "cost_stress__1.5x")
    family: str             # Scenario family (cost_stress, latency_proxy, etc.)
    name: str               # Human-readable name
    description: str        # What this scenario tests
    overrides: dict         # Nested config overrides (deep-merged onto baseline)
    tags: tuple[str, ...]   # Optional metadata tags
    enabled: bool           # Whether to include in campaign
```

#### CampaignSpec

```python
@dataclass(frozen=True)
class CampaignSpec:
    campaign_id: str                   # Unique campaign identifier
    baseline_config: str               # Path to baseline YAML
    description: str                   # Campaign description
    scenarios: tuple[RobustnessScenario, ...]  # Ordered list of scenarios
    baseline_run_id: str | None        # Optional: reference existing baseline run
```

#### NormalizedCampaign

Loaded campaign results with structured metrics:

```python
@dataclass
class NormalizedCampaign:
    campaign_id: str
    baseline_reference: str            # Baseline run ID
    baseline_metrics: dict             # Aggregated baseline metrics
    scenario_results: list[ScenarioResult]  # Per-scenario metrics
    regime_breakdown: pd.DataFrame | None   # Regime-stratified performance
```

## Campaign Configuration

### YAML Schema

```yaml
campaign_id: "w8_probsized"
baseline_config: "configs/w7_policy_probsized.yaml"
description: "Robustness campaign for probability-sized policy baseline"

scenarios:
  # Cost stress (spread/slippage multipliers)
  - family: cost_stress
    name: 0.5x_cost
    description: "Half baseline friction (best-case scenario)"
    overrides:
      execution:
        spread_points: 20      # 0.5x baseline (40)
        slippage_points: 5     # 0.5x baseline (10)

  - family: cost_stress
    name: 1.5x_cost
    description: "1.5x baseline friction"
    overrides:
      execution:
        spread_points: 60      # 1.5x baseline
        slippage_points: 15

  - family: cost_stress
    name: 2.0x_cost
    description: "2x baseline friction (worst-case scenario)"
    overrides:
      execution:
        spread_points: 80
        slippage_points: 20

  # Latency proxy (execution delay simulation)
  - family: latency_proxy
    name: 1bar_delay
    description: "1-bar execution delay (2-bar total)"
    overrides:
      execution:
        latency_bars: 1        # Adds 1 bar to standard 1-bar delay

  # Parameter sweep (grid search)
  - family: parameter_sweep
    name_prefix: p_enter       # Auto-generates scenario names
    description: "Entry threshold sensitivity"
    grid:
      strategy.policy.p_enter: [0.35, 0.40, 0.45]

  - family: parameter_sweep
    name_prefix: p_full
    description: "Full position threshold sensitivity"
    grid:
      strategy.policy.p_full: [0.45, 0.50, 0.55]
```

**Deep merge logic:** Nested keys in `overrides` are recursively merged onto baseline:

```python
# Baseline
execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01

# Override (1.5x cost)
overrides:
  execution:
    spread_points: 60
    slippage_points: 15

# Result (deep merge)
execution:
  spread_points: 60          # Overridden
  slippage_points: 15        # Overridden
  point_value: 0.01          # Preserved from baseline
```

### Parameter Sweep Grid Expansion

Grid specs auto-expand to individual scenarios:

```yaml
- family: parameter_sweep
  name_prefix: p_enter
  grid:
    strategy.policy.p_enter: [0.35, 0.40, 0.45]
```

**Expands to 3 scenarios:**
- `parameter_sweep__p_enter_0`: `p_enter = 0.35`
- `parameter_sweep__p_enter_1`: `p_enter = 0.40`
- `parameter_sweep__p_enter_2`: `p_enter = 0.45`

Multi-parameter grids expand to Cartesian product:

```yaml
grid:
  strategy.policy.p_enter: [0.35, 0.45]
  strategy.policy.p_full: [0.45, 0.55]
```

**Expands to 4 scenarios (2×2 grid).**

## Running a Campaign

### Step 1: Dry-Run (Validate Manifest)

```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml --dry-run
```

**Output:**
- Validates campaign YAML schema
- Expands grid scenarios
- Prints scenario matrix (no backtests run)

**Purpose:** Catch config errors before expensive backtest execution.

### Step 2: Execute Campaign

```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml
```

**Flow:**
1. Load campaign spec from YAML
2. Expand grid scenarios
3. **Run baseline** (if not referencing existing run)
4. **Run all enabled scenarios** in sequence
5. Write manifest to `runs/robustness/<campaign_id>_<timestamp>/manifest.json`

**Manifest structure:**
```json
{
  "campaign_id": "w8_probsized",
  "baseline_reference": "runs/20260308_143022",
  "scenario_runs": {
    "cost_stress__1.5x": "runs/20260308_143156",
    "latency_proxy__1bar_delay": "runs/20260308_143234",
    ...
  },
  "campaign_dir": "runs/robustness/w8_probsized_20260308_143000"
}
```

**Artifact location:**
```
runs/robustness/<campaign_id>_<timestamp>/
├── manifest.json          # Campaign manifest (baseline + scenario run IDs)
├── regime_summary.csv     # (generated in finalize step)
├── promotion_report.md    # (generated in finalize step)
└── robustness_report.md   # (generated in finalize step)
```

### Step 3: Finalize and Evaluate

```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml \
  --finalize runs/robustness/w8_probsized_20260308_143000
```

**Flow:**
1. Load campaign manifest
2. Load run artifacts (metrics, predictions, fold results) for baseline + all scenarios
3. Normalize into `NormalizedCampaign` structure
4. Compute regime-stratified performance (trend vs range)
5. Evaluate promotion gate (apply thresholds)
6. Generate reports (`regime_summary.csv`, `promotion_report.md`, `robustness_report.md`)

**Artifacts generated:**

#### `regime_summary.csv`

Performance by market regime (trend/range):

```csv
scenario_id,regime,n_bars,net_pnl,sharpe_ratio,win_rate,max_dd_pct
baseline,TREND,123456,1234.56,1.23,0.52,8.5
baseline,RANGE,234567,-234.56,-0.45,0.42,12.3
cost_stress__1.5x,TREND,123456,987.65,0.98,0.50,9.2
...
```

**Purpose:** Detect regime-specific failures (e.g., profitable in trend but catastrophic in range).

#### `promotion_report.md`

Pass/fail verdict with detailed diagnostics:

```markdown
# Promotion Gate Report

**Campaign:** w8_probsized
**Baseline:** runs/20260308_143022
**Verdict:** PASS

## Summary
- Rules Passed: 5
- Rules Failed: 0
- Rules Skipped: 1

## Rule Details

### Rule: baseline_net_pnl
**Status:** PASS
**Observed:** 1234.56
**Threshold:** 0.0
**Message:** Baseline net PnL 1234.56 exceeds minimum 0.0

### Rule: baseline_sharpe
**Status:** PASS
**Observed:** 1.23
**Threshold:** 0.5
**Message:** Baseline Sharpe 1.23 exceeds minimum 0.5

### Rule: cost_stress_tolerance
**Status:** PASS
**Observed:** -20.1%
**Threshold:** -50.0%
**Message:** 1.5x cost stress net PnL drop -20.1% within tolerance -50.0%

...
```

#### `robustness_report.md`

Comprehensive stress test analysis:

```markdown
# Robustness Report

**Campaign:** w8_probsized
**Generated:** 2026-03-08T14:45:00Z

## Baseline Performance
- Net PnL: 1234.56
- Sharpe Ratio: 1.23
- Max Drawdown: 8.5%
- Win Rate: 52%

## Cost Stress Results
| Scenario | Net PnL | PnL Change | Sharpe | DD% |
|----------|---------|------------|--------|-----|
| 0.5x cost | 1456.78 | +18.0% | 1.45 | 7.2% |
| 1.5x cost | 987.65 | -20.0% | 0.98 | 9.2% |
| 2.0x cost | 765.43 | -38.0% | 0.67 | 11.5% |

## Latency Proxy Results
| Scenario | Net PnL | PnL Change | Sharpe | DD% |
|----------|---------|------------|--------|-----|
| 1bar delay | 1098.76 | -11.0% | 1.08 | 9.1% |

## Parameter Sweep Results
### p_enter sensitivity
| p_enter | Net PnL | PnL Change | Sharpe | Win Rate |
|---------|---------|------------|--------|----------|
| 0.35 | 1123.45 | -9.0% | 1.15 | 0.50 |
| 0.40 | 1234.56 | 0.0% (baseline) | 1.23 | 0.52 |
| 0.45 | 1198.76 | -2.9% | 1.19 | 0.53 |

## Regime Breakdown
[Link to regime_summary.csv]
```

## Promotion Gate Logic

The promotion gate evaluates 6 rules:

### Rule 1: Baseline Minimum Thresholds

**1a. Baseline Net PnL**
```python
net_pnl >= min_net_pnl  # Default: 0.0
```

**1b. Baseline Sharpe Ratio**
```python
sharpe >= min_sharpe  # Default: 0.0
```

**1c. Baseline Max Drawdown**
```python
max_dd_pct <= max_drawdown_pct  # Default: 35.0%
```

### Rule 2: Cost Stress Tolerance

```python
cost_stress_1.5x_net_pnl_drop_pct <= max_cost_stress_net_pnl_drop_pct  # Default: -50.0%
```

**Logic:** 1.5x cost scenario must not drop more than 50% from baseline net PnL.

**Example:**
- Baseline net PnL: 1234.56
- 1.5x cost net PnL: 987.65
- Drop: (987.65 - 1234.56) / 1234.56 = -20.0%
- Threshold: -50.0%
- **Result:** PASS (-20.0% > -50.0%)

### Rule 3: Latency Tolerance

```python
latency_1bar_net_pnl_drop_pct <= max_latency_net_pnl_drop_pct  # Default: -30.0%
```

**Logic:** 1-bar delay scenario must not drop more than 30% from baseline net PnL.

### Rule 4: Parameter Stability

```python
stable_sweep_fraction >= min_stable_sweep_fraction  # Default: 0.60
```

**Definition:** A sweep variant is "stable" if its net PnL and Sharpe are within ±30% of baseline.

**Logic:** At least 60% of parameter sweep variants must be stable.

**Example:**
- 5 sweep variants: p_enter ∈ {0.30, 0.35, 0.40, 0.45, 0.50}
- Baseline: p_enter = 0.40
- Stable variants: 0.35 (within 30%), 0.40 (baseline), 0.45 (within 30%)
- Stable fraction: 3 / 5 = 0.60
- **Result:** PASS (0.60 >= 0.60)

### Rule 5: Regime Consistency (Optional)

**5a. Trend Regime Profitability**
```python
trend_regime_net_pnl > 0
```

**5b. Range Regime Not Catastrophic**
```python
range_regime_net_pnl > -0.5 * baseline_net_pnl
```

**Logic:** Strategy should be profitable in trend regimes and not lose more than 50% of baseline P&L in range regimes.

### Verdict

**Overall verdict:**
```python
if all_rules_passed:
    verdict = "PASS"
else:
    verdict = "FAIL"
```

**Confidence note:** If baseline Sharpe < 1.0, add note: "Baseline Sharpe is low (<1.0). Forward test with caution."

## Config Customization

### Promotion Gate Thresholds

Create `promotion_gate_config.yaml`:

```yaml
min_net_pnl: 500.0           # Require at least $500 net profit
min_sharpe: 0.8              # Require Sharpe > 0.8
max_drawdown_pct: 20.0       # Max 20% drawdown
max_cost_stress_net_pnl_drop_pct: 40.0   # Allow up to 40% drop in 1.5x cost
max_latency_net_pnl_drop_pct: 25.0       # Allow up to 25% drop in 1-bar delay
min_stable_sweep_fraction: 0.70          # Require 70% of sweeps to be stable
```

**Usage:**
```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml \
  --finalize runs/robustness/w8_probsized_20260308_143000 \
  --promotion-config promotion_gate_config.yaml
```

## Regime Detection Integration

Regime performance is computed by slicing predictions by regime label:

1. Load `predictions.csv` from ML runs (contains `yhat`, `y`, `position`)
2. Load `regime.csv` if strategy exports regime assignments (e.g., `DonchianATRRegime`)
3. Join on timestamp
4. Aggregate metrics by regime (TREND vs RANGE)
5. Write `regime_summary.csv`

**Requirements:**
- Strategy must export `regime.csv` via `extra_artifacts()` protocol
- Regime CSV must have columns: `time`, `regime` (TREND/RANGE)

**Example regime.csv:**
```csv
time,regime,confidence,raw_value
2024-01-15 09:00:00,TREND,0.85,0.72
2024-01-15 09:05:00,TREND,0.82,0.68
2024-01-15 09:10:00,RANGE,0.65,-0.23
...
```

## Key Files

| File | Description |
|------|-------------|
| `spec.py` | RobustnessScenario, CampaignSpec dataclasses |
| `campaign.py` | Campaign orchestrator (execute baseline + scenarios) |
| `expand.py` | Grid scenario expansion (Cartesian product logic) |
| `load.py` | Load campaign results into NormalizedCampaign |
| `promotion.py` | Promotion gate evaluation (rule engine) |
| `regimes.py` | Regime-stratified performance computation |
| `summarize.py` | Report generation (robustness_report.md, promotion_report.md) |

## See Also

- [Backtest Engine](../engine/README.md) — How backtests are executed
- [ML Pipeline](../ml/README.md) — How predictions are generated for regime slicing
- [Regime Detection](../regime/README.md) — How HMM regime labels are produced
- [Evaluation Suite](../eval/README.md) — Canonical trio comparison tools
