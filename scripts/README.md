# Utility Scripts

**Helper scripts for common workflows: robustness campaigns, canonical trio, run comparison, etc.**

## Overview

The `scripts/` directory contains standalone utility scripts for common research workflows. Each script is invocable via `uv run python scripts/<script_name>.py`.

## Scripts

### run_robustness_campaign.py

**Purpose:** Execute robustness campaigns (stress testing across cost variations, latency, parameter sweeps)

**Usage:**

**Dry-run (validate manifest):**
```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml --dry-run
```

**Execute campaign:**
```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml
```

**Finalize and evaluate:**
```bash
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml \
  --finalize runs/robustness/w8_probsized_20260308_143000
```

**Outputs:**
- `runs/robustness/<campaign_id>_<timestamp>/manifest.json`
- `regime_summary.csv`
- `promotion_report.md`
- `robustness_report.md`

**See:** [Robustness Testing](../src/robustness/README.md)

---

### run_canonical_trio.py

**Purpose:** Run canonical trio comparison (rule-based, logistic, LightGBM)

**Usage:**
```bash
uv run python scripts/run_canonical_trio.py
```

**Outputs:**
- `runs/comparisons/<timestamp>/rule_baseline/`
- `runs/comparisons/<timestamp>/logistic_calibrated/`
- `runs/comparisons/<timestamp>/lgbm_baseline/`
- `runs/comparisons/<timestamp>/comparison_report.md`

**See:** [Evaluation Suite](../src/eval/README.md)

---

### compare_runs.py

**Purpose:** Compare two backtest runs side-by-side

**Usage:**
```bash
uv run python scripts/compare_runs.py runs/20260308_143022 runs/20260308_154533
```

**Output:** Terminal output with metric deltas

---

### run_ev_sim.py

**Purpose:** Expected value simulator CLI (Week 1 deliverable)

**Usage:**
```bash
uv run python scripts/run_ev_sim.py --config configs/ev_sim_base.yaml
```

**Purpose:** Monte-Carlo simulation for trade expectancy concepts.

**See:** `src/ev/README.md` (if exists)

---

## Common Patterns

### Running Multiple Campaigns

```bash
# Campaign 1: Baseline
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_probsized.yaml

# Campaign 2: Conservative risk
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_conservative_risk.yaml

# Campaign 3: Aggressive entry
uv run python scripts/run_robustness_campaign.py configs/w8_robustness_aggressive_entry.yaml
```

### Comparing Campaign Results

```bash
# Compare baseline vs conservative risk
uv run python scripts/compare_runs.py \
  runs/robustness/w8_probsized_20260308_143000/baseline \
  runs/robustness/w8_conservative_20260308_150000/baseline
```

## See Also

- [Robustness Testing](../src/robustness/README.md) — Campaign harness details
- [Evaluation Suite](../src/eval/README.md) — Canonical trio comparison
- [Config Examples](../configs/README.md) — Configuration reference
