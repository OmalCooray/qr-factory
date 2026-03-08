# Risk Management

**Three-tier protection: global drawdown halt, daily loss limit, monthly loss limit.**

## Overview

The `src/risk/` package implements risk management controls that override strategy signals when risk thresholds are breached. The `RiskManager` enforces three levels of protection:

1. **Global Drawdown Halt**: Flatten and halt trading if equity drawdown exceeds threshold
2. **Daily Loss Limit**: Pause trading until next day if daily loss exceeds threshold
3. **Monthly Loss Limit**: Pause trading until next month if monthly loss exceeds threshold

**Key principle:** Risk checks run **after MTM** but **before strategy signal** in the per-bar pipeline. This ensures risk limits are checked before new trades can be generated.

## Architecture

### RiskAction

Output of `RiskManager.update()`:

```python
@dataclass(frozen=True)
class RiskAction:
    flatten: bool = False    # Force close all positions
    halted: bool = False     # Stop trading for remainder of run
    reason: str = ""         # Human-readable reason
```

**Examples:**
```python
RiskAction(flatten=False, halted=False, reason="")  # Pass (no risk breach)
RiskAction(flatten=True, halted=True, reason="Global DD halt: 10.2% > 10.0%")
RiskAction(flatten=False, halted=False, reason="Daily limit paused")
```

**Integration:** `TradingEngine` decision layer checks `RiskAction` before converting Signal → OrderIntent.

### RiskConfig

Configuration for risk thresholds:

```python
@dataclass(frozen=True)
class RiskConfig:
    max_drawdown_pct: float | None = None     # Global DD halt threshold (%)
    daily_dd_limit: float | None = None       # Daily loss limit (account currency)
    monthly_dd_limit: float | None = None     # Monthly loss limit (account currency)
```

**Example:**
```yaml
risk:
  global_dd_halt_pct: 10.0       # Halt if DD > 10%
  daily_loss_limit: 500.0        # Pause if daily loss > $500
  monthly_loss_limit: 2000.0     # Pause if monthly loss > $2000
```

## Three-Tier Protection

### Tier 1: Global Drawdown Halt

**Trigger:** Equity drawdown from peak exceeds `max_drawdown_pct`

**Action:** Flatten all positions + halt trading for remainder of backtest

**Logic:**
```python
current_dd_pct = (peak_equity - current_equity) / peak_equity * 100
if current_dd_pct > max_drawdown_pct:
    return RiskAction(flatten=True, halted=True, reason=f"Global DD halt: {current_dd_pct:.1f}% > {max_drawdown_pct}%")
```

**Example:**
- Starting capital: $10,000
- Peak equity: $12,000 (bar 500)
- Current equity: $10,500 (bar 750)
- Drawdown: (12,000 - 10,500) / 12,000 = 12.5%
- Threshold: 10.0%
- **Result:** Halt triggered at bar 750

**Permanent:** Once halted, no new trades for rest of backtest.

### Tier 2: Daily Loss Limit

**Trigger:** Realized P&L for current calendar day exceeds `daily_dd_limit`

**Action:** Pause trading until next calendar day (no new trades, but existing positions remain)

**Logic:**
```python
if bar_date != self._last_day:
    self._daily_pnl = 0.0  # Reset on new day
    self._last_day = bar_date

self._daily_pnl += realized_pnl_this_bar

if abs(self._daily_pnl) > daily_dd_limit and self._daily_pnl < 0:
    return RiskAction(flatten=False, halted=False, reason="Daily limit paused")
```

**Example:**
- Daily limit: $500
- 2024-01-15: Lose $600 by 14:00
- **Result:** Paused from 14:00 until 2024-01-16 00:00

**Temporary:** Resets at calendar day boundary (00:00 UTC).

### Tier 3: Monthly Loss Limit

**Trigger:** Realized P&L for current calendar month exceeds `monthly_dd_limit`

**Action:** Pause trading until next calendar month

**Logic:**
```python
if bar_month != self._last_month:
    self._monthly_pnl = 0.0  # Reset on new month
    self._last_month = bar_month

self._monthly_pnl += realized_pnl_this_bar

if abs(self._monthly_pnl) > monthly_dd_limit and self._monthly_pnl < 0:
    return RiskAction(flatten=False, halted=False, reason="Monthly limit paused")
```

**Example:**
- Monthly limit: $2000
- January 2024: Lose $2100 by Jan 25
- **Result:** Paused from Jan 25 until Feb 1

**Temporary:** Resets at calendar month boundary.

## DrawdownTracker

Stateful peak/trough tracker for drawdown calculation:

```python
class DrawdownTracker:
    def __init__(self):
        self._peak = 0.0
        self._current_dd_pct = 0.0

    def update(self, equity: float) -> float:
        """Update peak and return current drawdown %."""
        if equity > self._peak:
            self._peak = equity
            self._current_dd_pct = 0.0
        else:
            self._current_dd_pct = (self._peak - equity) / self._peak * 100
        return self._current_dd_pct
```

**Validation:** Drawdown is cross-validated against numpy recomputation in every run (assertion in `runner.py`).

## Execution Order in Per-Bar Pipeline

```
1. Execution (Open)     → Fill pending orders
2. MTM (Close)          → Update equity
3. Risk Check           → RiskManager.update() → RiskAction
4. Record Equity        → Append snapshot
5. Strategy Signal      → strategy.on_bar() → Signal (if not halted)
6. Decision Layer       → Signal + RiskAction → OrderIntent
```

**Key:** Risk check happens **after equity is updated** but **before strategy generates signal**.

**Risk override:**
```python
# TradingEngine._decide()
if risk_action.halted or risk_action.flatten:
    return OrderIntent(target_position=0.0, reason=risk_action.reason)

# Else: convert Signal → OrderIntent normally
```

## Config Examples

### Conservative Risk Limits

```yaml
risk:
  global_dd_halt_pct: 5.0        # Halt at 5% drawdown (tight)
  daily_loss_limit: 200.0        # Max $200 loss per day
  monthly_loss_limit: 1000.0     # Max $1000 loss per month
```

**Use case:** Small account or low risk tolerance.

### Moderate Risk Limits

```yaml
risk:
  global_dd_halt_pct: 10.0       # Halt at 10% drawdown
  daily_loss_limit: 500.0        # Max $500 loss per day
  monthly_loss_limit: 2000.0     # Max $2000 loss per month
```

**Use case:** Standard research baseline.

### Aggressive (No Limits)

```yaml
risk:
  global_dd_halt_pct: null       # No global halt
  daily_loss_limit: null         # No daily limit
  monthly_loss_limit: null       # No monthly limit
```

**Use case:** Isolate strategy performance without risk interventions.

### Global Halt Only

```yaml
risk:
  global_dd_halt_pct: 15.0       # Halt at 15% drawdown
  daily_loss_limit: null         # No daily limit
  monthly_loss_limit: null       # No monthly limit
```

**Use case:** Allow intraday/monthly drawdowns, but prevent catastrophic losses.

## Integration with Engine

### RiskManager Instantiation

```python
# runner.py
risk_config = RiskConfig(
    max_drawdown_pct=cfg.get("risk", {}).get("global_dd_halt_pct"),
    daily_dd_limit=cfg.get("risk", {}).get("daily_loss_limit"),
    monthly_dd_limit=cfg.get("risk", {}).get("monthly_loss_limit"),
)
risk_manager = RiskManager(risk_config, starting_capital)
```

### Per-Bar Update

```python
# TradingEngine.process_bar()
risk_action = self.risk_manager.update(bar_ts, current_equity)

if risk_action.reason:
    log.warning("Risk: %s at %s", risk_action.reason, bar_ts)
```

### Decision Layer Override

```python
# TradingEngine._decide()
def _decide(signal: Signal, risk_action: RiskAction) -> OrderIntent:
    if risk_action.halted or risk_action.flatten:
        return OrderIntent(target_position=0.0, reason=risk_action.reason)

    # Normal signal processing
    target = signal.direction * self.position_size
    return OrderIntent(target_position=target, reason=signal.reason)
```

## Key Files

| File | Description |
|------|-------------|
| `manager.py` | RiskManager class (three-tier logic) |
| `drawdown_tracker.py` | DrawdownTracker (peak/trough math) |
| `action.py` | RiskAction dataclass |
| `config.py` | RiskConfig dataclass |

## See Also

- [Backtest Engine](../engine/README.md) — How risk integrates with per-bar pipeline
- [Strategy System](../strategy/README.md) — How RiskAction overrides strategy signals
- [Robustness Testing](../robustness/README.md) — Stress testing with varied risk limits
