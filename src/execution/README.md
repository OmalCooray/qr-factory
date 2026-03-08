# Execution Layer

**OrderIntent → Fill conversion with transaction cost modeling.**

## Overview

The `src/execution/` package defines the execution layer data structures: **OrderIntent** (what to execute) and **Fill** (completed trade). It also implements transaction cost modeling (spread + slippage) for realistic backtesting.

**Key principle:** Execution happens at the **next bar's open price** with cost adjustments. This simulates realistic market execution with 1-bar delay (no lookahead).

## Data Structures

### OrderIntent

Represents an execution instruction (target position):

```python
@dataclass(frozen=True)
class OrderIntent:
    target_position: float    # Signed target position (+1.0 long, -1.0 short, 0.0 flat)
    reason: str = ""          # Human-readable audit trail
```

**Example:**
```python
OrderIntent(target_position=1.0, reason="MA crossover bullish")
OrderIntent(target_position=0.0, reason="Risk halt: DD > 10%")
```

**Created by:** Decision layer in `TradingEngine._decide()` (converts Signal → OrderIntent)

**Executed by:** Execution engine in `TradingEngine.process_bar()` (next bar's open price)

### Fill

Represents a completed round-trip trade:

```python
@dataclass(frozen=True)
class Fill:
    entry_ts: str             # Entry timestamp (ISO format)
    exit_ts: str              # Exit timestamp (ISO format)
    side: str                 # "long" or "short"
    qty: float                # Position size (always positive)
    entry_price: float        # Cost-adjusted entry price
    exit_price: float         # Cost-adjusted exit price
    gross_pnl: float          # P&L before transaction costs
    fees: float               # Transaction costs (spread + slippage)
    pnl: float                # Net P&L (gross_pnl - fees)
    reason: str               # Strategy-provided reason
```

**Example:**
```csv
entry_ts,exit_ts,side,qty,entry_price,exit_price,gross_pnl,fees,pnl,reason
2024-01-15 09:00:00,2024-01-15 10:30:00,long,1.0,2045.3,2047.8,2.5,0.5,2.0,MA crossover
```

**Created by:** `_execute_order()` in `TradingEngine` when position changes

**Exported to:** `trades.csv` artifact

## Transaction Cost Model

### Cost Components

**1. Spread (Bid-Ask Spread)**

Market maker's cost for providing liquidity.

**Config:**
```yaml
execution:
  spread_points: 40         # 40 points = 0.40 USD/oz for XAUUSD
```

**2. Slippage**

Execution price difference due to market impact / latency.

**Config:**
```yaml
execution:
  slippage_points: 10       # 10 points = 0.10 USD/oz
```

**3. Point Value**

Value of 1 point in account currency.

**Config:**
```yaml
execution:
  point_value: 0.01         # 1 point = 1 cent
```

### Cost-Adjusted Pricing

**Long entry (buying):**
```python
entry_price = mid_price + (spread_points / 2 + slippage_points) * point_value
```

**Long exit (selling):**
```python
exit_price = mid_price - (spread_points / 2 + slippage_points) * point_value
```

**Short entry (selling):**
```python
entry_price = mid_price - (spread_points / 2 + slippage_points) * point_value
```

**Short exit (buying):**
```python
exit_price = mid_price + (spread_points / 2 + slippage_points) * point_value
```

**Example (XAUUSD M5):**
- Mid price: 2045.00
- Spread: 40 points (0.40)
- Slippage: 10 points (0.10)
- Point value: 0.01

**Long entry:**
```python
entry_price = 2045.00 + (40/2 + 10) * 0.01
            = 2045.00 + 30 * 0.01
            = 2045.00 + 0.30
            = 2045.30
```

**Long exit (at mid 2047.00):**
```python
exit_price = 2047.00 - (40/2 + 10) * 0.01
           = 2047.00 - 0.30
           = 2046.70
```

**Net P&L:**
```python
gross_pnl = exit_mid - entry_mid = 2047.00 - 2045.00 = 2.00
fees = (entry_cost + exit_cost) = 0.30 + 0.30 = 0.60
net_pnl = gross_pnl - fees = 2.00 - 0.60 = 1.40
```

### CostConfig

```python
@dataclass(frozen=True)
class CostConfig:
    spread_points: float = 0.0
    slippage_points: float = 0.0
    point_value: float = 1.0
```

**Default:** Zero cost (frictionless backtest).

**Realistic XAUUSD M5:**
```python
CostConfig(spread_points=40.0, slippage_points=10.0, point_value=0.01)
```

## Execution Timing

### Per-Bar Execution Flow

```
Bar T-1:  Strategy generates Signal → Decision layer creates OrderIntent → Queue intent
Bar T:    Execute intent at Open (cost-adjusted) → Update position → Record Fill
```

**Diagram:**
```
Bar T-1:  [Close: 2045.00] → Signal(long) → OrderIntent(target=1.0) → Queue
Bar T:    [Open: 2045.20] → Execute @ 2045.50 (cost-adjusted) → position=1.0
```

**Key:** 1-bar delay ensures no lookahead (cannot use bar T's close to execute at bar T's open).

### Latency Simulation

Optional `latency_bars` parameter adds extra delay:

```python
engine = TradingEngine(
    strategy=strategy,
    risk_manager=risk_manager,
    starting_capital=1000.0,
    latency_bars=1,  # Total delay = 2 bars (1 standard + 1 latency)
)
```

**Execution timing:**
```
Bar T-1:  Signal → OrderIntent → Queue
Bar T:    Intent sits in queue (latency delay)
Bar T+1:  Execute intent at Open
```

**Use case:** Simulate network latency / decision-to-execution delay in robustness testing.

## Config Examples

### Minimal (Zero Cost)

```yaml
execution:
  spread_points: 0
  slippage_points: 0
  point_value: 1.0
```

**Use case:** Frictionless baseline (isolates strategy logic from execution costs).

### Realistic XAUUSD M5

```yaml
execution:
  spread_points: 40       # 0.40 USD/oz typical spread
  slippage_points: 10     # 0.10 USD/oz slippage
  point_value: 0.01       # 1 point = 1 cent
```

### Cost Stress (Robustness Testing)

```yaml
execution:
  spread_points: 80       # 2x baseline (worst-case scenario)
  slippage_points: 20
  point_value: 0.01
```

**Use case:** Test strategy robustness to increased transaction costs.

### Latency Proxy

```yaml
execution:
  spread_points: 40
  slippage_points: 10
  point_value: 0.01
  latency_bars: 1         # 2-bar total delay
```

**Use case:** Simulate execution delays in robustness campaigns.

## Integration with Engine

### Execution in process_bar()

```python
# TradingEngine.process_bar()

# 1. Pop ready intent from queue
if len(self._intent_queue) > self._latency_bars:
    ready_intent = self._intent_queue.popleft()
    target_position = ready_intent.target_position
else:
    target_position = self._position  # Hold current

# 2. Execute order
fills = _execute_order(
    target_position=target_position,
    current_position=self._position,
    entry_price=self._entry_price,
    bar_open=bar_row["open"],
    cost_config=self.cost_config,
)

# 3. Update state
for fill in fills:
    self._equity += fill.pnl
    self.all_fills.append(fill)
```

### _execute_order() Logic

```python
def _execute_order(
    target_position: float,
    current_position: float,
    entry_price: float,
    entry_mid: float,
    entry_ts: str,
    bar_open: float,
    bar_ts: str,
    cost_config: CostConfig,
) -> tuple[float, float, float, str, list[Fill]]:
    """Execute order and return (new_position, new_entry_price, new_entry_mid, new_entry_ts, fills)."""

    fills = []

    # Case 1: No position change
    if target_position == current_position:
        return current_position, entry_price, entry_mid, entry_ts, fills

    # Case 2: Flatten or reverse position → close existing
    if current_position != 0:
        exit_price_adj = cost_adjusted_price(
            mid=bar_open,
            side="sell" if current_position > 0 else "buy",
            config=cost_config,
        )
        pnl = compute_pnl(current_position, entry_price, exit_price_adj)
        gross_pnl = compute_pnl(current_position, entry_mid, bar_open)
        fees = gross_pnl - pnl

        fills.append(
            Fill(
                entry_ts=entry_ts,
                exit_ts=bar_ts,
                side="long" if current_position > 0 else "short",
                qty=abs(current_position),
                entry_price=entry_price,
                exit_price=exit_price_adj,
                gross_pnl=gross_pnl,
                fees=fees,
                pnl=pnl,
                reason="Exit",
            )
        )

    # Case 3: Open new position (if target != 0)
    if target_position != 0:
        entry_price_adj = cost_adjusted_price(
            mid=bar_open,
            side="buy" if target_position > 0 else "sell",
            config=cost_config,
        )
        return target_position, entry_price_adj, bar_open, bar_ts, fills

    # Case 4: Flattened (target = 0)
    return 0.0, 0.0, 0.0, "", fills
```

## Key Files

| File | Description |
|------|-------------|
| `models.py` | OrderIntent, Fill dataclasses |
| `costs.py` | CostConfig, cost_adjusted_price() |

## See Also

- [Backtest Engine](../engine/README.md) — How execution integrates with per-bar pipeline
- [Strategy System](../strategy/README.md) — How strategies produce signals (not OrderIntents)
- [Robustness Testing](../robustness/README.md) — Cost stress and latency proxy scenarios
