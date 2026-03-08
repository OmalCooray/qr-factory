# Bar Replay & Validation

**Deterministic bar iteration with timestamp canonicalization and data integrity checks.**

## Overview

The `src/replay/` package ensures clean, validated market data before backtesting. It provides:

1. **BarIterator**: Yields bars sequentially with canonicalized timestamps
2. **validate_bars()**: Fail-fast integrity checks (timestamps, OHLC, spread sanity)

**Key principle:** Garbage in, garbage out. Data validation is the first line of defense against incorrect backtest results.

## BarIterator

Canonicalizes timestamps and yields bars for event-driven backtesting:

```python
from src.replay.bar_iterator import BarIterator

for bar_index, bar_row in BarIterator(bars):
    # bar_row is a pd.Series with time, open, high, low, close, volume
    engine.process_bar(bar_row, features_row, bar_index)
```

**Guarantees:**
1. **Timestamp canonicalization**: All timestamps are `datetime64[ns, UTC]`
2. **Duplicate removal**: Duplicate timestamps are removed (keeps first)
3. **Sorted order**: Bars are sorted by timestamp (ascending)

**Example:**
```python
bars = pd.DataFrame({
    "time": ["2024-01-15 09:00:00", "2024-01-15 09:05:00", "2024-01-15 09:00:00"],  # Duplicate
    "open": [2045.0, 2046.0, 2045.5],
    "close": [2046.0, 2047.0, 2046.5],
})

for idx, bar in BarIterator(bars):
    print(idx, bar["time"])

# Output:
# 0 2024-01-15 09:00:00+00:00
# 1 2024-01-15 09:05:00+00:00
# (duplicate at 09:00:00 was dropped)
```

## validate_bars()

Fail-fast integrity checks before backtesting:

```python
from src.replay.validation import validate_bars

validate_bars(bars, sanity_threshold=0.1)  # Raises ValueError if checks fail
```

**Checks performed:**

### 1. Timestamp Checks

- Timestamps are `pd.Timestamp` (not string)
- Timestamps are timezone-aware (UTC)
- Timestamps are monotonically increasing
- No duplicate timestamps

**Example error:**
```
ValueError: Duplicate timestamps found: ['2024-01-15 09:00:00']
```

### 2. OHLC Integrity

- No NaNs in `open`, `high`, `low`, `close`
- `high >= max(open, close)` (high is highest price)
- `low <= min(open, close)` (low is lowest price)

**Example error:**
```
ValueError: Bar 1234 has high < close (high=2045.0, close=2046.0)
```

### 3. Spread Sanity

- `(high - low) / close < sanity_threshold` (default 0.1 = 10%)

**Purpose:** Detect data errors (e.g., flash crashes, corrupted bars).

**Example error:**
```
ValueError: Bar 5678 has extreme spread: (high-low)/close = 0.15 > 0.10
```

**Override threshold:**
```python
validate_bars(bars, sanity_threshold=0.2)  # Allow 20% intrabar range
```

## Integration with runner.py

```python
# Stage 4: Validation & Preparation

from src.replay.validation import validate_bars
from src.replay.bar_iterator import BarIterator

# Validate data
validate_bars(bars, sanity_threshold=0.1)

# Iterate bars
for bar_index, bar_row in BarIterator(bars):
    features_row = features.loc[bar_row["time"]]
    engine.process_bar(bar_row, features_row, bar_index)
```

## Key Files

| File | Description |
|------|-------------|
| `bar_iterator.py` | BarIterator class (timestamp canonicalization) |
| `validation.py` | validate_bars() (integrity checks) |

## See Also

- [Backtest Engine](../engine/README.md) — How BarIterator integrates with replay loop
- [Data Management](../../data/README.md) — MT5 data snapshot structure
