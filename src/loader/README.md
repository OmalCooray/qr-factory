# MT5 Data Loader

**MetaTrader 5 data pipeline with validation, versioning, and atomic writes.**

## Overview

The `src/loader/` package implements the MT5 data ingestion pipeline. It downloads OHLCV data from MetaTrader 5 terminal and writes it to timestamped snapshot directories with metadata.

**Core workflow:**
```
MT5 Terminal → Loader → Atomic Write → Snapshot Directory
```

**Key features:**
- Atomic file writes (temp + rename pattern)
- Per-year CSV files (e.g., `XAUUSD_M5_2023.csv`, `XAUUSD_M5_2024.csv`)
- Timestamped snapshots with metadata (`META.json`)
- Version log (`data/DATA_VERSION.md`)

## Usage

```bash
uv run loader
```

**Output:**
```
data/20260217_030324/
├── XAUUSD_M5_2023.csv
├── XAUUSD_M5_2024.csv
├── XAUUSD_M5_2025.csv
└── META.json
```

## Snapshot Structure

### CSV Files

Per-year OHLCV files:

```csv
time,open,high,low,close,tick_volume
2023-01-02 00:00:00,1824.50,1825.30,1823.80,1824.90,1234
...
```

**Columns:**
- `time`: Bar timestamp (UTC)
- `open`, `high`, `low`, `close`: OHLC prices
- `tick_volume`: Tick count (MT5 only provides tick volume, not real volume)

**Note:** `tick_volume` is aliased to `volume` by `_load_market_data()` in runner.py.

### META.json

Snapshot metadata:

```json
{
  "symbol": "XAUUSD",
  "timeframe": "M5",
  "created_at": "2026-02-17T03:03:24Z",
  "total_bars": 635421,
  "files": [
    "XAUUSD_M5_2023.csv",
    "XAUUSD_M5_2024.csv",
    "XAUUSD_M5_2025.csv"
  ]
}
```

## Architecture

### Loader Orchestrator

```python
from src.loader._loader import Loader

loader = Loader(
    symbol="XAUUSD",
    timeframe="M5",
    start_date="2023-01-01",
    end_date="2025-12-31",
    output_dir="data",
)

loader.run()
```

**Flow:**
1. Create snapshot directory (`data/<YYYYMMDD_HHMMSS>/`)
2. Download data from MT5 (via `DataSource`)
3. Write per-year CSVs atomically (via `DataWriter`)
4. Write `META.json`
5. Update `data/DATA_VERSION.md` (via `VersionLog`)

### DataSource Protocol

```python
class DataSource(Protocol):
    def fetch_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from data source."""
        ...
```

**Implementation:** `MT5DataSource` (uses MetaTrader5 Python API)

### DataWriter Protocol

```python
class DataWriter(Protocol):
    def write(
        self,
        bars: pd.DataFrame,
        output_path: Path,
    ) -> None:
        """Write bars to file atomically."""
        ...
```

**Implementation:** `CSVDataWriter` (atomic write via temp + rename)

### Atomic Writes

```python
def write(self, bars: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_suffix(".tmp")
    bars.to_csv(temp_path, index=False)
    temp_path.rename(output_path)  # Atomic on POSIX, near-atomic on Windows
```

**Purpose:** Prevent corruption if process is interrupted mid-write.

## Key Files

| File | Description |
|------|-------------|
| `_loader.py` | Loader orchestrator + CLI entry point |
| `data_source.py` | MT5DataSource implementation |
| `data_writer.py` | CSVDataWriter (atomic writes) |
| `version_log.py` | VersionLog (data/DATA_VERSION.md tracking) |

## See Also

- [Data Management](../../data/README.md) — Snapshot directory structure
- [Bar Validation](../replay/README.md) — Integrity checks on loaded data
- [Backtest Engine](../engine/README.md) — How runner.py loads data from snapshots
