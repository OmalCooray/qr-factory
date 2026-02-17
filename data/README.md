# Data Management System

This directory manages the acquisition, versioning, and storage of market data for the Quantitative Research Factory.

## Directory Structure

- `loader/`: A Python package containing the logic for downloading data from MetaTrader 5 (MT5). It is designed with SOLID principles and supports various data sources and writers.
- `DATA_VERSION.md`: A human-readable log of all data exports, tracking the symbol, timeframe, broker, and date range for each snapshot.
- `<YYYYMMDD_HHMMSS>/`: Snapshot directories created by the loader. Each snapshot represents a point-in-time export of market data.
  - `XAUUSD_M5_<YEAR>.csv`: Yearly CSV files containing price bars (Open, High, Low, Close, Volume, Spread).
  - `META.json`: Structured metadata for the snapshot, including row counts and download timestamps.
- `__init__.py`: Makes the `data` directory an importable Python package.

## How to Use the Loader

The data loader is managed via `uv`. You can run it using the following commands from the project root:

### 1. Default Download (XAUUSD M5)
```bash
uv run qr-loader
```

### 2. Custom Symbol and Date Range
```bash
uv run qr-loader --symbol EURUSD --start 2020-01-01 --end 2023-12-31
```

### 3. CLI Help
```bash
uv run qr-loader --help
```

## Data Format

Individual bar files are stored as CSVs with the following columns:
- `time`: UTC timestamp
- `open`, `high`, `low`, `close`: Price values
- `tick_volume`: Number of price changes
- `spread`: Bid/Ask spread
- `real_volume`: Traded volume (if available)

## Automation and Reliability

- **Atomic Writes**: The loader uses a "write-to-temp then rename" strategy to ensure snapshots are never corrupt even if the process is interrupted.
- **Protocol-Based Design**: The system is extensible; new data sources (e.g., REST APIs) or writers (e.g., Parquet, SQL) can be added by implementing the `DataSource` and `DataWriter` protocols.
