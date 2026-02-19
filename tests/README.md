# Tests

This directory contains automated tests for the `qr-factory` backtesting engine.

## Included Tests

### Unit Tests
- **[test_validate_bars.py](file:///c:/quant/qr-factory/tests/test_validate_bars.py)**: Verifies the data integrity checks for OHLC bars (e.g., missing columns, null timestamps, duplicate timestamps, non-monotonic data, and invalid spreads).

### Integration Tests
- **[test_replay_determinism.py](file:///c:/quant/qr-factory/tests/test_replay_determinism.py)**: Ensures the engine is deterministic by running identical backtests and verifying that output artifacts (equity curves, metrics, and data snapshots) match exactly across runs.

## Running Tests

Tests are managed via `pytest` and should be run through `uv`.

### Run All Tests
```bash
uv run pytest
```

### Run a Specific Test File
```bash
uv run pytest tests/test_replay_determinism.py
```

### Run with Verbose Output
```bash
uv run pytest -v
```
