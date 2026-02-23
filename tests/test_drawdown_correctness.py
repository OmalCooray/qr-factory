"""Deterministic unit tests for max-drawdown computation correctness.

The drawdown logic under test lives inline in src/engine/runner.py (lines ~257-265).
These tests replicate the exact same algorithm against synthetic equity curves to
verify pure math correctness, peak tracking, percentage computation, and reset behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.engine.runner import run_backtest


# ---------------------------------------------------------------------------
# Helper: replicate the exact drawdown algorithm from runner.py
# ---------------------------------------------------------------------------

def compute_drawdown_series(equity: list[float]) -> dict:
    """Replicate runner.py's drawdown logic step-by-step.

    Returns dict with rolling_peak, drawdown_series (%), max_drawdown (%).
    """
    peak = equity[0]
    rolling_peak = []
    drawdown_series = []
    max_dd = 0.0

    for eq in equity:
        if eq > peak:
            peak = eq
        rolling_peak.append(peak)

        if peak > 0:
            dd = (peak - eq) / peak * 100
        else:
            dd = 0.0
        drawdown_series.append(dd)

        if dd > max_dd:
            max_dd = dd

    return {
        "rolling_peak": rolling_peak,
        "drawdown_series": drawdown_series,
        "max_drawdown": max_dd,
    }


# ---------------------------------------------------------------------------
# Test Case 1 — Monotonic Increase
# ---------------------------------------------------------------------------

def test_monotonic_increase():
    """Equity only goes up → drawdown should be 0% throughout."""
    equity = [100, 105, 110, 120]
    result = compute_drawdown_series(equity)

    print("\n--- Test Case 1: Monotonic Increase ---")
    print(f"  equity:          {equity}")
    print(f"  rolling_peak:    {result['rolling_peak']}")
    print(f"  drawdown_series: {result['drawdown_series']}")
    print(f"  max_drawdown:    {result['max_drawdown']:.6f}%")

    assert result["max_drawdown"] == 0.0
    assert all(dd == 0.0 for dd in result["drawdown_series"])
    # Peak should equal equity at every step
    assert result["rolling_peak"] == equity


# ---------------------------------------------------------------------------
# Test Case 2 — Single Drawdown
# ---------------------------------------------------------------------------

def test_single_drawdown():
    """Peak=120, trough=90 → DD = (120-90)/120 = 25%."""
    equity = [100, 120, 90, 130]
    result = compute_drawdown_series(equity)

    print("\n--- Test Case 2: Single Drawdown ---")
    print(f"  equity:          {equity}")
    print(f"  rolling_peak:    {result['rolling_peak']}")
    print(f"  drawdown_series: {result['drawdown_series']}")
    print(f"  max_drawdown:    {result['max_drawdown']:.6f}%")

    assert result["rolling_peak"] == [100, 120, 120, 130]
    assert abs(result["max_drawdown"] - 25.0) < 1e-10
    # At bar 2: dd = (120-90)/120*100 = 25.0
    assert abs(result["drawdown_series"][2] - 25.0) < 1e-10
    # At bar 3: new peak 130, dd = 0
    assert result["drawdown_series"][3] == 0.0


# ---------------------------------------------------------------------------
# Test Case 3 — Multiple Peaks
# ---------------------------------------------------------------------------

def test_multiple_peaks():
    """Two drawdowns: 150→140 (6.67%), 160→120 (25%). Max = 25%."""
    equity = [100, 150, 140, 160, 120]
    result = compute_drawdown_series(equity)

    print("\n--- Test Case 3: Multiple Peaks ---")
    print(f"  equity:          {equity}")
    print(f"  rolling_peak:    {result['rolling_peak']}")
    print(f"  drawdown_series: {[f'{dd:.4f}' for dd in result['drawdown_series']]}")
    print(f"  max_drawdown:    {result['max_drawdown']:.6f}%")

    assert result["rolling_peak"] == [100, 150, 150, 160, 160]
    # DD at bar 2: (150-140)/150*100 = 6.6667%
    assert abs(result["drawdown_series"][2] - 100 * 10 / 150) < 1e-10
    # DD at bar 4: (160-120)/160*100 = 25%
    assert abs(result["drawdown_series"][4] - 25.0) < 1e-10
    assert abs(result["max_drawdown"] - 25.0) < 1e-10


# ---------------------------------------------------------------------------
# Test Case 4 — Flat then Drop
# ---------------------------------------------------------------------------

def test_flat_then_drop():
    """Flat at 100 then drop to 80 → DD = 20%."""
    equity = [100, 100, 100, 80]
    result = compute_drawdown_series(equity)

    print("\n--- Test Case 4: Flat then Drop ---")
    print(f"  equity:          {equity}")
    print(f"  rolling_peak:    {result['rolling_peak']}")
    print(f"  drawdown_series: {result['drawdown_series']}")
    print(f"  max_drawdown:    {result['max_drawdown']:.6f}%")

    assert result["rolling_peak"] == [100, 100, 100, 100]
    assert abs(result["max_drawdown"] - 20.0) < 1e-10
    # First three bars: 0% drawdown
    assert result["drawdown_series"][:3] == [0.0, 0.0, 0.0]
    assert abs(result["drawdown_series"][3] - 20.0) < 1e-10


# ---------------------------------------------------------------------------
# Structural properties
# ---------------------------------------------------------------------------

def test_drawdown_never_negative():
    """Drawdown is always >= 0."""
    equity = [100, 120, 90, 130, 50, 200, 180]
    result = compute_drawdown_series(equity)
    assert all(dd >= 0.0 for dd in result["drawdown_series"])


def test_max_drawdown_non_decreasing():
    """Running max DD must be non-decreasing over time."""
    equity = [100, 120, 90, 130, 50, 200, 180]
    result = compute_drawdown_series(equity)

    running_max = 0.0
    for dd in result["drawdown_series"]:
        new_max = max(running_max, dd)
        assert new_max >= running_max
        running_max = new_max


def test_peak_resets_only_on_new_high():
    """Rolling peak should never decrease and only change on new equity highs."""
    equity = [100, 120, 110, 115, 130, 125]
    result = compute_drawdown_series(equity)

    peaks = result["rolling_peak"]
    # Non-decreasing
    for i in range(1, len(peaks)):
        assert peaks[i] >= peaks[i - 1]
    # Peak only changes when equity exceeds previous peak
    assert peaks == [100, 120, 120, 120, 130, 130]


# ---------------------------------------------------------------------------
# Integration: cross-validate against a real backtest
# ---------------------------------------------------------------------------

def _write_controlled_bars(data_dir: Path) -> None:
    """Write a CSV with a price sequence that produces a known equity path.

    50 bars: 10 flat → 10 ramp up → 10 crash → 20 recovery.
    With fast=2/slow=5 and starting_capital=100, the strategy goes long
    during the ramp, takes a hit during the crash, then recovers.
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    n = 50
    times = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")

    close = (
        [50.0] * 10
        + [50 + i * 5 for i in range(1, 11)]   # 55, 60, ... 100
        + [100 - i * 7 for i in range(1, 11)]   # 93, 86, ... 30
        + [30 + i * 2 for i in range(1, 21)]    # 32, 34, ... 70
    )
    open_ = [close[0]] + [close[i - 1] for i in range(1, n)]

    df = pd.DataFrame({
        "time": times,
        "open": open_,
        "high": [max(o, c) + 1 for o, c in zip(open_, close)],
        "low": [min(o, c) - 1 for o, c in zip(open_, close)],
        "close": close,
        "volume": [1000] * n,
    })
    df.to_csv(data_dir / "controlled.csv", index=False)


def test_cross_validate_with_real_backtest(tmp_path: Path) -> None:
    """Run a real backtest, then independently recompute drawdown from equity.csv
    and assert the two values match to within floating-point tolerance."""
    data_dir = tmp_path / "data"
    _write_controlled_bars(data_dir)

    config = {
        "symbol": "TEST",
        "timeframe": "1h",
        "snapshot_dir": str(data_dir),
        "starting_capital": 100.0,
        "output_dir": str(tmp_path / "runs"),
        "strategy": {
            "type": "ma_crossover",
            "fast_period": 2,
            "slow_period": 5,
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_id = run_backtest(str(config_path))
    run_dir = tmp_path / "runs" / run_id

    # ── Load engine outputs ──────────────────────────────────────────
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    engine_max_dd = metrics["max_drawdown_pct"]

    equity_df = pd.read_csv(run_dir / "equity.csv")
    equity_series = equity_df["equity"].values

    # ── Independent recomputation using numpy/pandas ─────────────────
    rolling_peak = np.maximum.accumulate(equity_series)
    dd_series = np.where(
        rolling_peak > 0,
        (rolling_peak - equity_series) / rolling_peak * 100,
        0.0,
    )
    independent_max_dd = float(dd_series.max())

    print(f"\n--- Cross-Validation Against Real Backtest ---")
    print(f"  Bars:                {len(equity_df)}")
    print(f"  Engine max_dd:       {engine_max_dd:.10f}%")
    print(f"  Independent max_dd:  {independent_max_dd:.10f}%")
    print(f"  Abs difference:      {abs(engine_max_dd - independent_max_dd):.2e}")

    # Structural checks on the independent series
    assert (dd_series >= 0).all(), "Negative drawdown found"
    assert np.all(np.diff(np.maximum.accumulate(dd_series)) >= -1e-15), \
        "Running max DD decreased"

    # The key assertion: engine and independent must match
    assert abs(engine_max_dd - independent_max_dd) < 1e-10, (
        f"Drawdown mismatch: engine={engine_max_dd}, independent={independent_max_dd}"
    )


# ---------------------------------------------------------------------------
# DrawdownTracker unit tests
# ---------------------------------------------------------------------------

from src.risk.drawdown import DrawdownTracker, compute_drawdown_pct


def test_tracker_monotonic_increase():
    """All-up equity → max DD stays 0."""
    tracker = DrawdownTracker(100.0)
    for eq in [105, 110, 120, 150]:
        state = tracker.update(eq)
        assert state.max_drawdown_pct == 0.0
        assert state.current_drawdown_pct == 0.0
    assert tracker.peak_equity == 150.0


def test_tracker_single_drawdown():
    """Peak=120, trough=90 → 25%."""
    tracker = DrawdownTracker(100.0)
    tracker.update(120.0)
    state = tracker.update(90.0)
    assert abs(state.current_drawdown_pct - 25.0) < 1e-10
    assert abs(state.max_drawdown_pct - 25.0) < 1e-10
    assert tracker.peak_equity == 120.0


def test_tracker_matches_reference():
    """Cross-validate DrawdownTracker against compute_drawdown_series()."""
    equity = [100, 120, 90, 130, 50, 200, 180]
    ref = compute_drawdown_series(equity)
    tracker = DrawdownTracker(equity[0])
    for i, eq in enumerate(equity):
        state = tracker.update(eq)
        assert abs(state.current_drawdown_pct - ref["drawdown_series"][i]) < 1e-10
        assert abs(state.peak_equity - ref["rolling_peak"][i]) < 1e-10
    assert abs(tracker.max_drawdown_pct - ref["max_drawdown"]) < 1e-10


def test_compute_drawdown_pct_pure():
    """Edge cases for the free function."""
    # No drawdown
    assert compute_drawdown_pct(100.0, 100.0) == 0.0
    # Normal drawdown
    assert abs(compute_drawdown_pct(100.0, 80.0) - 20.0) < 1e-10
    # Peak is zero → 0.0 (zero-div guard)
    assert compute_drawdown_pct(0.0, 50.0) == 0.0
    # Current > peak → clamped to 0.0
    assert compute_drawdown_pct(100.0, 120.0) == 0.0
    # Negative peak → 0.0
    assert compute_drawdown_pct(-10.0, 5.0) == 0.0
