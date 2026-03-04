"""Post-hoc diagnostic analysis of a walk-forward run.

Reads trades.csv + fold_metrics.json from a completed run and produces:
  1. Turnover report  — avg gross/cost/net per trade, trades per 1000 bars
  2. Regime analysis   — per-fold trend strength vs PnL
  3. Cost sweep        — re-run with spread/slippage multipliers to find break-even

Usage:
    uv run python scripts/diagnose_walk_forward.py <run_dir> [--sweep]

The --sweep flag triggers H3 (cost sweep), which re-runs the backtest 5 times
and takes ~5 minutes.  Without it, only H1+H2 are computed from existing artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── H1: Turnover Report ─────────────────────────────────────────────────────


def h1_turnover(run_dir: Path, fold_metrics: dict) -> dict:
    """Per-fold and aggregate turnover analysis."""
    trades = pd.read_csv(run_dir / "trades.csv")
    if trades.empty:
        log.warning("No trades found.")
        return {}

    cost_per_trade = trades["cost_pnl"].mean()
    gross_per_trade = trades["gross_pnl"].mean()
    net_per_trade = trades["pnl"].mean()

    folds = fold_metrics["folds"]
    fold_rows = []
    for f in folds:
        n = f["n_trades"]
        bars = f["test_bars"]
        fold_rows.append({
            "fold": f["fold_id"],
            "trades": n,
            "trades_per_1k_bars": round(n / bars * 1000, 1),
            "avg_gross": round(f["gross_pnl"] / max(1, n), 4),
            "avg_cost": round(cost_per_trade, 4),
            "avg_net": round(f["net_pnl"] / max(1, n), 4),
            "gross_pnl": round(f["gross_pnl"], 2),
            "net_pnl": round(f["net_pnl"], 2),
        })

    report = {
        "aggregate": {
            "total_trades": len(trades),
            "avg_gross_per_trade": round(gross_per_trade, 4),
            "avg_cost_per_trade": round(cost_per_trade, 4),
            "avg_net_per_trade": round(net_per_trade, 4),
            "edge_vs_cost_ratio": round(gross_per_trade / cost_per_trade, 3)
            if cost_per_trade > 0
            else float("inf"),
            "verdict": (
                "EDGE < COST: strategy cannot survive without policy change"
                if gross_per_trade < cost_per_trade
                else "EDGE >= COST: marginal viability"
            ),
        },
        "per_fold": fold_rows,
    }

    # Print
    log.info("=" * 70)
    log.info("H1 — TURNOVER REPORT")
    log.info("=" * 70)
    agg = report["aggregate"]
    log.info("  Avg gross/trade: %.4f", agg["avg_gross_per_trade"])
    log.info("  Avg cost/trade:  %.4f", agg["avg_cost_per_trade"])
    log.info("  Avg net/trade:   %.4f", agg["avg_net_per_trade"])
    log.info("  Edge/cost ratio: %.3f", agg["edge_vs_cost_ratio"])
    log.info("  Verdict: %s", agg["verdict"])
    log.info("")
    log.info("  %-5s %6s %10s %10s %10s %10s %10s", "Fold", "Trades", "Tr/1kBar",
             "AvgGross", "AvgCost", "AvgNet", "NetPnL")
    for r in fold_rows:
        log.info("  F%-4d %6d %10.1f %10.4f %10.4f %10.4f %10.2f",
                 r["fold"], r["trades"], r["trades_per_1k_bars"],
                 r["avg_gross"], r["avg_cost"], r["avg_net"], r["net_pnl"])

    return report


# ── H2: Regime Stratification ────────────────────────────────────────────────


def h2_regime(run_dir: Path, fold_metrics: dict, config: dict) -> dict:
    """Classify each fold by trend strength and correlate with PnL."""
    snapshot_dir = config["snapshot_dir"]
    if not Path(snapshot_dir).is_absolute():
        snapshot_dir = REPO_ROOT / snapshot_dir

    csv_files = sorted(Path(snapshot_dir).glob("*.csv"))
    frames = [pd.read_csv(f) for f in csv_files]
    df = pd.concat(frames, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    # Compute regime proxies on full dataset
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    # EMA 50 / 200 (matching strategy)
    ema_fast = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_slow = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values

    # ATR 14
    tr = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values

    # Trend strength = |EMA_fast - EMA_slow| / ATR
    with np.errstate(divide="ignore", invalid="ignore"):
        trend_strength = np.abs(ema_fast - ema_slow) / atr
    # Replace inf with NaN so they don't pollute aggregates
    trend_strength[~np.isfinite(trend_strength)] = np.nan

    # Global median for regime threshold
    global_median_ts = float(np.nanmedian(trend_strength))

    # Per-fold regime stats
    folds = fold_metrics["folds"]
    fold_regime = []
    for f in folds:
        s, e = f["test_start"], f["test_end"] + 1
        ts_slice = trend_strength[s:e]
        atr_slice = atr[s:e]
        valid_ts = ts_slice[np.isfinite(ts_slice)]
        valid_atr = atr_slice[np.isfinite(atr_slice)]

        fold_regime.append({
            "fold": f["fold_id"],
            "net_pnl": round(f["net_pnl"], 2),
            "gross_pnl": round(f["gross_pnl"], 2),
            "n_trades": f["n_trades"],
            "avg_trend_strength": round(float(np.mean(valid_ts)), 3) if len(valid_ts) > 0 else None,
            "median_trend_strength": round(float(np.median(valid_ts)), 3) if len(valid_ts) > 0 else None,
            "avg_atr": round(float(np.mean(valid_atr)), 2) if len(valid_atr) > 0 else None,
            "regime_label": (
                "trend" if len(valid_ts) > 0 and float(np.median(valid_ts)) > global_median_ts
                else "chop"
            ),
        })

    # Aggregate by regime label
    trend_folds = [f for f in fold_regime if f["regime_label"] == "trend"]
    chop_folds = [f for f in fold_regime if f["regime_label"] == "chop"]

    report = {
        "per_fold": fold_regime,
        "trend_summary": {
            "n_folds": len(trend_folds),
            "total_net_pnl": round(sum(f["net_pnl"] for f in trend_folds), 2),
            "total_gross_pnl": round(sum(f["gross_pnl"] for f in trend_folds), 2),
            "avg_trend_strength": round(
                np.mean([f["avg_trend_strength"] for f in trend_folds if f["avg_trend_strength"]]), 3
            ) if trend_folds else 0,
        },
        "chop_summary": {
            "n_folds": len(chop_folds),
            "total_net_pnl": round(sum(f["net_pnl"] for f in chop_folds), 2),
            "total_gross_pnl": round(sum(f["gross_pnl"] for f in chop_folds), 2),
            "avg_trend_strength": round(
                np.mean([f["avg_trend_strength"] for f in chop_folds if f["avg_trend_strength"]]), 3
            ) if chop_folds else 0,
        },
    }

    # Print
    log.info("=" * 70)
    log.info("H2 — REGIME STRATIFICATION (|EMA50-EMA200|/ATR)")
    log.info("=" * 70)
    log.info("  %-5s %8s %10s %10s %10s %8s", "Fold", "Regime", "TrendStr", "AvgATR",
             "GrossPnL", "NetPnL")
    for r in fold_regime:
        log.info("  F%-4d %8s %10.3f %10.2f %10.2f %8.2f",
                 r["fold"], r["regime_label"],
                 r["avg_trend_strength"] or 0, r["avg_atr"] or 0,
                 r["gross_pnl"], r["net_pnl"])
    log.info("")
    ts = report["trend_summary"]
    cs = report["chop_summary"]
    log.info("  TREND folds (%d): gross=%+.2f  net=%+.2f  avg_strength=%.3f",
             ts["n_folds"], ts["total_gross_pnl"], ts["total_net_pnl"], ts["avg_trend_strength"])
    log.info("  CHOP  folds (%d): gross=%+.2f  net=%+.2f  avg_strength=%.3f",
             cs["n_folds"], cs["total_gross_pnl"], cs["total_net_pnl"], cs["avg_trend_strength"])

    # Scatter: trend_strength vs net_pnl
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in fold_regime:
        color = "#2ecc71" if r["regime_label"] == "trend" else "#e74c3c"
        ax.scatter(r["avg_trend_strength"], r["net_pnl"], c=color, s=80, zorder=3)
        ax.annotate(f"F{r['fold']}", (r["avg_trend_strength"], r["net_pnl"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xlabel("Avg Trend Strength (|EMA50-EMA200|/ATR)")
    ax.set_ylabel("Net PnL")
    ax.set_title("H2: Regime vs Fold Performance (green=trend, red=chop)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = run_dir / "plots" / "h2_regime_vs_pnl.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    log.info("  Saved %s", plot_path)

    return report


# ── H3: Cost Sweep ───────────────────────────────────────────────────────────


def h3_cost_sweep(run_dir: Path, config: dict) -> dict:
    """Re-run backtest at 5 cost multipliers, track net PnL."""
    from src.engine.runner import run_backtest
    import tempfile

    multipliers = [0.0, 0.5, 1.0, 1.5, 2.0]
    base_spread = config.get("execution", {}).get("spread_points", 0)
    base_slip = config.get("execution", {}).get("slippage_points", 0)

    results = []

    for mult in multipliers:
        log.info("  Cost sweep: multiplier=%.1f (spread=%.0f, slip=%.0f)",
                 mult, base_spread * mult, base_slip * mult)

        sweep_cfg = dict(config)
        sweep_cfg["execution"] = {
            "spread_points": base_spread * mult,
            "slippage_points": base_slip * mult,
            "point_value": config.get("execution", {}).get("point_value", 1.0),
        }
        # Remove validation to run single-pass (faster, same total PnL)
        sweep_cfg.pop("validation", None)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False,
                                         dir=REPO_ROOT / "configs") as f:
            yaml.dump(sweep_cfg, f)
            tmp_config = f.name

        try:
            sweep_run_id = run_backtest(tmp_config)
            sweep_dir = REPO_ROOT / config["output_dir"] / sweep_run_id
            sweep_metrics = json.loads((sweep_dir / "metrics.json").read_text())

            results.append({
                "multiplier": mult,
                "spread_points": base_spread * mult,
                "slippage_points": base_slip * mult,
                "total_gross_pnl": sweep_metrics["total_gross_pnl"],
                "total_net_pnl": sweep_metrics["total_net_pnl"],
                "total_cost_pnl": sweep_metrics["total_cost_pnl"],
                "n_trades": sweep_metrics["n_trades"],
            })
        finally:
            Path(tmp_config).unlink(missing_ok=True)

    # Find break-even multiplier by linear interpolation
    break_even = None
    for i in range(len(results) - 1):
        pnl_a = results[i]["total_net_pnl"]
        pnl_b = results[i + 1]["total_net_pnl"]
        if pnl_a >= 0 and pnl_b < 0:
            frac = pnl_a / (pnl_a - pnl_b)
            break_even = results[i]["multiplier"] + frac * (
                results[i + 1]["multiplier"] - results[i]["multiplier"]
            )
            break

    report = {
        "sweep": results,
        "break_even_multiplier": round(break_even, 3) if break_even is not None else None,
        "break_even_spread_points": round(break_even * base_spread, 1) if break_even is not None else None,
    }

    # Print
    log.info("=" * 70)
    log.info("H3 — COST SWEEP (base: spread=%d, slip=%d)", base_spread, base_slip)
    log.info("=" * 70)
    log.info("  %-6s %8s %8s %12s %12s %12s", "Mult", "Spread", "Slip",
             "GrossPnL", "NetPnL", "CostPnL")
    for r in results:
        log.info("  %-6.1f %8.0f %8.0f %12.2f %12.2f %12.2f",
                 r["multiplier"], r["spread_points"], r["slippage_points"],
                 r["total_gross_pnl"], r["total_net_pnl"], r["total_cost_pnl"])
    if break_even is not None:
        log.info("  Break-even multiplier: %.3f (spread ~%.1f points)",
                 break_even, break_even * base_spread)
    else:
        log.info("  Break-even: not found (strategy unprofitable even at zero cost, or always profitable)")

    # Plot
    mults = [r["multiplier"] for r in results]
    gross = [r["total_gross_pnl"] for r in results]
    net = [r["total_net_pnl"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(mults, gross, "o-", color="#3498db", label="Gross PnL")
    ax.plot(mults, net, "o-", color="#e74c3c", label="Net PnL")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    if break_even is not None:
        ax.axvline(break_even, color="orange", linewidth=1.5, linestyle="--",
                   label=f"Break-even: {break_even:.2f}x")
    ax.set_xlabel("Cost Multiplier")
    ax.set_ylabel("Total PnL")
    ax.set_title("H3: Cost Sweep — Break-Even Friction Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = run_dir / "plots" / "h3_cost_sweep.png"
    fig.savefig(plot_path, dpi=100)
    plt.close(fig)
    log.info("  Saved %s", plot_path)

    return report


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Diagnose a walk-forward run")
    parser.add_argument("run_dir", help="Path to the run directory")
    parser.add_argument("--sweep", action="store_true",
                        help="Run H3 cost sweep (re-runs backtest 5 times)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    # Load artifacts
    fm_path = run_dir / "fold_metrics.json"
    if not fm_path.exists():
        log.error("fold_metrics.json not found in %s", run_dir)
        sys.exit(1)

    fold_metrics = json.loads(fm_path.read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())

    # Run diagnostics
    h1 = h1_turnover(run_dir, fold_metrics)
    h2 = h2_regime(run_dir, fold_metrics, config)

    h3 = {}
    if args.sweep:
        h3 = h3_cost_sweep(run_dir, config)

    # Save combined report
    report = {"h1_turnover": h1, "h2_regime": h2}
    if h3:
        report["h3_cost_sweep"] = h3
    report_path = run_dir / "diagnosis.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s", report_path)


if __name__ == "__main__":
    main()
