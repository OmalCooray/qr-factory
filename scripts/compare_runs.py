"""Compare two walk-forward backtest runs side by side.

Loads metrics.json + fold_metrics.json from each run and produces:
  1. Side-by-side aggregate table (printed + written)
  2. Per-fold comparison table
  3. Structured comparison markdown file

Usage:
    uv run python scripts/compare_runs.py <run_dir_a> <run_dir_b>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_run(run_dir: Path) -> dict:
    """Load metrics.json, fold_metrics.json, and config.yaml from a run."""
    metrics = json.loads((run_dir / "metrics.json").read_text())
    fold_metrics = json.loads((run_dir / "fold_metrics.json").read_text())
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    return {"metrics": metrics, "fold_metrics": fold_metrics, "config": config}


def check_matched(cfg_a: dict, cfg_b: dict) -> list[str]:
    """Verify both runs share identical data, splits, and costs. Return warnings."""
    warnings = []
    for key in ("snapshot_dir", "starting_capital", "symbol", "timeframe"):
        va = cfg_a.get(key)
        vb = cfg_b.get(key)
        if va != vb:
            warnings.append(f"MISMATCH {key}: {va} vs {vb}")

    va = cfg_a.get("validation", {})
    vb = cfg_b.get("validation", {})
    for key in ("type", "mode", "train_size", "test_size", "step", "drop_last"):
        if va.get(key) != vb.get(key):
            warnings.append(f"MISMATCH validation.{key}: {va.get(key)} vs {vb.get(key)}")

    ea = cfg_a.get("execution", {})
    eb = cfg_b.get("execution", {})
    for key in ("spread_points", "slippage_points", "point_value"):
        if ea.get(key) != eb.get(key):
            warnings.append(f"MISMATCH execution.{key}: {ea.get(key)} vs {eb.get(key)}")

    return warnings


def fmt(v, decimals=2) -> str:
    """Format a numeric value for table display."""
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def build_aggregate_table(ma: dict, mb: dict, label_a: str, label_b: str) -> str:
    """Build a markdown table comparing aggregate metrics."""
    rows = [
        ("n_trades", ma.get("n_trades"), mb.get("n_trades"), 0),
        ("total_net_pnl", ma.get("total_net_pnl"), mb.get("total_net_pnl"), 2),
        ("total_gross_pnl", ma.get("total_gross_pnl"), mb.get("total_gross_pnl"), 2),
        ("total_cost_pnl", ma.get("total_cost_pnl"), mb.get("total_cost_pnl"), 2),
        ("win_rate", ma.get("win_rate"), mb.get("win_rate"), 4),
        ("worst_drawdown_pct", ma.get("worst_drawdown_pct"), mb.get("worst_drawdown_pct"), 2),
        ("ending_equity", ma.get("ending_equity"), mb.get("ending_equity"), 2),
    ]

    lines = [
        f"| Metric | {label_a} | {label_b} |",
        "|--------|-------:|-------:|",
    ]
    for name, va, vb, dec in rows:
        lines.append(f"| {name} | {fmt(va, dec)} | {fmt(vb, dec)} |")

    # Fold-level averages from aggregate block in fold_metrics
    return "\n".join(lines)


def build_fold_table(folds_a: list, folds_b: list, label_a: str, label_b: str) -> str:
    """Build a per-fold comparison markdown table."""
    header = (
        f"| Fold | {label_a} net_pnl | {label_b} net_pnl | "
        f"{label_a} trades | {label_b} trades | "
        f"{label_a} win% | {label_b} win% | "
        f"{label_a} max_dd% | {label_b} max_dd% |"
    )
    sep = "|------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|"

    lines = [header, sep]
    n = max(len(folds_a), len(folds_b))
    for i in range(n):
        fa = folds_a[i] if i < len(folds_a) else {}
        fb = folds_b[i] if i < len(folds_b) else {}
        fid = fa.get("fold_id", fb.get("fold_id", i))
        lines.append(
            f"| F{fid} "
            f"| {fmt(fa.get('net_pnl'))} | {fmt(fb.get('net_pnl'))} "
            f"| {fmt(fa.get('n_trades'), 0)} | {fmt(fb.get('n_trades'), 0)} "
            f"| {fmt(fa.get('win_rate', 0) * 100 if fa.get('win_rate') is not None else None, 1)} "
            f"| {fmt(fb.get('win_rate', 0) * 100 if fb.get('win_rate') is not None else None, 1)} "
            f"| {fmt(fa.get('max_drawdown_pct'))} | {fmt(fb.get('max_drawdown_pct'))} |"
        )
    return "\n".join(lines)


def build_ml_metrics_table(agg_a: dict, agg_b: dict, label_a: str, label_b: str) -> str | None:
    """Build ML-specific metrics table if either run has brier/log_loss."""
    has_ml_a = "avg_brier" in agg_a
    has_ml_b = "avg_brier" in agg_b
    if not has_ml_a and not has_ml_b:
        return None

    lines = [
        f"| ML Metric | {label_a} | {label_b} |",
        "|-----------|-------:|-------:|",
        f"| avg_brier | {fmt(agg_a.get('avg_brier'), 4)} | {fmt(agg_b.get('avg_brier'), 4)} |",
        f"| avg_log_loss | {fmt(agg_a.get('avg_log_loss'), 4)} | {fmt(agg_b.get('avg_log_loss'), 4)} |",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare two walk-forward runs")
    parser.add_argument("run_a", help="Path to first run directory")
    parser.add_argument("run_b", help="Path to second run directory")
    args = parser.parse_args()

    dir_a = Path(args.run_a)
    dir_b = Path(args.run_b)
    if not dir_a.is_absolute():
        dir_a = REPO_ROOT / dir_a
    if not dir_b.is_absolute():
        dir_b = REPO_ROOT / dir_b

    for d in (dir_a, dir_b):
        if not (d / "metrics.json").exists():
            log.error("metrics.json not found in %s", d)
            sys.exit(1)
        if not (d / "fold_metrics.json").exists():
            log.error("fold_metrics.json not found in %s", d)
            sys.exit(1)

    run_a = load_run(dir_a)
    run_b = load_run(dir_b)

    # Determine labels
    strat_a = run_a["config"].get("strategy", {}).get("type", "unknown")
    strat_b = run_b["config"].get("strategy", {}).get("type", "unknown")
    id_a = dir_a.name
    id_b = dir_b.name
    label_a = f"{strat_a} ({id_a})"
    label_b = f"{strat_b} ({id_b})"

    # Check match
    warnings = check_matched(run_a["config"], run_b["config"])
    if warnings:
        log.warning("Config mismatches detected:")
        for w in warnings:
            log.warning("  %s", w)
    else:
        log.info("Configs matched: data, splits, and costs are identical.")

    ma = run_a["metrics"]
    mb = run_b["metrics"]
    folds_a = run_a["fold_metrics"]["folds"]
    folds_b = run_b["fold_metrics"]["folds"]
    agg_a = run_a["fold_metrics"].get("aggregate", {})
    agg_b = run_b["fold_metrics"].get("aggregate", {})

    # Build tables
    agg_table = build_aggregate_table(ma, mb, label_a, label_b)
    fold_table = build_fold_table(folds_a, folds_b, label_a, label_b)
    ml_table = build_ml_metrics_table(agg_a, agg_b, label_a, label_b)

    # Print to stdout
    print()
    print("=" * 70)
    print("AGGREGATE COMPARISON")
    print("=" * 70)
    print(agg_table)
    print()
    print("=" * 70)
    print("PER-FOLD COMPARISON")
    print("=" * 70)
    print(fold_table)
    if ml_table:
        print()
        print("=" * 70)
        print("ML-SPECIFIC METRICS")
        print("=" * 70)
        print(ml_table)
    print()

    # Build markdown report
    md_lines = [
        f"# Walk-Forward Comparison: {label_a} vs {label_b}",
        "",
        "## Config Match",
        "",
    ]
    if warnings:
        md_lines.append("**WARNING: Configs do not fully match.**")
        for w in warnings:
            md_lines.append(f"- {w}")
    else:
        md_lines.append("Data, splits, and execution costs are identical.")
    md_lines.append("")

    md_lines += [
        "## Aggregate Metrics",
        "",
        agg_table,
        "",
        "## Per-Fold Comparison",
        "",
        fold_table,
        "",
    ]

    if ml_table:
        md_lines += [
            "## ML-Specific Metrics",
            "",
            ml_table,
            "",
        ]

    # Summary stats
    net_a = ma.get("total_net_pnl", 0)
    net_b = mb.get("total_net_pnl", 0)
    delta = net_a - net_b
    md_lines += [
        "## Summary",
        "",
        f"- **{label_a}** net PnL: {fmt(net_a)}",
        f"- **{label_b}** net PnL: {fmt(net_b)}",
        f"- **Delta (A - B)**: {fmt(delta)}",
        "",
        "## Conclusion",
        "",
        "<!-- Fill in after reviewing the comparison -->",
        "",
        "1. **What changed:** ",
        "2. **Why it matters:** ",
        "3. **What may be misleading:** ",
        "",
        "---",
        f"*Generated by `scripts/compare_runs.py` from runs `{id_a}` and `{id_b}`*",
    ]

    md_content = "\n".join(md_lines) + "\n"

    out_path = REPO_ROOT / "runs" / f"comparison_{id_a}_vs_{id_b}.md"
    out_path.write_text(md_content, encoding="utf-8")
    log.info("Wrote comparison: %s", out_path)


if __name__ == "__main__":
    main()
