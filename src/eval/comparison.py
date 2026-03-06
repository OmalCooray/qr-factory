"""Cross-run comparison with ranking and metric-disagreement detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ._formatting import fmt as _fmt
from .aggregator import evaluate_run

log = logging.getLogger(__name__)


def _load_evaluation(run_dir: Path) -> dict:
    """Load evaluation.json, running evaluate_run() if missing."""
    eval_path = run_dir / "evaluation.json"
    if eval_path.exists():
        return json.loads(eval_path.read_text(encoding="utf-8"))
    log.info("evaluation.json missing for %s — running evaluate_run()", run_dir.name)
    result = evaluate_run(run_dir)
    return json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))


def _rank_column(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Rank a series (1 = best). NaN values get worst rank."""
    return series.rank(ascending=ascending, na_option="bottom").astype(int)


def compare_runs(run_dirs: list[Path], out_dir: Path) -> Path:
    """Compare N runs: produce comparison.csv, ranking.csv, comparison.md.

    Parameters
    ----------
    run_dirs : list[Path]
        Paths to run directories.
    out_dir : Path
        Output directory for comparison artifacts.

    Returns
    -------
    Path
        Path to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    evals: list[dict] = []
    for rd in run_dirs:
        ev = _load_evaluation(Path(rd))
        evals.append(ev)

    # ── comparison.csv — all metrics, all runs ──
    rows = []
    for ev in evals:
        row: dict = {"run_id": ev["run_id"], "strategy": ev["strategy_type"]}
        for k, v in ev.get("prediction_metrics", {}).items():
            row[f"pred_{k}"] = v
        for k, v in ev.get("trading_metrics", {}).items():
            row[f"trade_{k}"] = v
        rows.append(row)

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_dir / "comparison.csv", index=False)
    log.info("Wrote comparison.csv (%d runs)", len(comp_df))

    # ── ranking.csv — key metrics + rank columns + disagreement ──
    rank_df = pd.DataFrame({"run_id": comp_df["run_id"], "strategy": comp_df["strategy"]})

    # Safely pull columns that may not exist (non-ML runs)
    for col in ["pred_brier", "pred_roc_auc", "pred_log_loss",
                 "trade_sharpe_per_bar", "trade_total_net_pnl",
                 "trade_profit_factor", "trade_max_dd_pct",
                 "trade_total_cost_pnl", "trade_win_rate", "trade_num_trades"]:
        rank_df[col] = comp_df[col] if col in comp_df.columns else np.nan

    # Rank columns (only if column has any non-NaN values)
    rank_specs = [
        ("pred_brier", True, "rank_brier"),           # lower is better
        ("pred_roc_auc", False, "rank_roc_auc"),      # higher is better
        ("trade_sharpe_per_bar", False, "rank_sharpe"),
        ("trade_total_net_pnl", False, "rank_net_pnl"),
        ("trade_profit_factor", False, "rank_profit_factor"),
    ]
    for src_col, ascending, rank_col in rank_specs:
        if rank_df[src_col].notna().any():
            rank_df[rank_col] = _rank_column(rank_df[src_col], ascending=ascending)
        else:
            rank_df[rank_col] = np.nan

    # Disagreement: |rank_roc_auc - rank_sharpe|
    if rank_df["rank_roc_auc"].notna().any() and rank_df["rank_sharpe"].notna().any():
        rank_df["disagreement_roc_vs_sharpe"] = (
            (rank_df["rank_roc_auc"] - rank_df["rank_sharpe"]).abs()
        )
    else:
        rank_df["disagreement_roc_vs_sharpe"] = 0

    rank_df.to_csv(out_dir / "ranking.csv", index=False)
    log.info("Wrote ranking.csv (%d runs)", len(rank_df))

    # ── comparison.md ──
    md = _build_comparison_md(comp_df, rank_df, evals)
    (out_dir / "comparison.md").write_text(md, encoding="utf-8")
    log.info("Wrote comparison.md → %s", out_dir / "comparison.md")

    return out_dir


def _df_to_md_table(df: pd.DataFrame, float_fmt: int = 4) -> str:
    """Convert a DataFrame to a markdown table."""
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["------:" for _ in headers]) + "|",
    ]
    for _, row in df.iterrows():
        cells = []
        for h in headers:
            v = row[h]
            if isinstance(v, float) and not np.isnan(v):
                cells.append(f"{v:.{float_fmt}f}")
            elif isinstance(v, float) and np.isnan(v):
                cells.append("N/A")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _build_comparison_md(
    comp_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    evals: list[dict],
) -> str:
    lines: list[str] = []
    lines.append(f"# Run Comparison ({len(evals)} runs)")
    lines.append("")

    # Aggregate metrics table
    lines.append("## Aggregate Metrics")
    lines.append("")
    trade_cols = [c for c in comp_df.columns if c.startswith("trade_")]
    display_cols = ["run_id", "strategy"] + trade_cols
    display = comp_df[[c for c in display_cols if c in comp_df.columns]]
    lines.append(_df_to_md_table(display))
    lines.append("")

    # Prediction metrics (if any run has them)
    pred_cols = [c for c in comp_df.columns if c.startswith("pred_")]
    if pred_cols:
        lines.append("## Prediction Metrics")
        lines.append("")
        pred_display = comp_df[["run_id", "strategy"] + pred_cols]
        lines.append(_df_to_md_table(pred_display))
        lines.append("")

    # Rankings
    lines.append("## Rankings")
    lines.append("")
    rank_cols = [c for c in rank_df.columns if c.startswith("rank_") or c == "disagreement_roc_vs_sharpe"]
    rank_display = rank_df[["run_id", "strategy"] + rank_cols]
    lines.append(_df_to_md_table(rank_display, float_fmt=0))
    lines.append("")

    # Disagreement analysis
    lines.append("## Metric Disagreement Analysis")
    lines.append("")
    has_disagreement = (rank_df["disagreement_roc_vs_sharpe"] > 0).any()
    if has_disagreement:
        for _, row in rank_df.iterrows():
            d = row.get("disagreement_roc_vs_sharpe", 0)
            if d > 0:
                run_id = row["run_id"]
                r_roc = row.get("rank_roc_auc", "?")
                r_sharpe = row.get("rank_sharpe", "?")
                if isinstance(r_roc, float) and not np.isnan(r_roc):
                    r_roc = int(r_roc)
                if isinstance(r_sharpe, float) and not np.isnan(r_sharpe):
                    r_sharpe = int(r_sharpe)

                direction = "prediction rank better" if r_roc < r_sharpe else "trading rank better"
                lines.append(
                    f"- **{run_id}**: disagreement={int(d)} "
                    f"(roc_auc rank={r_roc}, sharpe rank={r_sharpe} — {direction})"
                )

                # Heuristic causes
                ev = next((e for e in evals if e["run_id"] == run_id), None)
                if ev:
                    t = ev.get("trading_metrics", {})
                    net = t.get("total_net_pnl", 0)
                    cost = t.get("total_cost_pnl", 0)
                    gross = t.get("total_gross_pnl", 0)
                    if abs(gross) > 0 and abs(cost) > 0.3 * abs(gross):
                        lines.append(f"  - Likely cause: **cost dominance** (cost={_fmt(cost, 2)}, net={_fmt(net, 2)})")
                    elif r_roc < r_sharpe:
                        lines.append("  - Possible cause: **policy mismatch** — good signal not converted to profit")
                    else:
                        lines.append("  - Possible cause: **regime sensitivity** — trading edge from market structure, not signal")
        lines.append("")
    else:
        lines.append("No ranking disagreement detected between prediction quality and trading performance.")
        lines.append("")
        lines.append("To create an ablation pair, try changing the `p_enter` threshold in strategy config and re-running.")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by `src.eval` — {len(evals)} runs compared*")
    lines.append("")

    return "\n".join(lines)
