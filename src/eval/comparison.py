"""Cross-run comparison with ranking and metric-disagreement detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ._formatting import fmt as _fmt
from .aggregator import evaluate_run

log = logging.getLogger(__name__)


def _check_config_match(run_dirs: list[Path]) -> tuple[bool, list[str]]:
    """Verify all runs share identical controlled fields.

    Compares: snapshot_dir, starting_capital, symbol, timeframe,
    validation.*, execution.*, label.* (when both runs have labels).

    Returns (matched, warnings).
    """
    configs: list[dict] = []
    for rd in run_dirs:
        cfg_path = Path(rd) / "config.yaml"
        if cfg_path.exists():
            configs.append(yaml.safe_load(cfg_path.read_text(encoding="utf-8")))
        else:
            configs.append({})

    if len(configs) < 2:
        return True, []

    warnings: list[str] = []
    base = configs[0]
    for i, cfg in enumerate(configs[1:], start=1):
        run_label = Path(run_dirs[i]).name

        for key in ("snapshot_dir", "starting_capital", "symbol", "timeframe"):
            if base.get(key) != cfg.get(key):
                warnings.append(
                    f"MISMATCH {key}: {base.get(key)} vs {cfg.get(key)} (run {run_label})"
                )

        base_val = base.get("validation", {})
        cfg_val = cfg.get("validation", {})
        for key in ("type", "mode", "train_size", "test_size", "step", "drop_last"):
            if base_val.get(key) != cfg_val.get(key):
                warnings.append(
                    f"MISMATCH validation.{key}: {base_val.get(key)} vs {cfg_val.get(key)} (run {run_label})"
                )

        base_exec = base.get("execution", {})
        cfg_exec = cfg.get("execution", {})
        for key in ("spread_points", "slippage_points", "point_value"):
            if base_exec.get(key) != cfg_exec.get(key):
                warnings.append(
                    f"MISMATCH execution.{key}: {base_exec.get(key)} vs {cfg_exec.get(key)} (run {run_label})"
                )

        # Label fields — only check when both runs have a label block
        base_lbl = base.get("label", {})
        cfg_lbl = cfg.get("label", {})
        if base_lbl and cfg_lbl:
            for key in ("type", "horizon", "price_col", "cost_buffer", "direction"):
                if base_lbl.get(key) != cfg_lbl.get(key):
                    warnings.append(
                        f"MISMATCH label.{key}: {base_lbl.get(key)} vs {cfg_lbl.get(key)} (run {run_label})"
                    )

    return len(warnings) == 0, warnings


def _generate_conclusion(rank_df: pd.DataFrame, evals: list[dict]) -> list[str]:
    """Auto-generate a 3-line conclusion from ranking and evaluation data.

    Returns a list of markdown lines.
    """
    lines: list[str] = []

    # ── Line 1: What changed ──
    has_sharpe_rank = "rank_sharpe" in rank_df.columns and rank_df["rank_sharpe"].notna().any()
    has_roc_rank = "rank_roc_auc" in rank_df.columns and rank_df["rank_roc_auc"].notna().any()

    if has_sharpe_rank:
        best_sharpe_idx = rank_df["rank_sharpe"].idxmin()
        best_sharpe_run = rank_df.loc[best_sharpe_idx, "run_id"]
        best_sharpe_strat = rank_df.loc[best_sharpe_idx, "strategy"]

        if has_roc_rank:
            best_roc_idx = rank_df["rank_roc_auc"].idxmin()
            best_roc_run = rank_df.loc[best_roc_idx, "run_id"]
            if best_sharpe_run != best_roc_run:
                roc_rank_of_best = int(rank_df.loc[best_roc_idx, "rank_roc_auc"])
                lines.append(
                    f"1. **What changed:** {best_sharpe_strat} ({best_sharpe_run}) ranks #1 on Sharpe; "
                    f"prediction ranks disagree (roc_auc rank=1 for {best_roc_run})."
                )
            else:
                lines.append(
                    f"1. **What changed:** {best_sharpe_strat} ({best_sharpe_run}) ranks #1 on both "
                    f"Sharpe and roc_auc — prediction and trading metrics agree."
                )
        else:
            lines.append(
                f"1. **What changed:** {best_sharpe_strat} ({best_sharpe_run}) ranks #1 on Sharpe "
                f"(no prediction metrics available for comparison)."
            )
    else:
        lines.append("1. **What changed:** Insufficient ranking data to determine best run.")

    # ── Line 2: Why it matters ──
    has_disagreement = (
        "disagreement_roc_vs_sharpe" in rank_df.columns
        and (rank_df["disagreement_roc_vs_sharpe"] > 0).any()
    )

    if has_disagreement:
        # Find the run with max disagreement
        max_d_idx = rank_df["disagreement_roc_vs_sharpe"].idxmax()
        d_run = rank_df.loc[max_d_idx, "run_id"]
        r_roc = int(rank_df.loc[max_d_idx, "rank_roc_auc"]) if rank_df.loc[max_d_idx, "rank_roc_auc"] == rank_df.loc[max_d_idx, "rank_roc_auc"] else "?"
        r_sharpe = int(rank_df.loc[max_d_idx, "rank_sharpe"]) if rank_df.loc[max_d_idx, "rank_sharpe"] == rank_df.loc[max_d_idx, "rank_sharpe"] else "?"
        if r_roc < r_sharpe:
            lines.append(
                f"2. **Why it matters:** {d_run} has better predictions (roc_auc rank={r_roc}) "
                f"but worse trading (Sharpe rank={r_sharpe}) — policy or costs may not convert signal to profit."
            )
        else:
            lines.append(
                f"2. **Why it matters:** {d_run} has better trading (Sharpe rank={r_sharpe}) "
                f"but worse predictions (roc_auc rank={r_roc}) — trading edge may come from market structure, not signal."
            )
    else:
        # Check cost dominance on any run
        cost_flag = False
        for ev in evals:
            t = ev.get("trading_metrics", {})
            cost = abs(t.get("total_cost_pnl", 0))
            gross = abs(t.get("total_gross_pnl", 0))
            if gross > 0 and cost > 0.3 * gross:
                lines.append(
                    f"2. **Why it matters:** Cost drag is significant for {ev['run_id']} "
                    f"(cost={_fmt(t.get('total_cost_pnl'), 2)}, gross={_fmt(t.get('total_gross_pnl'), 2)})."
                )
                cost_flag = True
                break
        if not cost_flag:
            lines.append("2. **Why it matters:** Metrics are directionally consistent across runs.")

    # ── Line 3: What may be misleading ──
    if has_sharpe_rank:
        best_idx = rank_df["rank_sharpe"].idxmin()
        best_run_id = rank_df.loc[best_idx, "run_id"]
        best_ev = next((e for e in evals if e["run_id"] == best_run_id), None)
        n_trades = best_ev.get("trading_metrics", {}).get("num_trades", 0) if best_ev else 0

        if n_trades < 30:
            lines.append(
                f"3. **What may be misleading:** Top run ({best_run_id}) has only {n_trades} trades "
                f"— results may not be statistically significant."
            )
        elif len(rank_df) >= 2:
            sharpe_vals = rank_df.sort_values("rank_sharpe")["trade_sharpe_per_bar"]
            if sharpe_vals.notna().sum() >= 2:
                top_two = sharpe_vals.dropna().values[:2]
                if abs(top_two[0] - top_two[1]) < 0.001:
                    lines.append(
                        "3. **What may be misleading:** Sharpe difference between rank 1 and rank 2 is < 0.001 "
                        "— ranking may be noise."
                    )
                else:
                    lines.append(
                        "3. **What may be misleading:** Walk-forward results may not generalize to unseen regimes."
                    )
            else:
                lines.append(
                    "3. **What may be misleading:** Walk-forward results may not generalize to unseen regimes."
                )
        else:
            lines.append(
                "3. **What may be misleading:** Walk-forward results may not generalize to unseen regimes."
            )
    else:
        lines.append(
            "3. **What may be misleading:** Insufficient data to assess statistical significance."
        )

    return lines


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

    # ── Config match verification ──
    matched, config_warnings = _check_config_match(run_dirs)

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
    md = _build_comparison_md(
        comp_df, rank_df, evals,
        config_matched=matched, config_warnings=config_warnings,
    )
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
    *,
    config_matched: bool = True,
    config_warnings: list[str] | None = None,
) -> str:
    lines: list[str] = []
    lines.append(f"# Run Comparison ({len(evals)} runs)")
    lines.append("")

    # Config Assumptions section
    lines.append("## Config Assumptions")
    lines.append("")
    if config_matched:
        lines.append("All runs share identical data, splits, and execution costs.")
    else:
        lines.append("**WARNING: Config mismatches detected.**")
        for w in (config_warnings or []):
            lines.append(f"- {w}")
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

    # Auto-generated conclusion
    lines.append("## Conclusion")
    lines.append("")
    conclusion_lines = _generate_conclusion(rank_df, evals)
    lines.extend(conclusion_lines)
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by `src.eval` — {len(evals)} runs compared*")
    lines.append("")

    return "\n".join(lines)
