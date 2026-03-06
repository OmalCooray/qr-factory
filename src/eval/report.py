"""Per-run markdown report and drawdown plot generation."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from ._formatting import fmt as _fmt  # noqa: E402

log = logging.getLogger(__name__)


def _plot_drawdown(
    equity: pd.DataFrame,
    out_path: Path,
    equity_gross: pd.DataFrame | None = None,
) -> None:
    """Generate drawdown curve PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    eq_values = equity["equity"].values.astype(float)
    rolling_peak = np.maximum.accumulate(eq_values)
    dd_pct = np.where(rolling_peak > 0, (rolling_peak - eq_values) / rolling_peak * 100, 0.0)

    # Parse timestamps
    if "timestamp" in equity.columns:
        ts = pd.to_datetime(equity["timestamp"])
    else:
        ts = np.arange(len(eq_values))

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(ts, dd_pct, 0, color="#e74c3c", alpha=0.4, label="Net DD%")
    ax.plot(ts, dd_pct, color="#e74c3c", linewidth=0.8)

    if equity_gross is not None and not equity_gross.empty:
        eq_g = equity_gross["equity"].values.astype(float)
        peak_g = np.maximum.accumulate(eq_g)
        dd_g = np.where(peak_g > 0, (peak_g - eq_g) / peak_g * 100, 0.0)
        ax.plot(ts[:len(dd_g)], dd_g, color="#3498db", linewidth=0.8, alpha=0.7, label="Gross DD%")

    ax.invert_yaxis()
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Drawdown %")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info("Saved drawdown plot → %s", out_path)


def _metrics_table(metrics: dict, title: str) -> str:
    """Build a markdown table from a flat dict."""
    lines = [f"### {title}", "", "| Metric | Value |", "|--------|------:|"]
    for k, v in sorted(metrics.items()):
        lines.append(f"| {k} | {_fmt(v)} |")
    return "\n".join(lines)


def _disagreement_notes(pred: dict, trade: dict) -> str:
    """Heuristic disagreement analysis between prediction and trading quality."""
    notes: list[str] = []

    if not pred:
        notes.append("- No prediction metrics available (rule-based or non-ML run).")
        return "\n".join(notes) if notes else "No disagreement detected."

    roc_auc = pred.get("roc_auc")
    brier = pred.get("brier")
    profit_factor = trade.get("profit_factor", 0)
    total_cost = trade.get("total_cost_pnl", 0)
    total_net = trade.get("total_net_pnl", 0)
    total_gross = trade.get("total_gross_pnl", 0)
    sharpe = trade.get("sharpe_per_bar", 0)

    # Good predictions + bad trading
    good_pred = (roc_auc is not None and roc_auc > 0.55) or (brier is not None and brier < 0.22)
    bad_trade = profit_factor < 1.0 or (total_net < 0)
    if good_pred and bad_trade:
        notes.append("- **Good predictions + bad trading**: Model has predictive skill but profits are negative.")
        if abs(total_gross) > 0 and abs(total_cost) > 0.3 * abs(total_gross):
            notes.append("  - Likely cause: **cost dominance** — transaction costs exceed trading edge.")
        else:
            notes.append("  - Possible cause: **policy mismatch** — threshold or sizing not capitalizing on signal.")

    # Bad predictions + good trading
    bad_pred = (roc_auc is not None and roc_auc < 0.52) or (brier is not None and brier > 0.25)
    good_trade = profit_factor > 1.0 and total_net > 0
    if bad_pred and good_trade:
        notes.append("- **Bad predictions + good trading**: Profits may be due to **luck** or market regime, not model skill.")

    # Good calibration + low profit factor
    if brier is not None and brier < 0.22 and 0 < profit_factor < 1.2:
        notes.append("- **Good calibration + low profit_factor**: Policy may not be capitalizing on well-calibrated probabilities.")

    if not notes:
        notes.append("No significant metric disagreement detected.")

    return "\n".join(notes)


def generate_report(
    run_dir: str | Path,
    evaluation_dict: dict,
    equity_df: pd.DataFrame,
    equity_gross_df: pd.DataFrame | None = None,
) -> Path:
    """Generate report.md and plots/drawdown.png for a single run.

    Parameters
    ----------
    run_dir : str | Path
        Run directory to write report into.
    evaluation_dict : dict
        Output of ``asdict(EvaluationResult)``.
    equity_df : DataFrame
        Net equity curve.
    equity_gross_df : DataFrame | None
        Gross equity curve.

    Returns
    -------
    Path
        Path to the generated report.md.
    """
    run_dir = Path(run_dir)
    pred = evaluation_dict.get("prediction_metrics", {})
    trade = evaluation_dict.get("trading_metrics", {})
    meta = evaluation_dict.get("metadata", {})

    lines: list[str] = []

    # 1. Run Summary
    lines.append("# Evaluation Report")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- **Run ID**: {evaluation_dict.get('run_id', 'unknown')}")
    lines.append(f"- **Strategy**: {evaluation_dict.get('strategy_type', 'unknown')}")
    lines.append(f"- **Symbol**: {meta.get('symbol', 'N/A')} {meta.get('timeframe', '')}")
    lines.append(f"- **Period**: {meta.get('start_ts', 'N/A')} to {meta.get('end_ts', 'N/A')}")
    lines.append(f"- **Bars**: {meta.get('n_bars', 'N/A')}")
    lines.append(f"- **Starting Capital**: {_fmt(meta.get('starting_capital'), 2)}")
    lines.append(f"- **Ending Equity**: {_fmt(meta.get('ending_equity'), 2)}")
    if meta.get("is_walk_forward"):
        lines.append(f"- **Walk-Forward**: {meta.get('n_folds', '?')} folds")
    lines.append(f"- **Git**: {meta.get('git_commit', 'N/A')} (dirty={meta.get('git_dirty', 'N/A')})")
    lines.append("")

    # 1b. Strategy & Label Config (policy observability)
    cfg_path = run_dir / "config.yaml"
    if cfg_path.exists():
        run_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    else:
        run_cfg = {}

    lines.append("## Strategy & Label Config")
    lines.append("")

    strat_cfg = run_cfg.get("strategy", {})
    if strat_cfg:
        lines.append(f"- **Strategy type**: {strat_cfg.get('type', 'N/A')}")
        for key in ("p_enter", "p_exit", "min_hold_bars"):
            if key in strat_cfg:
                lines.append(f"- **{key}**: {strat_cfg[key]}")
        feat_list = strat_cfg.get("features", [])
        if feat_list:
            lines.append(f"- **Features**: {len(feat_list)} indicators")

    label_cfg = run_cfg.get("label", {})
    if label_cfg:
        lines.append(f"- **Label type**: {label_cfg.get('type', 'N/A')}")
        for key in ("horizon", "cost_buffer", "direction"):
            if key in label_cfg:
                lines.append(f"- **label.{key}**: {label_cfg[key]}")

    model_cfg = run_cfg.get("model", {})
    if model_cfg:
        lines.append(f"- **Model type**: {model_cfg.get('type', 'N/A')}")
        for key, val in model_cfg.items():
            if key != "type":
                lines.append(f"- **model.{key}**: {val}")

    if not strat_cfg and not label_cfg and not model_cfg:
        lines.append("N/A — no config.yaml found.")
    lines.append("")

    # 2. Costs & Friction
    lines.append("## Costs & Friction")
    lines.append("")
    lines.append(f"- **Total Cost PnL**: {_fmt(trade.get('total_cost_pnl'), 2)}")
    lines.append(f"- **Avg Cost/Trade**: {_fmt(trade.get('avg_cost_per_trade'), 4)}")
    lines.append(f"- **Total Gross PnL**: {_fmt(trade.get('total_gross_pnl'), 2)}")
    lines.append("")

    # 3. Prediction Metrics
    lines.append("## Prediction Metrics")
    lines.append("")
    if pred:
        lines.append(_metrics_table(pred, "Prediction Quality"))
        if meta.get("is_walk_forward"):
            lines.append("")
            lines.append(
                "> **Note (walk-forward):** Prediction metrics are pooled"
                " across all folds (each prediction counted once). Per-fold"
                " averaged values in `fold_metrics.json` may differ slightly"
                " when fold sizes are unequal."
            )
    else:
        lines.append("N/A — no predictions.csv found (rule-based or non-ML run).")
    lines.append("")

    # 4. Trading Metrics
    lines.append("## Trading Metrics")
    lines.append("")
    lines.append(_metrics_table(trade, "Trading Performance"))
    lines.append("")

    # 5. Drawdown
    lines.append("## Drawdown")
    lines.append("")
    lines.append(f"- **Max Drawdown**: {_fmt(trade.get('max_drawdown'), 2)}")
    lines.append(f"- **Max DD %**: {_fmt(trade.get('max_dd_pct'), 2)}%")
    lines.append("")
    lines.append("![Drawdown](plots/drawdown.png)")
    lines.append("")
    if meta.get("is_walk_forward"):
        lines.append(
            "> **Note (walk-forward):** Drawdown above is computed from the"
            " continuous compounded equity curve across all folds. Per-fold"
            " worst drawdown in `metrics.json` may differ because it reports"
            " the single worst fold, not the cross-fold peak-to-trough."
        )
        lines.append("")

    # 6. Gross vs Net
    if "sharpe_per_bar_gross" in trade:
        lines.append("## Gross vs Net")
        lines.append("")
        lines.append(f"- **Sharpe (net, per bar)**: {_fmt(trade.get('sharpe_per_bar'), 6)}")
        lines.append(f"- **Sharpe (gross, per bar)**: {_fmt(trade.get('sharpe_per_bar_gross'), 6)}")
        lines.append("")

    # 7. Disagreement Notes
    lines.append("## Metric Disagreement Notes")
    lines.append("")
    lines.append(_disagreement_notes(pred, trade))
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated by `src.eval`*")
    lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote report.md → %s", report_path)

    # Generate drawdown plot
    if not equity_df.empty:
        _plot_drawdown(
            equity_df,
            run_dir / "plots" / "drawdown.png",
            equity_gross_df,
        )

    return report_path
