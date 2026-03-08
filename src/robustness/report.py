"""Robustness report generator — deterministic markdown report from campaign artifacts.

Reads normalized campaign data and promotion verdict, produces a concise,
evidence-led robustness report covering all stress families evaluated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.robustness.load import NormalizedCampaign, ScenarioResult
from src.robustness.promotion import PromotionVerdict

log = logging.getLogger(__name__)


def _fmt(value: Any, decimals: int = 4) -> str:
    """Format a numeric value or return 'N/A'."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _pct(value: Any) -> str:
    """Format as percentage string."""
    if value is None:
        return "N/A"
    return f"{value:.2f}%"


def _scenario_pnl_table(scenarios: list[ScenarioResult], baseline_pnl: float | None) -> list[str]:
    """Build a markdown table comparing scenario PnLs to baseline."""
    lines = [
        "| Scenario | Net PnL | Gross PnL | Max DD % | Trades | vs Baseline |",
        "|----------|---------|-----------|----------|--------|-------------|",
    ]
    for s in scenarios:
        if s.status != "completed":
            lines.append(f"| {s.scenario_id} | (not completed) | | | | |")
            continue
        m = s.metrics
        net = m.get("total_net_pnl")
        gross = m.get("total_gross_pnl")
        dd = m.get("max_drawdown_pct")
        trades = m.get("num_trades")

        delta = ""
        if net is not None and baseline_pnl is not None and baseline_pnl != 0:
            change_pct = ((net - baseline_pnl) / abs(baseline_pnl)) * 100
            delta = f"{change_pct:+.1f}%"

        lines.append(
            f"| {s.scenario_id} | {_fmt(net)} | {_fmt(gross)} | {_pct(dd)} | {_fmt(trades, 0)} | {delta} |"
        )
    return lines


def generate_robustness_report(
    campaign: NormalizedCampaign,
    verdict: PromotionVerdict,
    output_dir: Path | None = None,
) -> str:
    """Generate a robustness report as markdown text.

    Parameters
    ----------
    campaign : NormalizedCampaign
        Loaded and normalized campaign data.
    verdict : PromotionVerdict
        Promotion gate evaluation result.
    output_dir : Path | None
        If provided, writes ``robustness_report.md`` to this directory.

    Returns
    -------
    str
        The full markdown report text.
    """
    bm = campaign.baseline_metrics
    baseline_pnl = bm.get("total_net_pnl")

    sections: list[str] = []

    # ── A. Executive Summary ─────────────────────────────────────────────
    sections.append("# Robustness Report")
    sections.append("")
    sections.append(f"**Candidate:** {campaign.campaign_id}")
    sections.append(f"**Baseline config:** `{campaign.baseline_config}`")
    if campaign.baseline_run_id:
        sections.append(f"**Baseline run:** `{campaign.baseline_run_id}`")
    sections.append(f"**Scenarios evaluated:** {len(campaign.all_scenarios)}"
                     f" ({campaign.n_completed} completed, {campaign.n_failed} failed)")
    sections.append("")
    sections.append(f"**Promotion verdict: {verdict.overall_verdict}**"
                     f" ({verdict.passed_rules_count} passed,"
                     f" {verdict.failed_rules_count} failed,"
                     f" {verdict.skipped_rules_count} skipped)")
    sections.append("")
    if campaign.description:
        sections.append(f"> {campaign.description}")
        sections.append("")

    # ── B. Canonical Baseline Performance ─────────────────────────────────
    sections.append("## Baseline Performance")
    sections.append("")
    if not bm:
        sections.append("*Baseline metrics not available.*")
    else:
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        sections.append(f"| Net PnL | {_fmt(bm.get('total_net_pnl'))} |")
        sections.append(f"| Gross PnL | {_fmt(bm.get('total_gross_pnl'))} |")
        sections.append(f"| Total Costs | {_fmt(bm.get('total_cost_pnl'))} |")
        sections.append(f"| Sharpe (per bar) | {_fmt(bm.get('sharpe_per_bar'), 6)} |")
        sections.append(f"| Max Drawdown | {_pct(bm.get('max_drawdown_pct'))} |")
        sections.append(f"| Trades | {_fmt(bm.get('num_trades'), 0)} |")
        sections.append(f"| Win Rate | {_pct(bm.get('win_rate'))} |")
        sections.append(f"| Profit Factor | {_fmt(bm.get('profit_factor'))} |")
        sections.append(f"| Avg Trade PnL | {_fmt(bm.get('avg_trade_pnl'))} |")
    sections.append("")

    # ── C. Cost Sensitivity ──────────────────────────────────────────────
    sections.append("## Cost Sensitivity")
    sections.append("")
    if not campaign.cost_stress:
        sections.append("*No cost stress scenarios in this campaign.*")
    else:
        sections.extend(_scenario_pnl_table(campaign.cost_stress, baseline_pnl))

        # Interpretation
        sections.append("")
        completed_cost = [s for s in campaign.cost_stress if s.status == "completed"]
        if completed_cost and baseline_pnl is not None:
            worst = min(
                (s.metrics.get("total_net_pnl", 0) for s in completed_cost),
                default=0,
            )
            if baseline_pnl > 0 and worst < 0:
                sections.append("**Warning:** Strategy turns negative under cost stress — performance is friction-dominated.")
            elif baseline_pnl > 0:
                drop = ((baseline_pnl - worst) / abs(baseline_pnl)) * 100
                sections.append(f"Worst-case cost-stress PnL drop: {drop:.1f}% from baseline.")
    sections.append("")

    # ── D. Latency Sensitivity ───────────────────────────────────────────
    sections.append("## Latency Sensitivity")
    sections.append("")
    if not campaign.latency_proxy:
        sections.append("*No latency proxy scenarios in this campaign.*")
    else:
        sections.extend(_scenario_pnl_table(campaign.latency_proxy, baseline_pnl))

        sections.append("")
        completed_lat = [s for s in campaign.latency_proxy if s.status == "completed"]
        if completed_lat and baseline_pnl is not None and baseline_pnl != 0:
            worst = min(
                (s.metrics.get("total_net_pnl", 0) for s in completed_lat),
                default=0,
            )
            drop = ((baseline_pnl - worst) / abs(baseline_pnl)) * 100
            sections.append(f"Worst-case latency PnL drop: {drop:.1f}% from baseline.")
            if drop > 30:
                sections.append("**Warning:** Strategy is timing-fragile — sensitive to execution delay.")
    sections.append("")

    # ── E. Parameter Stability ───────────────────────────────────────────
    sections.append("## Parameter Stability")
    sections.append("")
    if not campaign.parameter_sweep:
        sections.append("*No parameter sweep scenarios in this campaign.*")
    else:
        sections.extend(_scenario_pnl_table(campaign.parameter_sweep, baseline_pnl))

        sections.append("")
        completed_sw = [s for s in campaign.parameter_sweep if s.status == "completed"]
        if completed_sw:
            n_stable = sum(
                1 for s in completed_sw
                if s.metrics.get("total_net_pnl") is not None and s.metrics["total_net_pnl"] >= 0
            )
            frac = n_stable / len(completed_sw) if completed_sw else 0
            sections.append(
                f"Stable scenarios (net PnL >= 0): {n_stable}/{len(completed_sw)} ({frac:.0%})"
            )
            if frac < 0.6:
                sections.append("**Warning:** Strategy is fragile to local parameter perturbation.")
    sections.append("")

    # ── F. Regime / Session Findings ─────────────────────────────────────
    sections.append("## Regime and Session Analysis")
    sections.append("")

    if campaign.baseline_regime_metrics:
        sections.append("### Volatility Regimes (Baseline)")
        sections.append("")
        sections.append("| Regime | Trades | Net PnL | Win Rate |")
        sections.append("|--------|--------|---------|----------|")
        for regime, rm in sorted(campaign.baseline_regime_metrics.items()):
            if isinstance(rm, dict):
                sections.append(
                    f"| {regime} | {rm.get('n_trades', 'N/A')} "
                    f"| {_fmt(rm.get('net_pnl'))} | {_pct(rm.get('win_rate'))} |"
                )
        sections.append("")

    if campaign.baseline_session_metrics:
        sections.append("### Trading Sessions (Baseline)")
        sections.append("")
        sections.append("| Session | Trades | Net PnL | Win Rate |")
        sections.append("|---------|--------|---------|----------|")
        for session, sm in sorted(campaign.baseline_session_metrics.items()):
            if isinstance(sm, dict):
                sections.append(
                    f"| {session} | {sm.get('n_trades', 'N/A')} "
                    f"| {_fmt(sm.get('net_pnl'))} | {_pct(sm.get('win_rate'))} |"
                )
        sections.append("")

    if not campaign.baseline_regime_metrics and not campaign.baseline_session_metrics:
        sections.append("*Regime and session analysis not available for baseline.*")
        sections.append("")

    # ── G. Bottom-Line Interpretation ────────────────────────────────────
    sections.append("## Bottom Line")
    sections.append("")

    # Where it works
    works: list[str] = []
    if baseline_pnl is not None and baseline_pnl > 0:
        works.append("Baseline is profitable under canonical transaction costs")
    if campaign.baseline_regime_metrics:
        best_regime = max(
            ((k, v.get("net_pnl", 0)) for k, v in campaign.baseline_regime_metrics.items() if isinstance(v, dict)),
            key=lambda x: x[1] if x[1] is not None else float("-inf"),
            default=None,
        )
        if best_regime and best_regime[1] is not None and best_regime[1] > 0:
            works.append(f"Strongest in {best_regime[0]} regime (net PnL {best_regime[1]:.4f})")

    # Where it fails
    fails: list[str] = []
    if baseline_pnl is not None and baseline_pnl <= 0:
        fails.append("Baseline is unprofitable — no edge demonstrated")
    for r in verdict.rule_results:
        if r.status == "FAIL":
            fails.append(r.message)

    # Assumptions
    assumptions = [
        "Results assume the backtest transaction cost model (spread + slippage) is representative",
        "Walk-forward validation was used (if configured), but live market microstructure may differ",
        "Parameter sweep covers local neighborhood only — global fragility not tested",
    ]

    sections.append("### Where it works")
    if works:
        for w in works:
            sections.append(f"- {w}")
    else:
        sections.append("- No clear strengths identified from the evidence")
    sections.append("")

    sections.append("### Where it fails")
    if fails:
        for f in fails:
            sections.append(f"- {f}")
    else:
        sections.append("- No failures detected in the evaluated scenarios")
    sections.append("")

    sections.append("### Assumptions and caveats")
    for a in assumptions:
        sections.append(f"- {a}")
    if verdict.confidence_note:
        sections.append(f"- {verdict.confidence_note}")
    sections.append("")

    # ── Promotion Verdict Summary ────────────────────────────────────────
    sections.append("## Promotion Verdict")
    sections.append("")
    sections.append(f"**{verdict.overall_verdict}**"
                     f" — {verdict.passed_rules_count} passed,"
                     f" {verdict.failed_rules_count} failed,"
                     f" {verdict.skipped_rules_count} skipped")
    sections.append("")
    if verdict.reasons:
        for reason in verdict.reasons:
            sections.append(f"- {reason}")
    sections.append("")
    sections.append(f"*Generated: {verdict.generated_at}*")
    sections.append("")

    report_text = "\n".join(sections)

    # Write to file if output directory given
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report_path = output_dir / "robustness_report.md"
        report_path.write_text(report_text, encoding="utf-8")
        log.info("Wrote %s", report_path)

    return report_text
