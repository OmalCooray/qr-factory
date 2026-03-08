"""Promotion gate — deterministic pass/fail evaluation for forward-testing candidates.

Evaluates a normalized campaign against configurable rules and produces
a structured verdict with per-rule outcomes and reasons.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.robustness.load import NormalizedCampaign

log = logging.getLogger(__name__)

# ── Default gate thresholds ──────────────────────────────────────────────────

DEFAULT_GATE_CONFIG: dict[str, Any] = {
    "min_net_pnl": 0.0,
    "min_sharpe": 0.0,
    "max_drawdown_pct": 35.0,
    "max_cost_stress_net_pnl_drop_pct": 50.0,
    "max_latency_net_pnl_drop_pct": 30.0,
    "min_stable_sweep_fraction": 0.60,
}


# ── Data structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RuleResult:
    """Outcome of a single promotion gate rule."""

    rule_id: str
    status: str  # "PASS", "FAIL", "SKIP"
    observed_value: float | None
    threshold: float | None
    message: str


@dataclass
class PromotionVerdict:
    """Full promotion gate verdict for a campaign candidate."""

    candidate_id: str
    baseline_reference: str
    overall_verdict: str  # "PASS" or "FAIL"
    passed_rules_count: int
    failed_rules_count: int
    skipped_rules_count: int
    rule_results: list[RuleResult] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    confidence_note: str = ""
    generated_at: str = ""


# ── Rule implementations ─────────────────────────────────────────────────────


def _rule_baseline_net_pnl(
    campaign: NormalizedCampaign,
    config: dict[str, Any],
) -> RuleResult:
    """Rule 1a: Baseline net PnL must exceed configured minimum."""
    threshold = config.get("min_net_pnl", 0.0)
    net_pnl = campaign.baseline_metrics.get("total_net_pnl")

    if net_pnl is None:
        return RuleResult(
            rule_id="baseline_net_pnl",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="Baseline net PnL not available",
        )

    passed = net_pnl >= threshold
    return RuleResult(
        rule_id="baseline_net_pnl",
        status="PASS" if passed else "FAIL",
        observed_value=round(net_pnl, 4),
        threshold=threshold,
        message=(
            f"Baseline net PnL = {net_pnl:.4f} >= {threshold}"
            if passed
            else f"Baseline net PnL = {net_pnl:.4f} < {threshold} (unprofitable)"
        ),
    )


def _rule_baseline_sharpe(
    campaign: NormalizedCampaign,
    config: dict[str, Any],
) -> RuleResult:
    """Rule 1b: Baseline Sharpe must exceed configured minimum (if available)."""
    threshold = config.get("min_sharpe", 0.0)
    sharpe = campaign.baseline_metrics.get("sharpe_per_bar")

    if sharpe is None:
        return RuleResult(
            rule_id="baseline_sharpe",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="Baseline Sharpe not available",
        )

    passed = sharpe >= threshold
    return RuleResult(
        rule_id="baseline_sharpe",
        status="PASS" if passed else "FAIL",
        observed_value=round(sharpe, 6),
        threshold=threshold,
        message=(
            f"Baseline Sharpe = {sharpe:.6f} >= {threshold}"
            if passed
            else f"Baseline Sharpe = {sharpe:.6f} < {threshold}"
        ),
    )


def _rule_drawdown_safety(
    campaign: NormalizedCampaign,
    config: dict[str, Any],
) -> RuleResult:
    """Rule 2: Max drawdown must not exceed configured threshold."""
    threshold = config.get("max_drawdown_pct", 35.0)
    dd = campaign.baseline_metrics.get("max_drawdown_pct")

    if dd is None:
        return RuleResult(
            rule_id="drawdown_safety",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="Max drawdown not available",
        )

    passed = dd <= threshold
    return RuleResult(
        rule_id="drawdown_safety",
        status="PASS" if passed else "FAIL",
        observed_value=round(dd, 4),
        threshold=threshold,
        message=(
            f"Max drawdown = {dd:.2f}% <= {threshold:.1f}%"
            if passed
            else f"Max drawdown = {dd:.2f}% > {threshold:.1f}% (excessive risk)"
        ),
    )


def _rule_cost_robustness(
    campaign: NormalizedCampaign,
    config: dict[str, Any],
) -> RuleResult:
    """Rule 3: Cost-stressed scenarios must not collapse beyond tolerance.

    Compares worst-case cost-stress net PnL drop vs baseline.
    """
    threshold = config.get("max_cost_stress_net_pnl_drop_pct", 50.0)
    baseline_pnl = campaign.baseline_metrics.get("total_net_pnl")

    completed_cost = [s for s in campaign.cost_stress if s.status == "completed"]
    if not completed_cost:
        return RuleResult(
            rule_id="cost_robustness",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="No completed cost stress scenarios",
        )

    if baseline_pnl is None or baseline_pnl == 0:
        return RuleResult(
            rule_id="cost_robustness",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="Baseline net PnL unavailable or zero — cannot compute drop %",
        )

    # Find worst drop (most negative % change from baseline)
    worst_drop_pct = 0.0
    worst_scenario = ""
    for s in completed_cost:
        s_pnl = s.metrics.get("total_net_pnl")
        if s_pnl is None:
            continue
        if baseline_pnl > 0:
            drop_pct = ((baseline_pnl - s_pnl) / abs(baseline_pnl)) * 100
        else:
            # Baseline already negative — any further loss is a drop
            drop_pct = ((baseline_pnl - s_pnl) / abs(baseline_pnl)) * 100
        if drop_pct > worst_drop_pct:
            worst_drop_pct = drop_pct
            worst_scenario = s.scenario_id

    passed = worst_drop_pct <= threshold
    return RuleResult(
        rule_id="cost_robustness",
        status="PASS" if passed else "FAIL",
        observed_value=round(worst_drop_pct, 2),
        threshold=threshold,
        message=(
            f"Worst cost-stress PnL drop = {worst_drop_pct:.1f}% <= {threshold:.1f}%"
            if passed
            else (
                f"Worst cost-stress PnL drop = {worst_drop_pct:.1f}% > {threshold:.1f}%"
                f" (scenario: {worst_scenario})"
            )
        ),
    )


def _rule_latency_robustness(
    campaign: NormalizedCampaign,
    config: dict[str, Any],
) -> RuleResult:
    """Rule 4: Latency proxy must not degrade beyond tolerance."""
    threshold = config.get("max_latency_net_pnl_drop_pct", 30.0)
    baseline_pnl = campaign.baseline_metrics.get("total_net_pnl")

    completed_latency = [s for s in campaign.latency_proxy if s.status == "completed"]
    if not completed_latency:
        return RuleResult(
            rule_id="latency_robustness",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="No completed latency proxy scenarios",
        )

    if baseline_pnl is None or baseline_pnl == 0:
        return RuleResult(
            rule_id="latency_robustness",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="Baseline net PnL unavailable or zero — cannot compute drop %",
        )

    worst_drop_pct = 0.0
    worst_scenario = ""
    for s in completed_latency:
        s_pnl = s.metrics.get("total_net_pnl")
        if s_pnl is None:
            continue
        drop_pct = ((baseline_pnl - s_pnl) / abs(baseline_pnl)) * 100
        if drop_pct > worst_drop_pct:
            worst_drop_pct = drop_pct
            worst_scenario = s.scenario_id

    passed = worst_drop_pct <= threshold
    return RuleResult(
        rule_id="latency_robustness",
        status="PASS" if passed else "FAIL",
        observed_value=round(worst_drop_pct, 2),
        threshold=threshold,
        message=(
            f"Worst latency PnL drop = {worst_drop_pct:.1f}% <= {threshold:.1f}%"
            if passed
            else (
                f"Worst latency PnL drop = {worst_drop_pct:.1f}% > {threshold:.1f}%"
                f" (scenario: {worst_scenario})"
            )
        ),
    )


def _rule_parameter_stability(
    campaign: NormalizedCampaign,
    config: dict[str, Any],
) -> RuleResult:
    """Rule 5: Sufficient fraction of parameter sweep scenarios must be stable.

    A sweep scenario is "stable" if its net PnL is non-negative (>= 0).
    """
    threshold = config.get("min_stable_sweep_fraction", 0.60)

    completed_sweeps = [s for s in campaign.parameter_sweep if s.status == "completed"]
    if not completed_sweeps:
        return RuleResult(
            rule_id="parameter_stability",
            status="SKIP",
            observed_value=None,
            threshold=threshold,
            message="No completed parameter sweep scenarios",
        )

    n_stable = 0
    for s in completed_sweeps:
        s_pnl = s.metrics.get("total_net_pnl")
        if s_pnl is not None and s_pnl >= 0:
            n_stable += 1

    fraction = n_stable / len(completed_sweeps)
    passed = fraction >= threshold
    return RuleResult(
        rule_id="parameter_stability",
        status="PASS" if passed else "FAIL",
        observed_value=round(fraction, 4),
        threshold=threshold,
        message=(
            f"Stable sweep fraction = {fraction:.1%} ({n_stable}/{len(completed_sweeps)})"
            f" >= {threshold:.0%}"
            if passed
            else (
                f"Stable sweep fraction = {fraction:.1%} ({n_stable}/{len(completed_sweeps)})"
                f" < {threshold:.0%} (fragile to parameter perturbation)"
            )
        ),
    )


# ── Gate evaluation ──────────────────────────────────────────────────────────

_RULES = [
    _rule_baseline_net_pnl,
    _rule_baseline_sharpe,
    _rule_drawdown_safety,
    _rule_cost_robustness,
    _rule_latency_robustness,
    _rule_parameter_stability,
]


def evaluate_promotion_gate(
    campaign: NormalizedCampaign,
    gate_config: dict[str, Any] | None = None,
) -> PromotionVerdict:
    """Evaluate all promotion gate rules against a normalized campaign.

    Parameters
    ----------
    campaign : NormalizedCampaign
        Loaded campaign data.
    gate_config : dict | None
        Gate rule thresholds. Uses DEFAULT_GATE_CONFIG for missing keys.

    Returns
    -------
    PromotionVerdict
        Deterministic verdict with per-rule outcomes.
    """
    config = dict(DEFAULT_GATE_CONFIG)
    if gate_config:
        config.update(gate_config)

    results: list[RuleResult] = []
    for rule_fn in _RULES:
        results.append(rule_fn(campaign, config))

    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]
    skipped = [r for r in results if r.status == "SKIP"]

    # Verdict: PASS only if zero failures (skipped rules are neutral)
    overall = "PASS" if len(failed) == 0 else "FAIL"

    reasons = [r.message for r in failed]
    if not reasons and overall == "PASS":
        reasons = ["All evaluated rules passed"]

    # Confidence note
    caveats: list[str] = []
    if skipped:
        skip_ids = ", ".join(r.rule_id for r in skipped)
        caveats.append(f"Skipped rules (insufficient data): {skip_ids}")
    if campaign.n_completed == 0:
        caveats.append("No scenario runs completed — verdict based on baseline only")
    confidence_note = "; ".join(caveats) if caveats else "All rules evaluated with data"

    return PromotionVerdict(
        candidate_id=campaign.campaign_id,
        baseline_reference=campaign.baseline_run_id or campaign.baseline_config,
        overall_verdict=overall,
        passed_rules_count=len(passed),
        failed_rules_count=len(failed),
        skipped_rules_count=len(skipped),
        rule_results=results,
        reasons=reasons,
        confidence_note=confidence_note,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def write_promotion_artifacts(
    verdict: PromotionVerdict,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write promotion_gate.json and promotion_gate.md to the output directory.

    Returns
    -------
    tuple[Path, Path]
        Paths to (promotion_gate.json, promotion_gate.md).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON artifact
    json_path = output_dir / "promotion_gate.json"
    verdict_dict = asdict(verdict)
    json_path.write_text(
        json.dumps(verdict_dict, indent=2, default=str),
        encoding="utf-8",
    )

    # Markdown artifact
    md_path = output_dir / "promotion_gate.md"
    lines: list[str] = [
        f"# Promotion Gate — {verdict.candidate_id}",
        "",
        f"**Verdict: {verdict.overall_verdict}**",
        "",
        f"- Baseline: `{verdict.baseline_reference}`",
        f"- Passed: {verdict.passed_rules_count}",
        f"- Failed: {verdict.failed_rules_count}",
        f"- Skipped: {verdict.skipped_rules_count}",
        "",
        "## Rule Results",
        "",
        "| Rule | Status | Observed | Threshold | Message |",
        "|------|--------|----------|-----------|---------|",
    ]
    for r in verdict.rule_results:
        obs = f"{r.observed_value}" if r.observed_value is not None else "N/A"
        thr = f"{r.threshold}" if r.threshold is not None else "N/A"
        lines.append(f"| {r.rule_id} | {r.status} | {obs} | {thr} | {r.message} |")

    if verdict.reasons:
        lines.extend(["", "## Reasons", ""])
        for reason in verdict.reasons:
            lines.append(f"- {reason}")

    if verdict.confidence_note:
        lines.extend(["", f"**Note:** {verdict.confidence_note}"])

    lines.extend(["", f"*Generated: {verdict.generated_at}*", ""])

    md_path.write_text("\n".join(lines), encoding="utf-8")

    log.info("Wrote %s", json_path)
    log.info("Wrote %s", md_path)
    return json_path, md_path
