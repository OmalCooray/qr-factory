"""Tests for robustness report generation and promotion gate evaluation.

Covers:
- Normalized campaign loader with synthetic fixtures
- Promotion gate deterministic evaluation (PASS / FAIL cases)
- Stable reason strings
- Report generation from fixed inputs
- Missing optional files handled gracefully
- Artifact writing
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.robustness.load import NormalizedCampaign, ScenarioResult, load_campaign
from src.robustness.promotion import (
    DEFAULT_GATE_CONFIG,
    PromotionVerdict,
    RuleResult,
    evaluate_promotion_gate,
    write_promotion_artifacts,
)
from src.robustness.report import generate_robustness_report


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════


def _make_passing_campaign(tmp_path: Path) -> NormalizedCampaign:
    """Create a synthetic campaign that should PASS all rules."""
    return NormalizedCampaign(
        campaign_id="test_pass",
        campaign_dir=tmp_path,
        baseline_config="configs/test_baseline.yaml",
        baseline_run_id="20260308_120000",
        description="Synthetic passing campaign",
        baseline_metrics={
            "total_net_pnl": 500.0,
            "total_gross_pnl": 800.0,
            "total_cost_pnl": 300.0,
            "sharpe_per_bar": 0.02,
            "max_drawdown_pct": 15.0,
            "num_trades": 100,
            "win_rate": 55.0,
            "profit_factor": 1.5,
            "avg_trade_pnl": 5.0,
        },
        baseline_regime_metrics={
            "low_vol": {"n_trades": 40, "net_pnl": 200.0, "win_rate": 58.0},
            "high_vol": {"n_trades": 60, "net_pnl": 300.0, "win_rate": 53.0},
        },
        baseline_session_metrics={
            "asia": {"n_trades": 30, "net_pnl": 100.0, "win_rate": 50.0},
            "london": {"n_trades": 40, "net_pnl": 250.0, "win_rate": 60.0},
            "new_york": {"n_trades": 30, "net_pnl": 150.0, "win_rate": 55.0},
        },
        cost_stress=[
            ScenarioResult(
                scenario_id="cost_stress__1.5x_cost",
                family="cost_stress",
                name="1.5x_cost",
                description="1.5x friction",
                status="completed",
                run_id="run_cost_1.5x",
                metrics={"total_net_pnl": 350.0, "max_drawdown_pct": 18.0, "num_trades": 100},
            ),
            ScenarioResult(
                scenario_id="cost_stress__2.0x_cost",
                family="cost_stress",
                name="2.0x_cost",
                description="2x friction",
                status="completed",
                run_id="run_cost_2x",
                metrics={"total_net_pnl": 300.0, "max_drawdown_pct": 20.0, "num_trades": 100},
            ),
        ],
        latency_proxy=[
            ScenarioResult(
                scenario_id="latency_proxy__1bar_delay",
                family="latency_proxy",
                name="1bar_delay",
                description="1-bar delay",
                status="completed",
                run_id="run_latency_1bar",
                metrics={"total_net_pnl": 400.0, "max_drawdown_pct": 16.0, "num_trades": 95},
            ),
        ],
        parameter_sweep=[
            ScenarioResult(
                scenario_id=f"parameter_sweep__p_enter_{v}",
                family="parameter_sweep",
                name=f"p_enter_{v}",
                description="",
                status="completed",
                run_id=f"run_sweep_{i}",
                metrics={"total_net_pnl": pnl, "num_trades": 90},
            )
            for i, (v, pnl) in enumerate([
                ("0.35", 400.0),
                ("0.45", 300.0),
                ("0.50", 250.0),
                ("0.55", 100.0),
            ])
        ],
    )


def _make_failing_campaign(tmp_path: Path) -> NormalizedCampaign:
    """Create a synthetic campaign that should FAIL multiple rules."""
    return NormalizedCampaign(
        campaign_id="test_fail",
        campaign_dir=tmp_path,
        baseline_config="configs/test_baseline.yaml",
        baseline_run_id="20260308_130000",
        description="Synthetic failing campaign",
        baseline_metrics={
            "total_net_pnl": -50.0,
            "total_gross_pnl": 200.0,
            "total_cost_pnl": 250.0,
            "sharpe_per_bar": -0.001,
            "max_drawdown_pct": 42.0,
            "num_trades": 80,
            "win_rate": 40.0,
        },
        cost_stress=[
            ScenarioResult(
                scenario_id="cost_stress__2.0x_cost",
                family="cost_stress",
                name="2.0x_cost",
                description="2x friction",
                status="completed",
                run_id="run_cost_2x",
                metrics={"total_net_pnl": -200.0, "max_drawdown_pct": 55.0, "num_trades": 80},
            ),
        ],
        latency_proxy=[
            ScenarioResult(
                scenario_id="latency_proxy__1bar_delay",
                family="latency_proxy",
                name="1bar_delay",
                description="1-bar delay",
                status="completed",
                run_id="run_latency_1bar",
                metrics={"total_net_pnl": -120.0, "max_drawdown_pct": 50.0, "num_trades": 75},
            ),
        ],
        parameter_sweep=[
            ScenarioResult(
                scenario_id=f"parameter_sweep__p_{i}",
                family="parameter_sweep",
                name=f"p_{i}",
                description="",
                status="completed",
                run_id=f"run_sw_{i}",
                metrics={"total_net_pnl": pnl},
            )
            for i, pnl in enumerate([-30.0, -50.0, 10.0, -80.0, -20.0])
        ],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Normalized campaign loader
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadCampaign:

    def test_loads_manifest(self, tmp_path: Path):
        """Loader reads MANIFEST.json and populates campaign fields."""
        manifest = {
            "campaign_id": "test_load",
            "baseline_config": "configs/base.yaml",
            "baseline_run_id": None,
            "description": "Test campaign",
            "scenarios": {
                "cost_stress__2x": {
                    "family": "cost_stress",
                    "name": "2x",
                    "description": "Double",
                    "status": "pending",
                    "run_id": None,
                },
            },
        }
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        campaign = load_campaign(tmp_path)
        assert campaign.campaign_id == "test_load"
        assert len(campaign.cost_stress) == 1
        assert campaign.cost_stress[0].status == "pending"

    def test_missing_manifest_raises(self, tmp_path: Path):
        """Missing MANIFEST.json raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_campaign(tmp_path)

    def test_groups_by_family(self, tmp_path: Path):
        """Scenarios are grouped into correct family lists."""
        manifest = {
            "campaign_id": "grouped",
            "baseline_config": "x",
            "baseline_run_id": None,
            "description": "",
            "scenarios": {
                "cost_stress__a": {"family": "cost_stress", "name": "a", "status": "pending", "run_id": None},
                "latency_proxy__b": {"family": "latency_proxy", "name": "b", "status": "pending", "run_id": None},
                "parameter_sweep__c": {"family": "parameter_sweep", "name": "c", "status": "pending", "run_id": None},
                "other__d": {"family": "other", "name": "d", "status": "pending", "run_id": None},
            },
        }
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        campaign = load_campaign(tmp_path)
        assert len(campaign.cost_stress) == 1
        assert len(campaign.latency_proxy) == 1
        assert len(campaign.parameter_sweep) == 1
        assert len(campaign.other_scenarios) == 1
        assert len(campaign.all_scenarios) == 4

    def test_scenario_table_loaded(self, tmp_path: Path):
        """scenario_metrics.csv is loaded into scenario_table if present."""
        manifest = {
            "campaign_id": "csv_test",
            "baseline_config": "x",
            "baseline_run_id": None,
            "description": "",
            "scenarios": {},
        }
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
        (tmp_path / "scenario_metrics.csv").write_text(
            "scenario_id,family,total_net_pnl\ncost__2x,cost_stress,100.0\n",
            encoding="utf-8",
        )
        campaign = load_campaign(tmp_path)
        assert campaign.scenario_table is not None
        assert len(campaign.scenario_table) == 1


# ══════════════════════════════════════════════════════════════════════════════
# Promotion gate — PASS case
# ══════════════════════════════════════════════════════════════════════════════


class TestPromotionGatePass:

    def test_all_rules_pass(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        assert verdict.overall_verdict == "PASS"
        assert verdict.failed_rules_count == 0
        assert verdict.passed_rules_count > 0

    def test_candidate_id_matches(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        assert verdict.candidate_id == "test_pass"

    def test_reasons_on_pass(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        assert "All evaluated rules passed" in verdict.reasons


# ══════════════════════════════════════════════════════════════════════════════
# Promotion gate — FAIL case
# ══════════════════════════════════════════════════════════════════════════════


class TestPromotionGateFail:

    def test_fails_on_negative_pnl(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        assert verdict.overall_verdict == "FAIL"
        assert verdict.failed_rules_count > 0

    def test_baseline_net_pnl_fails(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        pnl_rule = next(r for r in verdict.rule_results if r.rule_id == "baseline_net_pnl")
        assert pnl_rule.status == "FAIL"

    def test_drawdown_fails(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        dd_rule = next(r for r in verdict.rule_results if r.rule_id == "drawdown_safety")
        assert dd_rule.status == "FAIL"

    def test_sharpe_fails(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        sharpe_rule = next(r for r in verdict.rule_results if r.rule_id == "baseline_sharpe")
        assert sharpe_rule.status == "FAIL"

    def test_parameter_stability_fails(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        stab_rule = next(r for r in verdict.rule_results if r.rule_id == "parameter_stability")
        assert stab_rule.status == "FAIL"
        assert "1/5" in stab_rule.message  # only 1 of 5 sweeps stable

    def test_failed_reasons_non_empty(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        assert len(verdict.reasons) > 0
        assert all(isinstance(r, str) and len(r) > 0 for r in verdict.reasons)


# ══════════════════════════════════════════════════════════════════════════════
# Promotion gate — determinism
# ══════════════════════════════════════════════════════════════════════════════


class TestPromotionGateDeterminism:

    def test_deterministic_evaluation(self, tmp_path: Path):
        """Same inputs produce identical rule results (excluding timestamp)."""
        campaign = _make_passing_campaign(tmp_path)
        v1 = evaluate_promotion_gate(campaign)
        v2 = evaluate_promotion_gate(campaign)
        assert v1.overall_verdict == v2.overall_verdict
        assert v1.passed_rules_count == v2.passed_rules_count
        assert v1.failed_rules_count == v2.failed_rules_count
        for r1, r2 in zip(v1.rule_results, v2.rule_results):
            assert r1.rule_id == r2.rule_id
            assert r1.status == r2.status
            assert r1.observed_value == r2.observed_value
            assert r1.threshold == r2.threshold
            assert r1.message == r2.message

    def test_reason_strings_stable(self, tmp_path: Path):
        """Failed reason strings are deterministic across evaluations."""
        campaign = _make_failing_campaign(tmp_path)
        v1 = evaluate_promotion_gate(campaign)
        v2 = evaluate_promotion_gate(campaign)
        assert v1.reasons == v2.reasons


# ══════════════════════════════════════════════════════════════════════════════
# Promotion gate — configurable thresholds
# ══════════════════════════════════════════════════════════════════════════════


class TestPromotionGateConfig:

    def test_custom_thresholds_change_result(self, tmp_path: Path):
        """Tightening thresholds can flip a PASS to FAIL."""
        campaign = _make_passing_campaign(tmp_path)
        # Default should pass
        assert evaluate_promotion_gate(campaign).overall_verdict == "PASS"
        # Impossibly tight Sharpe should fail
        tight = {"min_sharpe": 10.0}
        assert evaluate_promotion_gate(campaign, gate_config=tight).overall_verdict == "FAIL"

    def test_relaxed_thresholds_can_pass_failure(self, tmp_path: Path):
        """Relaxing thresholds can flip a FAIL to PASS."""
        campaign = _make_failing_campaign(tmp_path)
        relaxed = {
            "min_net_pnl": -1000.0,
            "min_sharpe": -1.0,
            "max_drawdown_pct": 99.0,
            "max_cost_stress_net_pnl_drop_pct": 999.0,
            "max_latency_net_pnl_drop_pct": 999.0,
            "min_stable_sweep_fraction": 0.0,
        }
        assert evaluate_promotion_gate(campaign, gate_config=relaxed).overall_verdict == "PASS"


# ══════════════════════════════════════════════════════════════════════════════
# Promotion gate — SKIP handling
# ══════════════════════════════════════════════════════════════════════════════


class TestPromotionGateSkip:

    def test_missing_metrics_skipped(self, tmp_path: Path):
        """Rules SKIP gracefully when metrics are unavailable."""
        campaign = NormalizedCampaign(
            campaign_id="empty",
            campaign_dir=tmp_path,
            baseline_config="x",
            baseline_run_id=None,
            description="",
        )
        verdict = evaluate_promotion_gate(campaign)
        # All rules should be SKIP (no baseline metrics, no scenarios)
        assert verdict.skipped_rules_count == len(verdict.rule_results)
        assert verdict.overall_verdict == "PASS"  # no failures = pass (with caveats)
        assert "Skipped rules" in verdict.confidence_note


# ══════════════════════════════════════════════════════════════════════════════
# Report generation
# ══════════════════════════════════════════════════════════════════════════════


class TestReportGeneration:

    def test_report_contains_sections(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        report = generate_robustness_report(campaign, verdict)

        assert "# Robustness Report" in report
        assert "## Baseline Performance" in report
        assert "## Cost Sensitivity" in report
        assert "## Latency Sensitivity" in report
        assert "## Parameter Stability" in report
        assert "## Regime and Session Analysis" in report
        assert "## Bottom Line" in report
        assert "## Promotion Verdict" in report

    def test_report_contains_verdict(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        report = generate_robustness_report(campaign, verdict)
        assert "PASS" in report

    def test_report_fail_case(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        report = generate_robustness_report(campaign, verdict)
        assert "FAIL" in report
        assert "unprofitable" in report.lower() or "net PnL" in report

    def test_report_written_to_file(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        generate_robustness_report(campaign, verdict, output_dir=tmp_path)
        assert (tmp_path / "robustness_report.md").exists()
        content = (tmp_path / "robustness_report.md").read_text(encoding="utf-8")
        assert "# Robustness Report" in content

    def test_report_deterministic(self, tmp_path: Path):
        """Same inputs produce identical report text (excluding timestamps)."""
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        r1 = generate_robustness_report(campaign, verdict)
        r2 = generate_robustness_report(campaign, verdict)
        assert r1 == r2

    def test_report_works_and_fails_sections(self, tmp_path: Path):
        campaign = _make_failing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        report = generate_robustness_report(campaign, verdict)
        assert "### Where it works" in report
        assert "### Where it fails" in report
        assert "### Assumptions and caveats" in report


# ══════════════════════════════════════════════════════════════════════════════
# Artifact writing
# ══════════════════════════════════════════════════════════════════════════════


class TestArtifactWriting:

    def test_promotion_artifacts_written(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        json_path, md_path = write_promotion_artifacts(verdict, tmp_path)

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.name == "promotion_gate.json"
        assert md_path.name == "promotion_gate.md"

    def test_promotion_json_structure(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        write_promotion_artifacts(verdict, tmp_path)

        data = json.loads((tmp_path / "promotion_gate.json").read_text(encoding="utf-8"))
        assert data["candidate_id"] == "test_pass"
        assert data["overall_verdict"] == "PASS"
        assert "rule_results" in data
        assert "reasons" in data
        assert "generated_at" in data

    def test_promotion_md_contains_table(self, tmp_path: Path):
        campaign = _make_passing_campaign(tmp_path)
        verdict = evaluate_promotion_gate(campaign)
        write_promotion_artifacts(verdict, tmp_path)

        md = (tmp_path / "promotion_gate.md").read_text(encoding="utf-8")
        assert "## Rule Results" in md
        assert "| Rule | Status |" in md


# ══════════════════════════════════════════════════════════════════════════════
# Finalization integration
# ══════════════════════════════════════════════════════════════════════════════


class TestFinalization:

    def test_finalize_writes_all_artifacts(self, tmp_path: Path):
        """finalize_campaign writes report + gate artifacts."""
        # Set up a minimal campaign directory
        manifest = {
            "campaign_id": "finalize_test",
            "baseline_config": "configs/test.yaml",
            "baseline_run_id": None,
            "description": "Finalization test",
            "scenarios": {},
        }
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

        from src.robustness.campaign import finalize_campaign

        finalize_campaign(tmp_path)

        assert (tmp_path / "robustness_report.md").exists()
        assert (tmp_path / "promotion_gate.json").exists()
        assert (tmp_path / "promotion_gate.md").exists()

    def test_finalize_with_custom_gate_config(self, tmp_path: Path):
        """Custom gate config is passed through to evaluation."""
        manifest = {
            "campaign_id": "custom_gate",
            "baseline_config": "configs/test.yaml",
            "baseline_run_id": None,
            "description": "",
            "scenarios": {},
        }
        (tmp_path / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")

        from src.robustness.campaign import finalize_campaign

        # This should not raise
        finalize_campaign(tmp_path, gate_config={"min_net_pnl": -9999.0})
        data = json.loads((tmp_path / "promotion_gate.json").read_text(encoding="utf-8"))
        assert data["overall_verdict"] in ("PASS", "FAIL")
