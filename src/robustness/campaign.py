"""Robustness campaign orchestration."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import yaml

from src._paths import REPO_ROOT
from src.robustness.expand import build_scenario_configs, load_campaign_spec
from src.robustness.manifest import write_manifest

log = logging.getLogger(__name__)


def _try_mlflow_log(
    run_dir: Path,
    evaluation_dict: dict,
    calibration: dict | None,
    extra_tags: dict[str, str] | None = None,
) -> str | None:
    """Attempt MLflow logging with optional extra tags; return run ID or None."""
    try:
        from src.eval.mlflow_logger import log_run_to_mlflow

        return log_run_to_mlflow(
            run_dir, evaluation_dict, calibration, extra_tags=extra_tags,
        )
    except Exception:
        log.warning("MLflow logging failed (non-fatal)", exc_info=True)
        return None


def run_campaign(
    spec_path: str | Path,
    dry_run: bool = False,
) -> Path:
    """Execute a robustness campaign.

    Parameters
    ----------
    spec_path : str | Path
        Path to the campaign YAML specification.
    dry_run : bool
        If True, write scenario configs and manifest but do not run backtests.

    Returns
    -------
    Path
        Path to the campaign output directory.
    """
    from dataclasses import asdict

    from src.engine.runner import run_backtest
    from src.eval.aggregator import evaluate_run
    from src.eval.artifacts import load_run_artifacts
    from src.eval.report import generate_report

    campaign = load_campaign_spec(spec_path)
    scenario_configs = build_scenario_configs(campaign)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    campaign_dir = (
        REPO_ROOT / "runs" / "robustness" / f"{campaign.campaign_id}__{timestamp}"
    )
    scenarios_dir = campaign_dir / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Campaign: %s (%d scenarios)", campaign.campaign_id, len(scenario_configs),
    )
    log.info("Output:   %s", campaign_dir)
    log.info("Dry run:  %s", dry_run)

    # ── Initialize manifest ──
    manifest: dict = {
        "campaign_id": campaign.campaign_id,
        "timestamp": timestamp,
        "baseline_config": campaign.baseline_config,
        "baseline_run_id": campaign.baseline_run_id,
        "description": campaign.description,
        "dry_run": dry_run,
        "scenarios": {},
    }

    # ── Run baseline if needed ──
    runs_dir = REPO_ROOT / "runs"
    if campaign.baseline_run_id is None and not dry_run:
        log.info("Running baseline: %s", campaign.baseline_config)
        try:
            baseline_run_id = run_backtest(campaign.baseline_config)
            manifest["baseline_run_id"] = baseline_run_id
            log.info("Baseline complete: %s", baseline_run_id)
        except Exception:
            log.exception("Baseline run failed")
            manifest["baseline_run_id"] = None
            manifest["baseline_status"] = "failed"

    # ── Write scenario configs and optionally run ──
    for scenario, resolved_config in scenario_configs:
        config_path = scenarios_dir / f"{scenario.scenario_id}.yaml"
        config_path.write_text(
            yaml.dump(resolved_config, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

        scenario_entry: dict = {
            "family": scenario.family,
            "name": scenario.name,
            "description": scenario.description,
            "tags": list(scenario.tags),
            "config_path": str(config_path.relative_to(REPO_ROOT)),
            "status": "pending",
            "run_id": None,
        }

        if not dry_run:
            log.info("Running scenario: %s", scenario.scenario_id)
            try:
                run_id = run_backtest(str(config_path))
                scenario_entry["run_id"] = run_id
                scenario_entry["status"] = "completed"
                log.info("  Completed: %s -> %s", scenario.scenario_id, run_id)

                # Evaluate + MLflow (non-fatal)
                run_dir = runs_dir / run_id
                try:
                    result = evaluate_run(run_dir)
                    artifacts = load_run_artifacts(run_dir)
                    generate_report(
                        run_dir=run_dir,
                        evaluation_dict=asdict(result),
                        equity_df=artifacts.equity,
                        equity_gross_df=artifacts.equity_gross,
                    )
                    cal_path = run_dir / "calibration.json"
                    cal = (
                        json.loads(cal_path.read_text(encoding="utf-8"))
                        if cal_path.exists()
                        else None
                    )
                    _try_mlflow_log(
                        run_dir,
                        asdict(result),
                        cal,
                        extra_tags={
                            "robustness": "true",
                            "campaign_id": campaign.campaign_id,
                            "scenario_id": scenario.scenario_id,
                            "scenario_family": scenario.family,
                            "baseline_config": campaign.baseline_config,
                        },
                    )
                except Exception:
                    log.warning(
                        "Eval/MLflow failed for %s (non-fatal)",
                        scenario.scenario_id,
                        exc_info=True,
                    )

            except Exception:
                log.exception("Scenario failed: %s", scenario.scenario_id)
                scenario_entry["status"] = "failed"

        manifest["scenarios"][scenario.scenario_id] = scenario_entry

    # ── Write manifest ──
    write_manifest(campaign_dir, manifest)
    log.info("Wrote MANIFEST.json -> %s", campaign_dir / "MANIFEST.json")

    # ── Post-run: regime analysis on completed scenarios ──
    if not dry_run:
        from src.robustness.regimes import analyze_run_regimes

        for sid, entry in manifest["scenarios"].items():
            if entry.get("status") == "completed" and entry.get("run_id"):
                run_dir = runs_dir / entry["run_id"]
                try:
                    analyze_run_regimes(run_dir)
                except Exception:
                    log.warning(
                        "Regime analysis failed for %s (non-fatal)",
                        sid,
                        exc_info=True,
                    )

        # Also analyze baseline if it ran as part of this campaign
        baseline_rid = manifest.get("baseline_run_id")
        if baseline_rid and (runs_dir / baseline_rid).exists():
            try:
                analyze_run_regimes(runs_dir / baseline_rid)
            except Exception:
                log.warning("Baseline regime analysis failed (non-fatal)", exc_info=True)

    # ── Generate machine-readable summaries ──
    if not dry_run:
        from src.robustness.summarize import (
            build_campaign_summary,
            build_scenario_metrics_table,
        )

        try:
            build_campaign_summary(campaign_dir)
            build_scenario_metrics_table(campaign_dir)
        except Exception:
            log.warning("Summary generation failed (non-fatal)", exc_info=True)

    return campaign_dir


def finalize_campaign(
    campaign_dir: str | Path,
    gate_config: dict | None = None,
    tag_mlflow: bool = False,
) -> Path:
    """Finalize a completed campaign: generate report + promotion gate verdict.

    Reads existing campaign artifacts, evaluates promotion rules, and writes:
    - ``robustness_report.md``
    - ``promotion_gate.json``
    - ``promotion_gate.md``

    Parameters
    ----------
    campaign_dir : str | Path
        Path to the campaign output directory containing MANIFEST.json.
    gate_config : dict | None
        Optional promotion gate thresholds (overrides defaults).
    tag_mlflow : bool
        If True, tag the baseline MLflow run with promotion result.

    Returns
    -------
    Path
        Path to the campaign directory (same as input).
    """
    from src.robustness.load import load_campaign
    from src.robustness.promotion import (
        evaluate_promotion_gate,
        write_promotion_artifacts,
    )
    from src.robustness.report import generate_robustness_report

    campaign_dir = Path(campaign_dir)
    log.info("Finalizing campaign: %s", campaign_dir)

    # Load normalized campaign data
    campaign = load_campaign(campaign_dir)
    log.info(
        "Loaded: %s (%d scenarios, %d completed)",
        campaign.campaign_id,
        len(campaign.all_scenarios),
        campaign.n_completed,
    )

    # Evaluate promotion gate
    verdict = evaluate_promotion_gate(campaign, gate_config)
    log.info("Promotion verdict: %s", verdict.overall_verdict)

    # Write promotion artifacts
    write_promotion_artifacts(verdict, campaign_dir)

    # Generate robustness report
    generate_robustness_report(campaign, verdict, output_dir=campaign_dir)

    # Optional: tag MLflow
    if tag_mlflow:
        _tag_mlflow_promotion(campaign, verdict)

    return campaign_dir


def _tag_mlflow_promotion(
    campaign: "NormalizedCampaign",  # noqa: F821
    verdict: "PromotionVerdict",  # noqa: F821
) -> None:
    """Tag the campaign's MLflow runs with the promotion result."""
    try:
        import mlflow

        tracking_uri = "sqlite:///" + str(Path.cwd() / "mlflow.db").replace("\\", "/")
        mlflow.set_tracking_uri(tracking_uri)

        failed_ids = ",".join(
            r.rule_id for r in verdict.rule_results if r.status == "FAIL"
        )

        # Search for runs tagged with this campaign
        experiment = mlflow.get_experiment_by_name("qr-factory")
        if experiment is None:
            log.warning("MLflow experiment 'qr-factory' not found — skipping tagging")
            return

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.campaign_id = '{verdict.candidate_id}'",
            output_format="list",
        )

        for run in runs:
            mlflow.set_tag(run.info.run_id, "promotion_result", verdict.overall_verdict)
            if failed_ids:
                mlflow.set_tag(run.info.run_id, "promotion_failed_rules", failed_ids)
            mlflow.set_tag(run.info.run_id, "robustness_report_generated", "true")

        log.info("Tagged %d MLflow runs with promotion result", len(runs))
    except ImportError:
        log.warning("mlflow not installed — skipping promotion tagging")
    except Exception:
        log.warning("MLflow promotion tagging failed (non-fatal)", exc_info=True)
