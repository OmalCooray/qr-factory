"""Normalized campaign data loader for report generation and promotion gating.

Reads campaign artifacts (MANIFEST.json, campaign_summary.json, scenario_metrics.csv,
regime/session slice metrics) into a single normalized structure consumed by
the report generator and promotion gate.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ScenarioResult:
    """Normalized metrics for a single scenario."""

    scenario_id: str
    family: str
    name: str
    description: str
    status: str
    run_id: str | None
    metrics: dict[str, Any] = field(default_factory=dict)
    regime_metrics: dict[str, Any] = field(default_factory=dict)
    session_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedCampaign:
    """All campaign data needed for reporting and promotion gating."""

    campaign_id: str
    campaign_dir: Path
    baseline_config: str
    baseline_run_id: str | None
    description: str

    # Baseline metrics (from the baseline run or summary)
    baseline_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_regime_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_session_metrics: dict[str, Any] = field(default_factory=dict)

    # Scenario results grouped by family
    cost_stress: list[ScenarioResult] = field(default_factory=list)
    latency_proxy: list[ScenarioResult] = field(default_factory=list)
    parameter_sweep: list[ScenarioResult] = field(default_factory=list)
    other_scenarios: list[ScenarioResult] = field(default_factory=list)

    # Raw scenario table (for convenience)
    scenario_table: pd.DataFrame | None = None

    @property
    def all_scenarios(self) -> list[ScenarioResult]:
        return self.cost_stress + self.latency_proxy + self.parameter_sweep + self.other_scenarios

    @property
    def n_completed(self) -> int:
        return sum(1 for s in self.all_scenarios if s.status == "completed")

    @property
    def n_failed(self) -> int:
        return sum(1 for s in self.all_scenarios if s.status == "failed")


def _load_run_metrics(run_dir: Path) -> dict[str, Any]:
    """Load key metrics from a completed run directory."""
    eval_path = run_dir / "evaluation.json"
    metrics_path = run_dir / "metrics.json"

    if eval_path.exists():
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        return evaluation.get("trading_metrics", {})
    elif metrics_path.exists():
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {
            "total_net_pnl": raw.get("total_net_pnl", raw.get("total_pnl")),
            "total_gross_pnl": raw.get("total_gross_pnl"),
            "total_cost_pnl": raw.get("total_cost_pnl"),
            "num_trades": raw.get("n_trades"),
            "win_rate": raw.get("win_rate"),
            "max_drawdown_pct": raw.get("max_drawdown_pct"),
            "sharpe_per_bar": raw.get("sharpe_per_bar"),
        }
    return {}


def _load_regime_slice(run_dir: Path) -> dict[str, Any]:
    """Load regime_slice_metrics.json if it exists."""
    path = run_dir / "regime_slice_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _load_session_slice(run_dir: Path) -> dict[str, Any]:
    """Load session_slice_metrics.json if it exists."""
    path = run_dir / "session_slice_metrics.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_campaign(campaign_dir: str | Path) -> NormalizedCampaign:
    """Load a completed campaign into a normalized structure.

    Parameters
    ----------
    campaign_dir : str | Path
        Path to the campaign output directory containing MANIFEST.json.

    Returns
    -------
    NormalizedCampaign
        Normalized campaign data ready for report generation and promotion gating.

    Raises
    ------
    FileNotFoundError
        If MANIFEST.json is missing.
    """
    campaign_dir = Path(campaign_dir)
    manifest_path = campaign_dir / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"MANIFEST.json not found in {campaign_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs_dir = _REPO_ROOT / "runs"

    campaign = NormalizedCampaign(
        campaign_id=manifest.get("campaign_id", "unknown"),
        campaign_dir=campaign_dir,
        baseline_config=manifest.get("baseline_config", ""),
        baseline_run_id=manifest.get("baseline_run_id"),
        description=manifest.get("description", ""),
    )

    # Load baseline metrics if baseline run exists
    baseline_rid = manifest.get("baseline_run_id")
    if baseline_rid:
        baseline_dir = runs_dir / baseline_rid
        if baseline_dir.exists():
            campaign.baseline_metrics = _load_run_metrics(baseline_dir)
            campaign.baseline_regime_metrics = _load_regime_slice(baseline_dir)
            campaign.baseline_session_metrics = _load_session_slice(baseline_dir)

    # Load scenario results
    for sid, entry in manifest.get("scenarios", {}).items():
        family = entry.get("family", "unknown")
        status = entry.get("status", "pending")
        run_id = entry.get("run_id")

        scenario = ScenarioResult(
            scenario_id=sid,
            family=family,
            name=entry.get("name", ""),
            description=entry.get("description", ""),
            status=status,
            run_id=run_id,
        )

        # Load metrics for completed runs
        if run_id and status == "completed":
            run_path = runs_dir / run_id
            if run_path.exists():
                scenario.metrics = _load_run_metrics(run_path)
                scenario.regime_metrics = _load_regime_slice(run_path)
                scenario.session_metrics = _load_session_slice(run_path)

        if family == "cost_stress":
            campaign.cost_stress.append(scenario)
        elif family == "latency_proxy":
            campaign.latency_proxy.append(scenario)
        elif family == "parameter_sweep":
            campaign.parameter_sweep.append(scenario)
        else:
            campaign.other_scenarios.append(scenario)

    # Load scenario metrics CSV if available
    csv_path = campaign_dir / "scenario_metrics.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            campaign.scenario_table = pd.read_csv(csv_path)
        except Exception:
            log.warning("Could not load scenario_metrics.csv", exc_info=True)

    return campaign
