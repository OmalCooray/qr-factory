"""Machine-readable campaign summary generation.

Reads manifest + per-run evaluation artifacts and produces:
- ``campaign_summary.json`` — structured overview with per-scenario key metrics
- ``scenario_metrics.csv`` — flat table (one row per scenario) for easy analysis
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Key trading metrics to extract per scenario
_KEY_METRICS = [
    "total_net_pnl",
    "total_gross_pnl",
    "total_cost_pnl",
    "sharpe_per_bar",
    "max_drawdown_pct",
    "num_trades",
    "win_rate",
    "avg_trade_pnl",
    "profit_factor",
    "trades_per_day",
]


def _load_scenario_metrics(run_dir: Path) -> dict[str, Any]:
    """Load key metrics from a completed run's evaluation.json or metrics.json."""
    # Prefer evaluation.json (richer), fall back to metrics.json
    eval_path = run_dir / "evaluation.json"
    metrics_path = run_dir / "metrics.json"

    trading: dict[str, Any] = {}
    prediction: dict[str, Any] = {}

    if eval_path.exists():
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        trading = evaluation.get("trading_metrics", {})
        prediction = evaluation.get("prediction_metrics", {})
    elif metrics_path.exists():
        raw = json.loads(metrics_path.read_text(encoding="utf-8"))
        trading = {
            "total_net_pnl": raw.get("total_net_pnl", raw.get("total_pnl")),
            "total_gross_pnl": raw.get("total_gross_pnl"),
            "total_cost_pnl": raw.get("total_cost_pnl"),
            "num_trades": raw.get("n_trades"),
            "win_rate": raw.get("win_rate"),
            "max_drawdown_pct": raw.get("max_drawdown_pct"),
        }

    result: dict[str, Any] = {}
    for key in _KEY_METRICS:
        val = trading.get(key)
        if val is not None and isinstance(val, (int, float)):
            result[key] = val
    # Include prediction metrics if available
    for key in ("brier", "roc_auc", "accuracy"):
        val = prediction.get(key)
        if val is not None and isinstance(val, (int, float)):
            result[f"pred_{key}"] = val

    return result


def build_campaign_summary(campaign_dir: Path) -> Path:
    """Build ``campaign_summary.json`` from the manifest and run artifacts.

    Parameters
    ----------
    campaign_dir : Path
        Campaign output directory containing MANIFEST.json.

    Returns
    -------
    Path
        Path to the written campaign_summary.json.
    """
    manifest_path = campaign_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs_dir = _REPO_ROOT / "runs"

    scenarios_data = manifest.get("scenarios", {})

    # Aggregate by family
    family_counts: dict[str, dict[str, int]] = {}
    scenario_results: dict[str, dict[str, Any]] = {}

    for sid, entry in scenarios_data.items():
        family = entry.get("family", "unknown")
        status = entry.get("status", "pending")

        if family not in family_counts:
            family_counts[family] = {"n_scenarios": 0, "completed": 0, "failed": 0}
        family_counts[family]["n_scenarios"] += 1
        if status == "completed":
            family_counts[family]["completed"] += 1
        elif status == "failed":
            family_counts[family]["failed"] += 1

        scenario_entry: dict[str, Any] = {
            "family": family,
            "name": entry.get("name", ""),
            "description": entry.get("description", ""),
            "status": status,
            "run_id": entry.get("run_id"),
            "key_metrics": {},
        }

        # Load metrics for completed runs
        run_id = entry.get("run_id")
        if run_id and status == "completed":
            run_path = runs_dir / run_id
            if run_path.exists():
                scenario_entry["key_metrics"] = _load_scenario_metrics(run_path)

        scenario_results[sid] = scenario_entry

    n_total = len(scenarios_data)
    n_completed = sum(
        1 for e in scenarios_data.values() if e.get("status") == "completed"
    )
    n_failed = sum(
        1 for e in scenarios_data.values() if e.get("status") == "failed"
    )

    summary = {
        "campaign_id": manifest.get("campaign_id"),
        "baseline_config": manifest.get("baseline_config"),
        "baseline_run_id": manifest.get("baseline_run_id"),
        "description": manifest.get("description", ""),
        "dry_run": manifest.get("dry_run", False),
        "n_scenarios": n_total,
        "n_completed": n_completed,
        "n_failed": n_failed,
        "families": family_counts,
        "scenario_results": scenario_results,
    }

    out_path = campaign_dir / "campaign_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_path)
    return out_path


def build_scenario_metrics_table(campaign_dir: Path) -> Path:
    """Build ``scenario_metrics.csv`` — one row per completed scenario.

    Parameters
    ----------
    campaign_dir : Path
        Campaign output directory containing MANIFEST.json.

    Returns
    -------
    Path
        Path to the written CSV file.
    """
    manifest_path = campaign_dir / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    runs_dir = _REPO_ROOT / "runs"

    rows: list[dict[str, Any]] = []
    for sid, entry in manifest.get("scenarios", {}).items():
        run_id = entry.get("run_id")
        if entry.get("status") != "completed" or not run_id:
            continue

        run_path = runs_dir / run_id
        metrics = _load_scenario_metrics(run_path) if run_path.exists() else {}

        row: dict[str, Any] = {
            "scenario_id": sid,
            "family": entry.get("family", ""),
            "name": entry.get("name", ""),
            "run_id": run_id,
        }
        row.update(metrics)
        rows.append(row)

    out_path = campaign_dir / "scenario_metrics.csv"
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
    else:
        out_path.touch()
    log.info("Wrote %s (%d rows)", out_path, len(rows))
    return out_path
