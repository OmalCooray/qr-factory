"""Run paired comparison: fixed-threshold vs probability-sized policy.

Orchestrates:
  1. Two backtests (w7_policy_fixed, w7_policy_probsized)
  2. Per-run evaluation + report generation
  3. MLflow logging (if available)
  4. Named comparison: fixed_vs_probsized
  5. Summary table + MANIFEST.json

Usage:
    uv run python scripts/run_policy_comparison.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.engine.runner import run_backtest
from src.eval.aggregator import evaluate_run
from src.eval.artifacts import load_run_artifacts
from src.eval.comparison import compare_runs
from src.eval.report import generate_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CONFIGS = [
    ("fixed", "configs/w7_policy_fixed.yaml"),
    ("probsized", "configs/w7_policy_probsized.yaml"),
]


def _try_mlflow_log(run_dir: Path, evaluation_dict: dict, calibration: dict | None) -> str | None:
    """Attempt MLflow logging; return run ID or None."""
    try:
        from src.eval.mlflow_logger import log_run_to_mlflow
        return log_run_to_mlflow(run_dir, evaluation_dict, calibration)
    except Exception:
        log.warning("MLflow logging failed (non-fatal)", exc_info=True)
        return None


def main() -> None:
    runs_dir = _REPO_ROOT / "runs"
    comparisons_dir = runs_dir / "comparisons"

    # ── 1. Run backtests ──
    results: dict[str, dict] = {}
    for label, config_path in CONFIGS:
        log.info("=" * 60)
        log.info("Running: %s (%s)", label, config_path)
        log.info("=" * 60)

        try:
            run_id = run_backtest(config_path)
        except Exception:
            log.exception("FAILED: %s", label)
            continue

        run_dir = runs_dir / run_id
        results[label] = {"run_id": run_id, "run_dir": str(run_dir)}
        log.info("Completed: %s -> %s", label, run_id)

    if len(results) < 2:
        log.error("Need both runs for comparison. Got %d.", len(results))
        sys.exit(1)

    # ── 2. Evaluate each run + MLflow ──
    eval_data: dict[str, dict] = {}
    for label, info in results.items():
        run_dir = Path(info["run_dir"])
        log.info("Evaluating: %s (%s)", label, info["run_id"])
        result = evaluate_run(run_dir)
        artifacts = load_run_artifacts(run_dir)
        generate_report(
            run_dir=run_dir,
            evaluation_dict=asdict(result),
            equity_df=artifacts.equity,
            equity_gross_df=artifacts.equity_gross,
        )
        result_dict = asdict(result)
        eval_data[label] = result_dict

        # MLflow logging
        cal_path = run_dir / "calibration.json"
        cal = json.loads(cal_path.read_text(encoding="utf-8")) if cal_path.exists() else None
        mlf_id = _try_mlflow_log(run_dir, result_dict, cal)
        if mlf_id:
            log.info("  MLflow run: %s", mlf_id)

    # ── 3. Comparison ──
    comp_name = "fixed_vs_probsized"
    pair_dirs = [Path(results[l]["run_dir"]) for l in ("fixed", "probsized")]
    out_dir = comparisons_dir / comp_name
    log.info("Comparing: %s", comp_name)
    compare_runs(pair_dirs, out_dir)

    # ── 4. Summary table ──
    summary_rows = []
    for label, info in results.items():
        ev = eval_data.get(label, {})
        trade = ev.get("trading_metrics", {})
        pred = ev.get("prediction_metrics", {})
        summary_rows.append({
            "variant": label,
            "run_id": info["run_id"],
            "net_pnl": trade.get("total_net_pnl", "N/A"),
            "sharpe": trade.get("sharpe_per_bar", "N/A"),
            "max_dd": trade.get("max_drawdown_pct", "N/A"),
            "n_trades": trade.get("n_trades", "N/A"),
            "avg_position_size": trade.get("avg_position_size", "N/A"),
        })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print("=" * 70)
    print("POLICY COMPARISON SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()

    # ── 5. MANIFEST.json ──
    manifest = {
        "runs": results,
        "comparisons": {
            comp_name: {
                "runs": ["fixed", "probsized"],
                "path": str(out_dir),
            },
        },
    }

    comparisons_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = comparisons_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote MANIFEST.json -> %s", manifest_path)

    log.info("Done. Check runs/comparisons/%s/ for comparison artifacts.", comp_name)


if __name__ == "__main__":
    main()
