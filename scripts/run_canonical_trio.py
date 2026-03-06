"""Run the canonical Week 4/5 trio: rule baseline, logistic baseline, LightGBM ablation.

Orchestrates:
  1. Three backtests (w4_rule_baseline, w4_logistic_baseline, w4_lgbm_ablation)
  2. Per-run evaluation + report generation
  3. Two named comparisons: rule_vs_logistic, logistic_vs_lgbm
  4. Summary table + MANIFEST.json

Usage:
    uv run python scripts/run_canonical_trio.py
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
    ("rule", "configs/w4_rule_baseline.yaml"),
    ("logistic", "configs/w4_logistic_baseline.yaml"),
    ("lgbm", "configs/w4_lgbm_ablation.yaml"),
]


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
        log.info("Completed: %s → %s", label, run_id)

    if len(results) < 2:
        log.error("Need at least 2 successful runs for comparison. Got %d.", len(results))
        sys.exit(1)

    # ── 2. Evaluate each run ──
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
        eval_data[label] = asdict(result)

    # ── 3. Generate comparisons ──
    comparison_pairs = [
        ("rule_vs_logistic", ["rule", "logistic"]),
        ("logistic_vs_lgbm", ["logistic", "lgbm"]),
    ]

    for comp_name, pair_labels in comparison_pairs:
        pair_dirs = []
        for pl in pair_labels:
            if pl not in results:
                log.warning("Skipping comparison %s — missing run: %s", comp_name, pl)
                break
            pair_dirs.append(Path(results[pl]["run_dir"]))
        else:
            out_dir = comparisons_dir / comp_name
            log.info("Comparing: %s (%s)", comp_name, " vs ".join(pair_labels))
            compare_runs(pair_dirs, out_dir)

    # ── 4. Summary table ──
    summary_rows = []
    for label, info in results.items():
        ev = eval_data.get(label, {})
        trade = ev.get("trading_metrics", {})
        pred = ev.get("prediction_metrics", {})
        summary_rows.append({
            "strategy": label,
            "run_id": info["run_id"],
            "net_pnl": trade.get("total_net_pnl", "N/A"),
            "sharpe": trade.get("sharpe_per_bar", "N/A"),
            "roc_auc": pred.get("roc_auc", "N/A"),
        })

    summary_df = pd.DataFrame(summary_rows)
    print()
    print("=" * 70)
    print("CANONICAL TRIO SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print()

    # ── 5. MANIFEST.json ──
    manifest = {
        "runs": results,
        "comparisons": {},
    }
    for comp_name, pair_labels in comparison_pairs:
        if all(pl in results for pl in pair_labels):
            manifest["comparisons"][comp_name] = {
                "runs": pair_labels,
                "path": str(comparisons_dir / comp_name),
            }

    comparisons_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = comparisons_dir / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Wrote MANIFEST.json → %s", manifest_path)

    log.info("Done. Check runs/comparisons/ for comparison artifacts.")


if __name__ == "__main__":
    main()
