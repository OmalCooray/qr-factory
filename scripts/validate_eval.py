"""Independent validation of the eval suite against real backtest runs.

Recomputes all metrics from raw artifacts and compares against evaluation.json.
Produces:
  - VALIDATION_REPORT.md      — human-readable pass/fail summary
  - validation_findings.json   — machine-readable findings list
  - validation_recomputed_metrics.csv — side-by-side comparison

Usage:
    uv run python scripts/validate_eval.py
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.eval.aggregator import evaluate_run
from src.eval.artifacts import load_run_artifacts
from src.eval.comparison import compare_runs
from src.eval.prediction_metrics import compute_prediction_metrics
from src.eval.report import generate_report
from src.eval.trading_metrics import compute_trading_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RUNS_DIR = PROJECT_ROOT / "runs"
TOL = 1e-10


def _find_run_dirs() -> list[Path]:
    """Find all valid run directories (must have equity.csv + metrics.json)."""
    dirs = sorted(RUNS_DIR.iterdir())
    return [d for d in dirs if d.is_dir() and (d / "equity.csv").exists() and (d / "metrics.json").exists()]


def _recompute_metrics(run_dir: Path) -> dict:
    """Independently recompute all metrics from raw artifacts."""
    artifacts = load_run_artifacts(run_dir)

    pred_metrics = compute_prediction_metrics(artifacts.predictions)
    trade_metrics = compute_trading_metrics(
        equity=artifacts.equity,
        trades=artifacts.trades,
        starting_capital=artifacts.starting_capital,
        equity_gross=artifacts.equity_gross,
    )
    return {"prediction_metrics": pred_metrics, "trading_metrics": trade_metrics}


def _compare_dicts(
    reported: dict, recomputed: dict, prefix: str, tol: float,
) -> list[dict]:
    """Compare two flat dicts, return list of finding dicts."""
    findings = []
    all_keys = set(reported.keys()) | set(recomputed.keys())
    for k in sorted(all_keys):
        rv = reported.get(k)
        cv = recomputed.get(k)
        if rv is None and cv is None:
            continue
        if rv is None or cv is None:
            findings.append({
                "metric": f"{prefix}.{k}",
                "reported": rv,
                "recomputed": cv,
                "delta": None,
                "status": "MISSING",
            })
            continue
        if isinstance(rv, (int, float)) and isinstance(cv, (int, float)):
            if np.isinf(rv) and np.isinf(cv) and np.sign(rv) == np.sign(cv):
                findings.append({
                    "metric": f"{prefix}.{k}",
                    "reported": rv,
                    "recomputed": cv,
                    "delta": 0.0,
                    "status": "PASS",
                })
            elif abs(rv - cv) <= tol:
                findings.append({
                    "metric": f"{prefix}.{k}",
                    "reported": rv,
                    "recomputed": cv,
                    "delta": abs(rv - cv),
                    "status": "PASS",
                })
            else:
                findings.append({
                    "metric": f"{prefix}.{k}",
                    "reported": rv,
                    "recomputed": cv,
                    "delta": abs(rv - cv),
                    "status": "FAIL",
                })
        elif rv != cv:
            findings.append({
                "metric": f"{prefix}.{k}",
                "reported": rv,
                "recomputed": cv,
                "delta": None,
                "status": "MISMATCH",
            })
        else:
            findings.append({
                "metric": f"{prefix}.{k}",
                "reported": rv,
                "recomputed": cv,
                "delta": 0,
                "status": "PASS",
            })
    return findings


def _check_cost_invariant(trades_df: pd.DataFrame) -> list[dict]:
    """Verify per-row: gross_pnl - cost_pnl == pnl (net)."""
    findings = []
    if trades_df.empty:
        findings.append({
            "check": "cost_invariant",
            "status": "SKIP",
            "detail": "No trades to validate.",
        })
        return findings

    required = {"pnl", "gross_pnl", "cost_pnl"}
    if not required.issubset(trades_df.columns):
        findings.append({
            "check": "cost_invariant",
            "status": "SKIP",
            "detail": f"Missing columns: {required - set(trades_df.columns)}",
        })
        return findings

    diff = (trades_df["gross_pnl"] - trades_df["cost_pnl"]) - trades_df["pnl"]
    max_err = float(diff.abs().max())
    n_violations = int((diff.abs() > TOL).sum())

    if n_violations == 0:
        findings.append({
            "check": "cost_invariant",
            "status": "PASS",
            "detail": f"All {len(trades_df)} rows satisfy gross - cost = net (max_err={max_err:.2e}).",
        })
    else:
        findings.append({
            "check": "cost_invariant",
            "status": "FAIL",
            "detail": f"{n_violations}/{len(trades_df)} rows violate gross - cost = net (max_err={max_err:.2e}).",
        })
    return findings


def _check_dd_discrepancy(run_dir: Path, eval_dd_pct: float, metrics_json: dict) -> dict:
    """Document walk-forward DD discrepancy between eval and metrics.json."""
    metrics_dd = metrics_json.get("max_drawdown_pct") or metrics_json.get("worst_drawdown_pct", 0.0)
    is_wf = metrics_json.get("mode") == "walk_forward"
    delta = abs(eval_dd_pct - metrics_dd)

    if not is_wf:
        return {
            "check": "dd_discrepancy",
            "status": "INFO",
            "detail": f"Not walk-forward. eval_dd={eval_dd_pct:.4f}%, metrics_dd={metrics_dd:.4f}%, delta={delta:.4f}pp",
        }

    return {
        "check": "dd_discrepancy",
        "status": "DOCUMENTED" if delta > 0.01 else "PASS",
        "detail": (
            f"Walk-forward DD discrepancy (expected): "
            f"eval={eval_dd_pct:.4f}% (full curve), "
            f"metrics.json={metrics_dd:.4f}% (worst fold), "
            f"delta={delta:.4f}pp"
        ),
    }


def _check_report_sections(run_dir: Path) -> list[dict]:
    """Verify report.md has required sections."""
    report_path = run_dir / "report.md"
    findings = []
    if not report_path.exists():
        findings.append({
            "check": "report_sections",
            "status": "FAIL",
            "detail": "report.md does not exist.",
        })
        return findings

    content = report_path.read_text(encoding="utf-8")
    required_sections = [
        "## Run Summary",
        "## Costs & Friction",
        "## Prediction Metrics",
        "## Trading Metrics",
        "## Drawdown",
        "## Metric Disagreement Notes",
    ]
    for section in required_sections:
        if section in content:
            findings.append({
                "check": f"report_section:{section}",
                "status": "PASS",
                "detail": f"Found '{section}' in report.md",
            })
        else:
            findings.append({
                "check": f"report_section:{section}",
                "status": "FAIL",
                "detail": f"Missing '{section}' in report.md",
            })
    return findings


def _check_idempotency(run_dir: Path) -> dict:
    """Run evaluate_run twice and check byte-identical output."""
    evaluate_run(run_dir)
    first = (run_dir / "evaluation.json").read_bytes()
    evaluate_run(run_dir)
    second = (run_dir / "evaluation.json").read_bytes()

    if first == second:
        return {"check": "idempotency", "status": "PASS", "detail": "Byte-identical evaluation.json on re-run."}
    return {"check": "idempotency", "status": "FAIL", "detail": "evaluation.json changed between runs."}


def _check_comparison_determinism(run_dirs: list[Path], out_dir: Path) -> dict:
    """Run compare_runs twice and check deterministic output."""
    compare_runs(run_dirs, out_dir)
    first_csv = (out_dir / "comparison.csv").read_bytes()
    first_rank = (out_dir / "ranking.csv").read_bytes()

    compare_runs(run_dirs, out_dir)
    second_csv = (out_dir / "comparison.csv").read_bytes()
    second_rank = (out_dir / "ranking.csv").read_bytes()

    if first_csv == second_csv and first_rank == second_rank:
        return {"check": "comparison_determinism", "status": "PASS", "detail": "Comparison artifacts are deterministic."}
    return {"check": "comparison_determinism", "status": "FAIL", "detail": "Comparison artifacts differ between runs."}


def validate_all() -> None:
    """Main validation entry point."""
    run_dirs = _find_run_dirs()
    if not run_dirs:
        log.error("No valid run directories found in %s", RUNS_DIR)
        sys.exit(1)

    log.info("Found %d run directories to validate", len(run_dirs))

    all_findings: list[dict] = []
    metric_rows: list[dict] = []

    for run_dir in run_dirs:
        run_id = run_dir.name
        log.info("─── Validating %s ───", run_id)

        # 1. Evaluate (produces evaluation.json + report.md)
        result = evaluate_run(run_dir)
        eval_dict = asdict(result)
        artifacts = load_run_artifacts(run_dir)

        # Generate report
        generate_report(run_dir, eval_dict, artifacts.equity, artifacts.equity_gross)

        # 2. Independently recompute metrics
        recomputed = _recompute_metrics(run_dir)

        # 3. Compare prediction metrics
        pred_findings = _compare_dicts(
            eval_dict.get("prediction_metrics", {}),
            recomputed["prediction_metrics"],
            f"{run_id}/pred",
            TOL,
        )
        all_findings.extend(pred_findings)
        for f in pred_findings:
            metric_rows.append({"run_id": run_id, **f})

        # 4. Compare trading metrics
        trade_findings = _compare_dicts(
            eval_dict.get("trading_metrics", {}),
            recomputed["trading_metrics"],
            f"{run_id}/trade",
            TOL,
        )
        all_findings.extend(trade_findings)
        for f in trade_findings:
            metric_rows.append({"run_id": run_id, **f})

        # 5. Cost invariant
        cost_findings = _check_cost_invariant(artifacts.trades)
        for cf in cost_findings:
            cf["run_id"] = run_id
        all_findings.extend(cost_findings)

        # 6. DD discrepancy documentation
        eval_dd = eval_dict.get("trading_metrics", {}).get("max_dd_pct", 0.0)
        dd_finding = _check_dd_discrepancy(run_dir, eval_dd, artifacts.metrics)
        dd_finding["run_id"] = run_id
        all_findings.append(dd_finding)

        # 7. Report sections
        report_findings = _check_report_sections(run_dir)
        for rf in report_findings:
            rf["run_id"] = run_id
        all_findings.extend(report_findings)

        # 8. Idempotency
        idem = _check_idempotency(run_dir)
        idem["run_id"] = run_id
        all_findings.append(idem)

    # 9. Comparison determinism (use last two runs)
    if len(run_dirs) >= 2:
        compare_out = RUNS_DIR / "_validation_compare"
        compare_out.mkdir(exist_ok=True)
        comp_det = _check_comparison_determinism(run_dirs[-2:], compare_out)
        comp_det["run_id"] = "cross-run"
        all_findings.append(comp_det)

        # Clean up temp comparison dir
        import shutil
        shutil.rmtree(compare_out, ignore_errors=True)

    # ── Write outputs ──
    _write_findings_json(all_findings)
    _write_recomputed_csv(metric_rows)
    _write_validation_report(all_findings, run_dirs)

    # Summary
    n_pass = sum(1 for f in all_findings if f.get("status") == "PASS")
    n_fail = sum(1 for f in all_findings if f.get("status") == "FAIL")
    n_doc = sum(1 for f in all_findings if f.get("status") in ("DOCUMENTED", "INFO", "SKIP", "MISSING"))
    log.info("══════════════════════════════════════════")
    log.info("  PASS: %d  |  FAIL: %d  |  INFO/SKIP: %d", n_pass, n_fail, n_doc)
    log.info("══════════════════════════════════════════")

    if n_fail > 0:
        log.error("VALIDATION FAILED — see VALIDATION_REPORT.md for details")
        sys.exit(1)
    else:
        log.info("VALIDATION PASSED")


def _write_findings_json(findings: list[dict]) -> None:
    path = PROJECT_ROOT / "validation_findings.json"
    path.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %s (%d findings)", path.name, len(findings))


def _write_recomputed_csv(rows: list[dict]) -> None:
    if not rows:
        return
    path = PROJECT_ROOT / "validation_recomputed_metrics.csv"
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    log.info("Wrote %s (%d rows)", path.name, len(df))


def _write_validation_report(findings: list[dict], run_dirs: list[Path]) -> None:
    lines: list[str] = []
    lines.append("# Evaluation Suite Validation Report")
    lines.append("")
    lines.append(f"**Runs validated:** {len(run_dirs)}")
    lines.append("")

    n_pass = sum(1 for f in findings if f.get("status") == "PASS")
    n_fail = sum(1 for f in findings if f.get("status") == "FAIL")
    n_other = len(findings) - n_pass - n_fail
    lines.append(f"**Results:** {n_pass} PASS, {n_fail} FAIL, {n_other} INFO/SKIP/DOCUMENTED")
    lines.append("")

    overall = "PASS" if n_fail == 0 else "FAIL"
    lines.append(f"## Overall: **{overall}**")
    lines.append("")

    # Group findings by run_id
    by_run: dict[str, list[dict]] = {}
    for f in findings:
        rid = f.get("run_id", "unknown")
        by_run.setdefault(rid, []).append(f)

    for rid in sorted(by_run.keys()):
        lines.append(f"### {rid}")
        lines.append("")
        run_findings = by_run[rid]
        run_pass = sum(1 for f in run_findings if f.get("status") == "PASS")
        run_fail = sum(1 for f in run_findings if f.get("status") == "FAIL")
        lines.append(f"- {run_pass} PASS, {run_fail} FAIL")
        lines.append("")

        # Show non-PASS findings in detail
        non_pass = [f for f in run_findings if f.get("status") != "PASS"]
        if non_pass:
            lines.append("| Check | Status | Detail |")
            lines.append("|-------|--------|--------|")
            for f in non_pass:
                check = f.get("check") or f.get("metric", "?")
                status = f.get("status", "?")
                detail = f.get("detail", "")
                if "delta" in f and f["delta"] is not None:
                    detail = f"reported={f.get('reported')}, recomputed={f.get('recomputed')}, delta={f['delta']}"
                lines.append(f"| {check} | {status} | {detail} |")
            lines.append("")

    lines.append("---")
    lines.append("*Generated by `scripts/validate_eval.py`*")
    lines.append("")

    path = PROJECT_ROOT / "VALIDATION_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Wrote %s", path.name)


if __name__ == "__main__":
    validate_all()
