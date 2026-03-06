"""CLI entry point for the evaluation package.

Usage:
    python -m src.eval.cli evaluate-run --run_dir runs/<run_id>
    python -m src.eval.cli compare-runs --run_dirs runs/A runs/B --out_dir runs/compare/X
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _cmd_evaluate_run(args: argparse.Namespace) -> None:
    from .aggregator import evaluate_run
    from .artifacts import load_run_artifacts
    from .report import generate_report

    run_dir = Path(args.run_dir)
    if not (run_dir / "equity.csv").exists():
        log.error("equity.csv not found in %s", run_dir)
        sys.exit(1)

    result = evaluate_run(run_dir)
    artifacts = load_run_artifacts(run_dir)

    generate_report(
        run_dir=run_dir,
        evaluation_dict=asdict(result),
        equity_df=artifacts.equity,
        equity_gross_df=artifacts.equity_gross,
    )

    log.info("Evaluation complete: %s", run_dir)
    log.info("  evaluation.json, report.md, plots/drawdown.png")


def _cmd_compare_runs(args: argparse.Namespace) -> None:
    from .comparison import compare_runs

    run_dirs = [Path(d) for d in args.run_dirs]

    # Validate all runs have equity.csv
    for rd in run_dirs:
        if not (rd / "equity.csv").exists():
            log.error("equity.csv not found in %s", rd)
            sys.exit(1)

    out_dir = Path(args.out_dir)
    compare_runs(run_dirs, out_dir)

    log.info("Comparison complete: %s", out_dir)
    log.info("  comparison.csv, ranking.csv, comparison.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="src.eval.cli",
        description="Evaluation suite for qr-factory backtest runs",
    )
    sub = parser.add_subparsers(dest="command")

    # evaluate-run
    p_eval = sub.add_parser("evaluate-run", help="Evaluate a single run")
    p_eval.add_argument("--run_dir", required=True, help="Path to run directory")

    # compare-runs
    p_cmp = sub.add_parser("compare-runs", help="Compare multiple runs")
    p_cmp.add_argument("--run_dirs", nargs="+", required=True, help="Paths to run directories")
    p_cmp.add_argument("--out_dir", required=True, help="Output directory for comparison")

    args = parser.parse_args()

    if args.command == "evaluate-run":
        _cmd_evaluate_run(args)
    elif args.command == "compare-runs":
        _cmd_compare_runs(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
