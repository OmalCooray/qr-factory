"""Run or finalize a robustness campaign.

Orchestrates:
  1. Load campaign spec (baseline + scenarios)
  2. Expand grid scenarios into concrete configs
  3. Optionally run each scenario through run_backtest()
  4. Write resolved configs and MANIFEST.json
  5. (--finalize) Generate robustness report and promotion gate verdict

Usage:
    uv run python scripts/run_robustness_campaign.py configs/w8_robustness_example.yaml
    uv run python scripts/run_robustness_campaign.py configs/w8_robustness_example.yaml --dry-run
    uv run python scripts/run_robustness_campaign.py --finalize runs/robustness/w8_fixed__20260308_120000
    uv run python scripts/run_robustness_campaign.py --finalize runs/robustness/w8_fixed__20260308_120000 --tag-mlflow
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or finalize a robustness campaign",
    )
    parser.add_argument(
        "spec",
        nargs="?",
        help="Path to campaign YAML specification (for running a campaign)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write scenario configs and manifest without running backtests",
    )
    parser.add_argument(
        "--finalize",
        metavar="CAMPAIGN_DIR",
        help="Finalize an existing campaign: generate report + promotion gate",
    )
    parser.add_argument(
        "--tag-mlflow",
        action="store_true",
        help="Tag MLflow runs with promotion result (use with --finalize)",
    )
    args = parser.parse_args()

    if args.finalize:
        from src.robustness.campaign import finalize_campaign

        campaign_dir = finalize_campaign(
            args.finalize,
            tag_mlflow=args.tag_mlflow,
        )
        print(f"\nFinalized campaign: {campaign_dir}")
    elif args.spec:
        from src.robustness.campaign import run_campaign

        campaign_dir = run_campaign(args.spec, dry_run=args.dry_run)
        print(f"\nCampaign output: {campaign_dir}")
    else:
        parser.error("Either provide a spec path or use --finalize CAMPAIGN_DIR")


if __name__ == "__main__":
    main()
