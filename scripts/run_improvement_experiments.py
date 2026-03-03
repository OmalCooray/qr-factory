"""Run improvement experiments and collect comparison metrics."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.engine.runner import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

EXPERIMENTS = [
    ("Baseline",          "configs/donchian_atr_regime.yaml"),
    ("A: Tighter regime", "configs/exp_A_tighter_regime.yaml"),
    ("B: Long-only",      "configs/exp_B_long_only.yaml"),
    ("C: ATR sizing",     "configs/exp_C_atr_sizing.yaml"),
    ("D: Wider stop",     "configs/exp_D_wider_stop.yaml"),
    ("E: Wider exit",     "configs/exp_E_wider_exit.yaml"),
]


def main() -> None:
    runs_dir = _REPO_ROOT / "runs"
    rows: list[dict] = []

    for name, config_path in EXPERIMENTS:
        log.info("=" * 60)
        log.info("EXPERIMENT: %s", name)
        log.info("Config: %s", config_path)
        log.info("=" * 60)

        t0 = time.time()
        try:
            run_id = run_backtest(config_path)
        except Exception:
            log.exception("Experiment %s FAILED", name)
            continue
        elapsed = time.time() - t0

        metrics_path = runs_dir / run_id / "metrics.json"
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        rows.append({
            "experiment": name,
            "run_id": run_id,
            "ending_equity": metrics["ending_equity"],
            "total_pnl": metrics["total_pnl"],
            "n_trades": metrics["n_trades"],
            "win_rate": metrics["win_rate"],
            "average_win": metrics["average_win"],
            "average_loss": metrics["average_loss"],
            "max_drawdown_pct": metrics["max_drawdown_pct"],
            "elapsed_s": round(elapsed, 1),
        })

        log.info("  -> Equity: %.2f, PnL: %.2f, Trades: %d, Time: %.1fs",
                 metrics["ending_equity"], metrics["total_pnl"],
                 metrics["n_trades"], elapsed)

    if not rows:
        log.error("No experiments completed.")
        sys.exit(1)

    # Print summary table
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print("IMPROVEMENT EXPERIMENT COMPARISON")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("=" * 110)

    # Save
    out_path = runs_dir / "improvement_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    log.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
