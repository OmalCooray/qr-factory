#!/usr/bin/env python
"""CLI entry-point for the EV simulator.

Usage:
    uv run python scripts/run_ev_sim.py --config configs/ev_base.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

# Ensure project root is on sys.path so `src.*` imports resolve when
# invoked directly via `python scripts/run_ev_sim.py`.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ev.simulator import (
    ScenarioConfig,
    SimResult,
    plot_distributions,
    result_to_dict,
    save_results_json,
    simulate_scenario,
)


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EV Monte-Carlo simulation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/ev_base.yaml"),
        help="Path to YAML config file",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    seed: int = cfg["seed"]
    starting_capital: float = cfg["starting_capital"]
    n_trades: int = cfg["n_trades"]
    n_sims: int = cfg["n_sims"]

    results: list[SimResult] = []

    print("=" * 60)
    print("  EV Simulator")
    print("=" * 60)

    for i, sc_raw in enumerate(cfg["scenarios"]):
        sc = ScenarioConfig(
            name=sc_raw["name"],
            p_win=sc_raw["p_win"],
            avg_win=sc_raw["avg_win"],
            avg_loss=sc_raw["avg_loss"],
            win_std=sc_raw.get("win_std", 0.0),
            loss_std=sc_raw.get("loss_std", 0.0),
            commission_per_trade=sc_raw.get("commission_per_trade", 0.0),
            slippage=sc_raw.get("slippage", 0.0),
        )

        # Unique seed per scenario for independence, but still deterministic
        scenario_seed = seed + i
        r = simulate_scenario(scenario_seed, sc, n_trades, n_sims, starting_capital)
        results.append(r)

        d = result_to_dict(r)
        print(f"\n  Scenario: {d['name']}")
        print(f"    Expectancy/trade : {d['expectancy_per_trade']:+.4f}")
        print(f"    Mean ending eq.  : {d['mean_ending_equity']:>10,.2f}")
        print(f"    Median ending eq.: {d['median_ending_equity']:>10,.2f}")
        print(f"    Prob of loss     : {d['probability_of_loss']:.2%}")
        print(f"    Mean max DD      : {d['mean_max_drawdown']:>10,.2f}")

    # ------ outputs ------
    json_path = Path("notes/ev_sim_results.json")
    plot_path = Path("notes/plots/ev_ending_equity.png")

    save_results_json(results, json_path)
    plot_distributions(results, plot_path)

    print("\n" + "-" * 60)
    print(f"  Results JSON : {json_path}")
    print(f"  Plot         : {plot_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
