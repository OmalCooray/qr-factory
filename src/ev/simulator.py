"""EV (Expected Value) Monte-Carlo trade simulator.

Simulates independent, identically-distributed trade sequences to
illustrate the relationship between win rate, expectancy, and variance.
Deterministic via numpy's SeedSequence / default_rng.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend – no GUI needed
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ScenarioConfig:
    """Parameters for one simulation scenario."""
    name: str
    p_win: float
    avg_win: float
    avg_loss: float
    win_std: float = 0.0
    loss_std: float = 0.0
    commission_per_trade: float = 0.0
    slippage: float = 0.0


@dataclass
class SimResult:
    """Aggregated results for a single scenario."""
    name: str
    expectancy_per_trade: float
    variance_per_trade: float
    mean_ending_equity: float
    median_ending_equity: float
    std_ending_equity: float
    probability_of_loss: float
    mean_max_drawdown: float
    ending_equities: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def _simulate_equity_paths(
    rng: np.random.Generator,
    sc: ScenarioConfig,
    n_trades: int,
    n_sims: int,
    starting_capital: float,
) -> np.ndarray:
    """Return an (n_sims, n_trades+1) equity matrix."""
    # Determine win/loss per trade: shape (n_sims, n_trades)
    is_win = rng.random((n_sims, n_trades)) < sc.p_win

    # Payoffs
    if sc.win_std > 0:
        win_payoffs = rng.normal(sc.avg_win, sc.win_std, (n_sims, n_trades))
    else:
        win_payoffs = np.full((n_sims, n_trades), sc.avg_win, dtype=np.float64)

    if sc.loss_std > 0:
        loss_payoffs = rng.normal(sc.avg_loss, sc.loss_std, (n_sims, n_trades))
    else:
        loss_payoffs = np.full((n_sims, n_trades), sc.avg_loss, dtype=np.float64)

    # Payoff matrix: +win if win, -loss if loss
    payoffs = np.where(is_win, win_payoffs, -loss_payoffs)

    # Deduct costs
    costs_per_trade = sc.commission_per_trade + sc.slippage
    payoffs -= costs_per_trade

    # Build equity curves via cumulative sum
    equity = np.empty((n_sims, n_trades + 1), dtype=np.float64)
    equity[:, 0] = starting_capital
    equity[:, 1:] = starting_capital + np.cumsum(payoffs, axis=1)

    return equity


def _max_drawdown_per_sim(equity: np.ndarray) -> np.ndarray:
    """Simple peak-to-trough drawdown for each sim (row)."""
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdowns = running_max - equity
    return np.max(drawdowns, axis=1)


def simulate_scenario(
    seed: int,
    sc: ScenarioConfig,
    n_trades: int,
    n_sims: int,
    starting_capital: float,
) -> SimResult:
    """Run the full Monte-Carlo simulation for one scenario."""
    rng = np.random.default_rng(seed)

    equity = _simulate_equity_paths(rng, sc, n_trades, n_sims, starting_capital)

    ending = equity[:, -1]
    costs = sc.commission_per_trade + sc.slippage

    # Analytical expectancy
    expectancy = sc.p_win * sc.avg_win + (1 - sc.p_win) * (-sc.avg_loss) - costs

    # Per-trade variance (analytical for constant payoffs)
    mean_payoff = sc.p_win * sc.avg_win + (1 - sc.p_win) * (-sc.avg_loss)
    variance = (
        sc.p_win * (sc.avg_win - mean_payoff) ** 2
        + (1 - sc.p_win) * (-sc.avg_loss - mean_payoff) ** 2
    )

    dd = _max_drawdown_per_sim(equity)

    return SimResult(
        name=sc.name,
        expectancy_per_trade=float(expectancy),
        variance_per_trade=float(variance),
        mean_ending_equity=float(np.mean(ending)),
        median_ending_equity=float(np.median(ending)),
        std_ending_equity=float(np.std(ending)),
        probability_of_loss=float(np.mean(ending < starting_capital)),
        mean_max_drawdown=float(np.mean(dd)),
        ending_equities=ending,
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def result_to_dict(r: SimResult) -> dict[str, Any]:
    """Serialise a SimResult to a plain dict (no numpy arrays)."""
    return {
        "name": r.name,
        "expectancy_per_trade": round(r.expectancy_per_trade, 4),
        "variance_per_trade": round(r.variance_per_trade, 4),
        "mean_ending_equity": round(r.mean_ending_equity, 2),
        "median_ending_equity": round(r.median_ending_equity, 2),
        "std_ending_equity": round(r.std_ending_equity, 2),
        "probability_of_loss": round(r.probability_of_loss, 4),
        "mean_max_drawdown": round(r.mean_max_drawdown, 2),
    }


def save_results_json(results: list[SimResult], path: Path) -> None:
    """Write all scenario results to a single JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [result_to_dict(r) for r in results]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def plot_distributions(results: list[SimResult], output_path: Path) -> None:
    """Histogram of ending equity for each scenario, side-by-side."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)

    for idx, r in enumerate(results):
        ax = axes[0, idx]
        ax.hist(r.ending_equities, bins=80, color="#4682b4", edgecolor="white",
                alpha=0.85, linewidth=0.4)
        ax.axvline(r.mean_ending_equity, color="#e74c3c", linestyle="--",
                   linewidth=1.5, label=f"Mean: {r.mean_ending_equity:,.0f}")
        ax.axvline(r.median_ending_equity, color="#2ecc71", linestyle="--",
                   linewidth=1.5, label=f"Median: {r.median_ending_equity:,.0f}")
        ax.axvline(10_000, color="grey", linestyle=":", linewidth=1,
                   label="Starting capital")
        ax.set_title(r.name.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Ending Equity ($)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)

    fig.suptitle("EV Simulator – Ending Equity Distribution", fontsize=13,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
