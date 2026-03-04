"""Plotting utilities for backtest reporting."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)


def plot_close_price(df: pd.DataFrame, out_path: str | Path) -> None:
    """Plot close price vs time and save as PNG.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``time`` and ``close`` columns.
    out_path : str | Path
        Destination file path (e.g. ``plots/close_price.png``).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["time"], df["close"], linewidth=0.5, color="#d4af37")
    ax.set_title(f"Close Price  ({df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    log.info("Saved close-price plot → %s", out_path)


def plot_equity(df: pd.DataFrame, out_path: str | Path) -> None:
    """Plot equity curve and save as PNG.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp`` and ``equity`` columns.
    out_path : str | Path
        Destination file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df["timestamp"], df["equity"], linewidth=1.0, color="#2ecc71")
    ax.set_title(f"Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    log.info("Saved equity plot → %s", out_path)


def plot_gross_vs_net_equity(df: pd.DataFrame, out_path: str | Path) -> None:
    """Plot gross and net equity curves on the same chart.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``timestamp``, ``equity`` (net), and ``equity_gross`` columns.
    out_path : str | Path
        Destination file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(plot_df["timestamp"]):
        plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"], utc=True)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(plot_df["timestamp"], plot_df["equity_gross"], linewidth=1.0,
            color="#3498db", label="Gross")
    ax.plot(plot_df["timestamp"], plot_df["equity"], linewidth=1.0,
            color="#e74c3c", label="Net")
    ax.set_title("Gross vs Net Equity")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    log.info("Saved gross-vs-net equity plot → %s", out_path)


def plot_fold_pnl(fold_results: list[dict], out_path: str | Path) -> None:
    """Bar chart of gross vs net PnL per fold.

    Parameters
    ----------
    fold_results : list[dict]
        Each dict must have ``fold_id``, ``gross_pnl``, ``net_pnl``.
    out_path : str | Path
        Destination file path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fold_ids = [f["fold_id"] for f in fold_results]
    gross = [f["gross_pnl"] for f in fold_results]
    net = [f["net_pnl"] for f in fold_results]

    x = range(len(fold_ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([i - width / 2 for i in x], gross, width, label="Gross PnL", color="#3498db")
    ax.bar([i + width / 2 for i in x], net, width, label="Net PnL", color="#e74c3c")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"F{fid}" for fid in fold_ids])
    ax.set_xlabel("Fold")
    ax.set_ylabel("PnL")
    ax.set_title("Walk-Forward: Gross vs Net PnL per Fold")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)

    log.info("Saved fold PnL plot → %s", out_path)
