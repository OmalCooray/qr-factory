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
