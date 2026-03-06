"""Trading and equity-curve metrics computed from run artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _drawdown_series(equity_values: np.ndarray) -> tuple[float, float]:
    """Compute max drawdown (absolute) and max drawdown percent.

    Uses the same numpy rolling-peak logic as runner.py.

    Returns (max_drawdown_abs, max_dd_pct).
    """
    if len(equity_values) == 0:
        return 0.0, 0.0

    rolling_peak = np.maximum.accumulate(equity_values)
    drawdowns = rolling_peak - equity_values
    dd_pct = np.where(
        rolling_peak > 0,
        drawdowns / rolling_peak * 100,
        0.0,
    )
    return float(drawdowns.max()), float(dd_pct.max())


def compute_trading_metrics(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    starting_capital: float,
    equity_gross: pd.DataFrame | None = None,
) -> dict:
    """Compute trading and equity-curve metrics.

    Parameters
    ----------
    equity : DataFrame
        Must contain ``equity`` column. Optionally ``timestamp``.
    trades : DataFrame
        Columns from Fill.to_dict(): pnl, gross_pnl, cost_pnl, etc.
        May be empty.
    starting_capital : float
        Initial capital for return calculations.
    equity_gross : DataFrame | None
        Gross equity curve with ``equity`` column (no costs).

    Returns
    -------
    dict
    """
    result: dict = {}

    # ── Trade-level metrics ──
    n_trades = len(trades)
    result["num_trades"] = n_trades

    if n_trades == 0:
        result.update({
            "total_net_pnl": 0.0,
            "total_net_return": 0.0,
            "win_rate": 0.0,
            "avg_trade_pnl": 0.0,
            "profit_factor": 0.0,
            "total_cost_pnl": 0.0,
            "avg_cost_per_trade": 0.0,
            "total_gross_pnl": 0.0,
        })
    else:
        pnl = trades["pnl"]
        total_net = float(pnl.sum())
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        sum_wins = float(wins.sum()) if len(wins) > 0 else 0.0
        sum_losses_abs = float(losses.abs().sum()) if len(losses) > 0 else 0.0

        result["total_net_pnl"] = total_net
        result["total_net_return"] = total_net / starting_capital * 100 if starting_capital > 0 else 0.0
        result["win_rate"] = float((pnl > 0).mean())
        result["avg_trade_pnl"] = float(pnl.mean())
        result["profit_factor"] = sum_wins / sum_losses_abs if sum_losses_abs > 0 else float("inf") if sum_wins > 0 else 0.0

        # Cost breakdown
        if "cost_pnl" in trades.columns:
            total_cost = float(trades["cost_pnl"].sum())
        elif "gross_pnl" in trades.columns:
            total_cost = float(trades["gross_pnl"].sum()) - total_net
        else:
            total_cost = 0.0

        total_gross = float(trades["gross_pnl"].sum()) if "gross_pnl" in trades.columns else total_net + total_cost

        result["total_cost_pnl"] = total_cost
        result["avg_cost_per_trade"] = total_cost / n_trades
        result["total_gross_pnl"] = total_gross

    # ── Equity-curve metrics ──
    if not equity.empty:
        eq_values = equity["equity"].values.astype(float)
        max_dd_abs, max_dd_pct = _drawdown_series(eq_values)
        result["max_drawdown"] = max_dd_abs
        result["max_dd_pct"] = max_dd_pct

        # Sharpe per bar (not annualized)
        if len(eq_values) > 1:
            denom = eq_values[:-1].copy()
            denom[denom == 0] = np.nan
            bar_returns = np.nan_to_num(np.diff(eq_values) / denom, nan=0.0)
            std = float(np.std(bar_returns, ddof=1))
            result["sharpe_per_bar"] = float(np.mean(bar_returns)) / std if std > 0 else 0.0
        else:
            result["sharpe_per_bar"] = 0.0

        # Trades per day
        if "timestamp" in equity.columns and n_trades > 0:
            ts = pd.to_datetime(equity["timestamp"])
            total_days = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400
            result["trades_per_day"] = n_trades / total_days if total_days > 0 else 0.0
        else:
            result["trades_per_day"] = 0.0
    else:
        result.update({
            "max_drawdown": 0.0,
            "max_dd_pct": 0.0,
            "sharpe_per_bar": 0.0,
            "trades_per_day": 0.0,
        })

    # ── Gross equity metrics ──
    if equity_gross is not None and not equity_gross.empty:
        eq_gross_values = equity_gross["equity"].values.astype(float)
        max_dd_abs_g, max_dd_pct_g = _drawdown_series(eq_gross_values)
        result["max_drawdown_gross"] = max_dd_abs_g
        result["max_dd_pct_gross"] = max_dd_pct_g

        if len(eq_gross_values) > 1:
            denom_g = eq_gross_values[:-1].copy()
            denom_g[denom_g == 0] = np.nan
            bar_returns_g = np.nan_to_num(np.diff(eq_gross_values) / denom_g, nan=0.0)
            std_g = float(np.std(bar_returns_g, ddof=1))
            result["sharpe_per_bar_gross"] = float(np.mean(bar_returns_g)) / std_g if std_g > 0 else 0.0
        else:
            result["sharpe_per_bar_gross"] = 0.0

    return result
