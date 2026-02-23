"""Pure-math drawdown utilities — no side effects, no I/O."""

from __future__ import annotations

from dataclasses import dataclass


def compute_drawdown_pct(peak: float, current: float) -> float:
    """Return drawdown as a percentage of *peak*, clamped >= 0.

    Parameters
    ----------
    peak : float
        High-water-mark equity.
    current : float
        Current equity value.

    Returns
    -------
    float
        ``(peak - current) / peak * 100``, or ``0.0`` when *peak* <= 0
        or *current* >= *peak*.
    """
    if peak <= 0:
        return 0.0
    dd = (peak - current) / peak * 100
    return max(dd, 0.0)


@dataclass(frozen=True)
class DrawdownState:
    """Immutable snapshot returned by :class:`DrawdownTracker.update`."""

    peak_equity: float
    current_drawdown_pct: float
    max_drawdown_pct: float


class DrawdownTracker:
    """Stateful high-water-mark tracker with running max drawdown.

    Create one instance per equity curve you want to monitor.  Call
    :meth:`update` on every bar with the current mark-to-market equity.
    """

    def __init__(self, initial_equity: float) -> None:
        self._peak_equity = initial_equity
        self._max_drawdown_pct = 0.0

    # -- public API ----------------------------------------------------------

    def update(self, current_equity: float) -> DrawdownState:
        """Ingest *current_equity* and return a frozen state snapshot."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        dd_pct = compute_drawdown_pct(self._peak_equity, current_equity)

        if dd_pct > self._max_drawdown_pct:
            self._max_drawdown_pct = dd_pct

        return DrawdownState(
            peak_equity=self._peak_equity,
            current_drawdown_pct=dd_pct,
            max_drawdown_pct=self._max_drawdown_pct,
        )

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def max_drawdown_pct(self) -> float:
        return self._max_drawdown_pct
