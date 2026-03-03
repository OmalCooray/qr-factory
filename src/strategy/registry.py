"""Strategy registry — maps type strings to builder functions."""

from __future__ import annotations

from typing import Callable

from src.indicators.core.pipeline import FeatureSpec
from .base import Strategy

StrategyBuilder = Callable[[dict], tuple[Strategy, list[FeatureSpec]]]

_REGISTRY: dict[str, StrategyBuilder] = {}


def register(name: str, builder: StrategyBuilder) -> None:
    """Register a strategy builder under the given name."""
    _REGISTRY[name] = builder


def build_strategy(strategy_cfg: dict) -> tuple[Strategy, list[FeatureSpec]]:
    """Build a strategy + feature specs from a strategy config block.

    Parameters
    ----------
    strategy_cfg : dict
        Must contain a ``type`` key that maps to a registered builder.
        Remaining keys are passed as ``params`` to the builder.

    Returns
    -------
    tuple of (Strategy, list[FeatureSpec])
    """
    cfg = dict(strategy_cfg)  # shallow copy so we don't mutate caller's dict
    strategy_type = cfg.pop("type", None)
    if strategy_type is None:
        raise ValueError("strategy config must contain a 'type' key")
    if strategy_type not in _REGISTRY:
        raise ValueError(
            f"Unknown strategy type '{strategy_type}'. "
            f"Registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[strategy_type](cfg)


# Auto-register built-in strategies
from .ma_crossover import build as _build_ma  # noqa: E402
from .adx_filtered_crossover import build as _build_adx  # noqa: E402
from .donchian_atr_regime import build as _build_donchian  # noqa: E402
from .lorentzian_classification import build as _build_lorentzian  # noqa: E402
from .rsi_bb_confluence import build as _build_rsi_bb  # noqa: E402
from .bb_mean_reversion import build as _build_bb_mr  # noqa: E402
from .graph_mss import build as _build_graph_mss  # noqa: E402

register("ma_crossover", _build_ma)
register("adx_filtered", _build_adx)
register("donchian_atr_regime", _build_donchian)
register("lorentzian_classification", _build_lorentzian)
register("rsi_bb_confluence", _build_rsi_bb)
register("bb_mean_reversion", _build_bb_mr)
register("graph_mss", _build_graph_mss)
