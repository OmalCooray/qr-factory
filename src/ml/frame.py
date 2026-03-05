"""ForecastFrame builder -- fold-scoped DataFrame with features + optional labels.

ForecastFrame is the boundary between feature/label engineering (ML world) and
trading simulation (engine world).  It is a plain ``pd.DataFrame`` with columns:

- ``X_*``      : features (from FeaturePipeline, prefixed)
- ``y``        : label (optional, forward-looking)
- ``ret_fwd``  : realised forward return used for labeling (optional)
- ``w``        : sample weight (optional, not yet implemented)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_forecast_frame(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    label_cfg: dict | None = None,
) -> pd.DataFrame:
    """Build a ForecastFrame: features (X_-prefixed) + optional labels.

    Parameters
    ----------
    bars : pd.DataFrame
        OHLCV bars for this fold.  Must have the same length as *features*.
    features : pd.DataFrame
        Pre-computed feature columns (from ``FeaturePipeline``).
    label_cfg : dict | None
        Label configuration.  If ``None``, returns features-only frame.
        Supported ``type`` values: ``"return_sign"``.

    Returns
    -------
    pd.DataFrame
        Index aligned to *bars* (minus trailing rows when labels are present).
        Feature columns prefixed ``X_``, optional ``y`` and ``ret_fwd``.
    """
    if len(bars) != len(features):
        raise ValueError(
            f"bars ({len(bars)}) and features ({len(features)}) length mismatch"
        )

    # Prefix all feature columns with X_
    ff = features.rename(columns={c: f"X_{c}" for c in features.columns}).copy()
    ff.index = bars.index  # ensure shared index

    if label_cfg is not None:
        ff = _add_labels(ff, bars, label_cfg)

    # Drop rows where any X_* feature is NaN (warmup rows, etc.)
    x_cols = [c for c in ff.columns if c.startswith("X_")]
    ff = ff.dropna(subset=x_cols)

    return ff


def _add_labels(
    ff: pd.DataFrame,
    bars: pd.DataFrame,
    label_cfg: dict,
) -> pd.DataFrame:
    """Add forward-looking labels computed on the provided slice ONLY.

    After computing, rows with NaN labels (last *horizon* rows) are dropped
    so the returned frame has valid X and y on every row.

    Supported label types
    ---------------------
    - ``return_sign``            : y=1 iff ret_fwd > threshold
    - ``return_sign_cost_aware`` : y=1 iff move exceeds cost_buffer (direction-aware)
    """
    label_type = label_cfg["type"]
    horizon: int = label_cfg["horizon"]
    price_col: str = label_cfg.get("price_col", "close")
    prices = bars[price_col].reindex(ff.index)
    ret_fwd = prices.shift(-horizon) / prices - 1.0

    ff = ff.copy()
    ff["ret_fwd"] = ret_fwd.values

    if label_type == "return_sign":
        threshold = label_cfg.get("threshold", 0.0)
        ff["y"] = (ret_fwd.values > threshold).astype(int)

    elif label_type == "return_sign_cost_aware":
        cost_buffer = label_cfg.get("cost_buffer", 0.0)
        direction = label_cfg.get("direction", "up")
        if direction == "up":
            ff["y"] = (ret_fwd.values > cost_buffer).astype(int)
        elif direction == "down":
            ff["y"] = (ret_fwd.values < -cost_buffer).astype(int)
        elif direction == "both":
            ff["y"] = (np.abs(ret_fwd.values) > cost_buffer).astype(int)
        else:
            raise ValueError(f"Unknown direction: {direction!r}")
    else:
        raise ValueError(f"Unsupported label type: {label_type!r}")

    ff = ff.dropna(subset=["ret_fwd"])
    ff["y"] = ff["y"].astype(int)

    return ff
