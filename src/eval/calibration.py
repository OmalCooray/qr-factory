"""Calibration evaluation: ECE computation and reliability diagram rendering."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

log = logging.getLogger(__name__)


def compute_calibration(
    predictions: pd.DataFrame | None,
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error and per-bin statistics.

    Parameters
    ----------
    predictions : DataFrame | None
        Must contain ``y_true`` (binary) and ``y_prob`` (float in [0, 1]).
        Returns empty dict if None or empty.
    n_bins : int
        Number of equal-width bins in [0, 1].

    Returns
    -------
    dict
        Keys: n_bins, ece, brier, n_predictions, bins.
        Empty dict if no valid predictions.

    Raises
    ------
    ValueError
        If required columns ``y_true`` or ``y_prob`` are missing.
    """
    if predictions is None or predictions.empty:
        return {}

    missing = {"y_true", "y_prob"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = predictions[["y_true", "y_prob"]].dropna()
    if df.empty:
        return {}

    y_true = df["y_true"].values.astype(float)
    y_prob = df["y_prob"].values.astype(float)
    n = len(y_true)

    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Build equal-width bins: [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins_data: list[dict] = []
    ece = 0.0

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        # Last bin is closed on the right to include 1.0
        if i == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)

        count = int(mask.sum())
        if count > 0:
            mean_predicted = float(y_prob[mask].mean())
            observed_positive_rate = float(y_true[mask].mean())
            ece += (count / n) * abs(mean_predicted - observed_positive_rate)
        else:
            mean_predicted = None
            observed_positive_rate = None

        bins_data.append({
            "bin_lower": round(float(lower), 4),
            "bin_upper": round(float(upper), 4),
            "count": count,
            "mean_predicted": round(mean_predicted, 4) if mean_predicted is not None else None,
            "observed_positive_rate": round(observed_positive_rate, 4) if observed_positive_rate is not None else None,
        })

    return {
        "n_bins": n_bins,
        "ece": round(float(ece), 6),
        "brier": round(float(brier), 6),
        "n_predictions": n,
        "bins": bins_data,
    }


def plot_reliability_diagram(calibration: dict, out_path: Path) -> None:
    """Render a reliability diagram from a calibration dict.

    Parameters
    ----------
    calibration : dict
        Output of ``compute_calibration()``.
    out_path : Path
        Where to save the PNG.
    """
    if not calibration or not calibration.get("bins"):
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    bins = calibration["bins"]
    non_empty = [b for b in bins if b["count"] > 0]

    if not non_empty:
        return

    mean_pred = [b["mean_predicted"] for b in non_empty]
    obs_rate = [b["observed_positive_rate"] for b in non_empty]
    counts = [b["count"] for b in non_empty]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8, 8), gridspec_kw={"height_ratios": [3, 1]},
    )

    # Top: reliability curve
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.plot(mean_pred, obs_rate, "s-", color="#e74c3c", markersize=8, label="Model")
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Observed positive rate")
    ax1.set_title(f"Reliability Diagram (ECE = {calibration['ece']:.4f})")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Bottom: histogram of prediction counts per bin
    bin_centers = [(b["bin_lower"] + b["bin_upper"]) / 2 for b in bins]
    bin_counts = [b["count"] for b in bins]
    width = bins[0]["bin_upper"] - bins[0]["bin_lower"]
    ax2.bar(bin_centers, bin_counts, width=width * 0.9, color="#3498db", alpha=0.7)
    ax2.set_xlabel("Predicted probability")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    log.info("Saved reliability diagram → %s", out_path)


def write_calibration_artifacts(
    run_dir: Path,
    predictions: pd.DataFrame | None,
    n_bins: int = 10,
) -> dict:
    """Compute calibration, write calibration.json and reliability_diagram.png.

    Parameters
    ----------
    run_dir : Path
        Run directory to write into.
    predictions : DataFrame | None
        Predictions with y_true/y_prob.
    n_bins : int
        Number of calibration bins.

    Returns
    -------
    dict
        The calibration dict (empty if no predictions).
    """
    cal = compute_calibration(predictions, n_bins=n_bins)
    if not cal:
        return {}

    # Write calibration.json
    cal_path = run_dir / "calibration.json"
    cal_path.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    log.info("Wrote calibration.json → %s", cal_path)

    # Render reliability diagram
    plot_reliability_diagram(cal, run_dir / "plots" / "reliability_diagram.png")

    return cal
