"""Prediction-quality metrics for binary classification forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_prediction_metrics(predictions: pd.DataFrame | None) -> dict:
    """Compute prediction-quality metrics from a predictions DataFrame.

    Parameters
    ----------
    predictions : DataFrame | None
        Must contain ``y_true`` and ``y_prob`` columns.
        Optionally contains ``fold_id``.
        Returns empty dict if None or empty.

    Returns
    -------
    dict
        Keys: brier, log_loss, roc_auc, accuracy, precision, recall, f1,
        n_predictions.  roc_auc is None if only one class present.
    """
    if predictions is None or predictions.empty:
        return {}

    df = predictions[["y_true", "y_prob"]].dropna()
    if df.empty:
        return {}

    y_true = df["y_true"].values
    y_prob = df["y_prob"].values
    y_pred = (y_prob >= 0.5).astype(int)

    n = len(y_true)

    # Brier score
    brier = float(np.mean((y_prob - y_true) ** 2))

    # Log loss and ROC AUC — both require two classes
    ll: float | None = None
    roc_auc: float | None = None
    if len(np.unique(y_true)) > 1:
        from sklearn.metrics import log_loss as sk_log_loss, roc_auc_score
        ll = float(sk_log_loss(y_true, y_prob))
        roc_auc = float(roc_auc_score(y_true, y_prob))

    # Classification metrics at threshold 0.5
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0.0))
    recall = float(recall_score(y_true, y_pred, zero_division=0.0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0.0))

    return {
        "brier": brier,
        "log_loss": ll,
        "roc_auc": roc_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_predictions": n,
    }
