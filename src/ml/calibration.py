"""Platt scaling calibrator for post-hoc probability calibration.

Fits a sigmoid (via logistic regression) on held-out raw probabilities
to map them to better-calibrated probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class PlattCalibrator:
    """Platt scaling: fit sigmoid on held-out (raw_prob, y_true) pairs.

    Parameters
    ----------
    C : float
        Inverse regularisation strength for the internal logistic regression.
        Default ``1e10`` effectively means no regularisation (pure sigmoid fit).

    Usage
    -----
    >>> cal = PlattCalibrator()
    >>> cal.fit(y_prob_raw, y_true)       # on calibration slice
    >>> calibrated = cal.transform(yhat)  # on test set
    """

    def __init__(self, C: float = 1e10) -> None:
        self._C = C
        self._model = None
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, y_prob_raw: np.ndarray, y_true: np.ndarray) -> None:
        """Fit sigmoid parameters on held-out raw probabilities.

        Parameters
        ----------
        y_prob_raw : array-like, shape (n,)
            Raw predicted probabilities from the base model.
        y_true : array-like, shape (n,)
            Binary ground-truth labels (0 or 1).
        """
        from sklearn.linear_model import LogisticRegression

        y_prob_raw = np.asarray(y_prob_raw, dtype=float).ravel()
        y_true = np.asarray(y_true, dtype=int).ravel()

        self._model = LogisticRegression(C=self._C, solver="lbfgs", max_iter=1000)
        self._model.fit(y_prob_raw.reshape(-1, 1), y_true)
        self._fitted = True

    def transform(self, y_prob_raw: np.ndarray) -> np.ndarray:
        """Apply learned sigmoid to raw probabilities.

        Returns input unchanged if not yet fitted (safety fallback).

        Parameters
        ----------
        y_prob_raw : array-like, shape (n,)
            Raw predicted probabilities.

        Returns
        -------
        np.ndarray
            Calibrated probabilities in [0, 1].
        """
        y_prob_raw = np.asarray(y_prob_raw, dtype=float).ravel()
        if not self._fitted:
            return y_prob_raw
        return self._model.predict_proba(y_prob_raw.reshape(-1, 1))[:, 1]

    def transform_series(self, s: pd.Series) -> pd.Series:
        """Transform a pandas Series, preserving index and name."""
        calibrated = self.transform(s.values)
        return pd.Series(calibrated, index=s.index, name=s.name)
