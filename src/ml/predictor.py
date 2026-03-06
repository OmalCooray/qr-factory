"""Predictor interface and implementations for walk-forward ML integration."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd


class Predictor(Protocol):
    """Minimal predictor interface for walk-forward ML integration.

    ``fit`` receives a ForecastFrame with ``X_*`` feature columns and a ``y``
    label column.  ``predict`` receives a ForecastFrame (with or without ``y``)
    and returns a Series aligned to the input index.
    """

    def fit(self, train_ff: pd.DataFrame) -> None: ...

    def predict(self, test_ff: pd.DataFrame) -> pd.Series: ...


class SklearnLogisticPredictor:
    """Logistic regression predictor using sklearn.

    Parameters
    ----------
    scale : bool
        If True, fit a StandardScaler on training features and apply it
        to both train and test.  Encapsulated inside the predictor so the
        runner needs no changes.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 200,
        class_weight: str = "balanced",
        solver: str = "lbfgs",
        scale: bool = False,
        random_state: int = 0,
    ) -> None:
        from sklearn.linear_model import LogisticRegression

        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            solver=solver,
            random_state=random_state,
        )
        self._scale = scale
        self._scaler = None
        self._x_cols: list[str] = []

    def fit(self, train_ff: pd.DataFrame) -> None:
        self._x_cols = sorted(c for c in train_ff.columns if c.startswith("X_"))
        X = train_ff[self._x_cols].values
        y = train_ff["y"].values.astype(int)
        if self._scale:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        self._model.fit(X, y)

    def predict(self, test_ff: pd.DataFrame) -> pd.Series:
        X = test_ff[self._x_cols].values
        if self._scale and self._scaler is not None:
            X = self._scaler.transform(X)
        proba = self._model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=test_ff.index, name="yhat")


class SklearnGBClassifierPredictor:
    """Gradient boosting classifier predictor using sklearn.

    Tree-based model — no feature scaling needed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 4,
        min_samples_leaf: int = 50,
        subsample: float = 0.8,
        random_state: int = 0,
    ) -> None:
        from sklearn.ensemble import GradientBoostingClassifier

        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
        )
        self._x_cols: list[str] = []

    def fit(self, train_ff: pd.DataFrame) -> None:
        self._x_cols = sorted(c for c in train_ff.columns if c.startswith("X_"))
        X = train_ff[self._x_cols].values
        y = train_ff["y"].values.astype(int)
        self._model.fit(X, y)

    def predict(self, test_ff: pd.DataFrame) -> pd.Series:
        X = test_ff[self._x_cols].values
        proba = self._model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=test_ff.index, name="yhat")


class XGBoostPredictor:
    """XGBoost classifier predictor.

    Tree-based model — no feature scaling needed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        min_child_weight: int = 5,
        random_state: int = 0,
    ) -> None:
        from xgboost import XGBClassifier

        self._model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_weight=min_child_weight,
            random_state=random_state,
            eval_metric="logloss",
        )
        self._x_cols: list[str] = []

    def fit(self, train_ff: pd.DataFrame) -> None:
        self._x_cols = sorted(c for c in train_ff.columns if c.startswith("X_"))
        X = train_ff[self._x_cols].values
        y = train_ff["y"].values.astype(int)
        self._model.fit(X, y)

    def predict(self, test_ff: pd.DataFrame) -> pd.Series:
        X = test_ff[self._x_cols].values
        proba = self._model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=test_ff.index, name="yhat")


class LightGBMPredictor:
    """LightGBM classifier predictor.

    Tree-based model — no feature scaling needed.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        min_child_samples: int = 50,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: int = 0,
    ) -> None:
        from lightgbm import LGBMClassifier

        self._model = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_samples=min_child_samples,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            verbose=-1,
        )
        self._x_cols: list[str] = []

    def fit(self, train_ff: pd.DataFrame) -> None:
        self._x_cols = sorted(c for c in train_ff.columns if c.startswith("X_"))
        X = train_ff[self._x_cols].values
        y = train_ff["y"].values.astype(int)
        self._model.fit(X, y)

    def predict(self, test_ff: pd.DataFrame) -> pd.Series:
        X = test_ff[self._x_cols].values
        proba = self._model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=test_ff.index, name="yhat")


class DummyPredictor:
    """Always-0.5 predictor for wiring tests.

    ``fit`` is a no-op.  ``predict`` returns 0.5 for every row.
    """

    def fit(self, train_ff: pd.DataFrame) -> None:
        pass

    def predict(self, test_ff: pd.DataFrame) -> pd.Series:
        return pd.Series(0.5, index=test_ff.index, name="yhat")


# ── Registry ────────────────────────────────────────────────────────────────

_PREDICTOR_REGISTRY: dict[str, type] = {
    "dummy": DummyPredictor,
    "logistic": SklearnLogisticPredictor,
    "gradient_boosting": SklearnGBClassifierPredictor,
    "xgboost": XGBoostPredictor,
    "lightgbm": LightGBMPredictor,
}


def build_predictor(model_cfg: dict) -> Predictor:
    """Build a Predictor from a config dict.

    Parameters
    ----------
    model_cfg : dict
        Must contain a ``"type"`` key (e.g. ``"dummy"``).  Remaining keys are
        forwarded as keyword arguments to the predictor constructor.

    Returns
    -------
    Predictor
    """
    cfg = dict(model_cfg)  # shallow copy — don't mutate caller's dict
    model_type = cfg.pop("type")
    cls = _PREDICTOR_REGISTRY.get(model_type)
    if cls is None:
        raise ValueError(
            f"Unknown predictor type: {model_type!r}. "
            f"Available: {list(_PREDICTOR_REGISTRY)}"
        )
    return cls(**cfg)
