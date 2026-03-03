"""PELT change-point detection regime detector."""

from __future__ import annotations

import numpy as np

from .features import compute_regime_features
from .protocol import RegimeLabel


class PELTDetector:
    """Detect change points via PELT, label segments by Efficiency Ratio.

    Batch detector: call fit_batch() with full price series before bar iteration.
    Segments with mean ER above threshold → TREND, else → RANGE.
    """

    name: str = "PELT"

    def __init__(
        self,
        k: int = 20,
        model: str = "l2",
        pen: float = 10.0,
        min_size: int = 100,
        er_threshold: float = 0.3,
    ) -> None:
        self._k = k
        self._model = model
        self._pen = pen
        self._min_size = min_size
        self._er_threshold = er_threshold
        self.warmup_bars: int = 0  # batch: no streaming warmup

        self._labels: np.ndarray | None = None
        self._confidences: np.ndarray | None = None
        self._raw_values: np.ndarray | None = None
        self._fitted: bool = False

    def fit_batch(self, closes: np.ndarray) -> None:
        """Detect change points and label segments.

        Parameters
        ----------
        closes : np.ndarray
            Full close price series.
        """
        import ruptures

        n = len(closes)

        # Compute ER for each bar (after warmup)
        er_series = np.zeros(n)
        for i in range(self._k, n):
            segment = closes[i - self._k: i + 1]
            er, _, _ = compute_regime_features(segment, self._k)
            er_series[i] = er

        # Run PELT on the ER series
        signal = er_series[self._k:].reshape(-1, 1)
        algo = ruptures.Pelt(model=self._model, min_size=self._min_size)
        algo.fit(signal)
        bkps = algo.predict(pen=self._pen)

        # Label each segment
        self._labels = np.full(n, "RANGE", dtype="U5")
        self._confidences = np.zeros(n)
        self._raw_values = np.zeros(n)

        prev = 0
        segments = []
        for bp in bkps:
            segments.append((prev, bp))
            prev = bp

        for seg_start, seg_end in segments:
            # Map back to original indices
            orig_start = seg_start + self._k
            orig_end = min(seg_end + self._k, n)
            if orig_start >= n:
                continue
            orig_end = min(orig_end, n)

            seg_er = er_series[orig_start:orig_end]
            mean_er = float(seg_er.mean()) if len(seg_er) > 0 else 0.0

            regime = "TREND" if mean_er > self._er_threshold else "RANGE"
            conf = min(1.0, abs(mean_er - self._er_threshold) / self._er_threshold)

            self._labels[orig_start:orig_end] = regime
            self._confidences[orig_start:orig_end] = conf
            self._raw_values[orig_start:orig_end] = mean_er

        self._fitted = True

    def update(
        self, close: float, high: float, low: float, bar_index: int
    ) -> RegimeLabel:
        if not self._fitted or self._labels is None:
            return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)

        if bar_index < len(self._labels):
            return RegimeLabel(
                regime=str(self._labels[bar_index]),
                confidence=float(self._confidences[bar_index]),
                raw_value=float(self._raw_values[bar_index]),
            )
        return RegimeLabel(regime="RANGE", confidence=0.0, raw_value=0.0)
