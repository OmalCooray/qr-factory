"""Leakage-safe walk-forward time-series splitter.

Yields (train_indices, test_indices) numpy arrays for expanding or rolling
walk-forward validation.  Every fold guarantees ``train.max() < test.min()``.

Modes
-----
- **expanding**: training window grows from index 0; test window slides forward.
- **rolling**: training window is a fixed-size sliding window; test window follows.

Parameters
----------
mode : {"expanding", "rolling"}
train_size : int  – initial (expanding) or fixed (rolling) training window length.
test_size : int   – length of each test window.
step : int        – how far to advance between folds (default = test_size).
                    Must be >= test_size to prevent overlapping test windows.
drop_last : bool  – if True (default), skip a final fold whose test window would
                     be shorter than *test_size*.  If False, allow at most one
                     partial final fold (test length in ``[1, test_size-1]``).

Raises
------
ValueError
    At construction: if mode is invalid, any size/step <= 0, or step < test_size.
    At split(): if *n_samples* cannot produce at least one fold:
    - drop_last=True  requires ``n_samples >= train_size + test_size``
    - drop_last=False requires ``n_samples >= train_size + 1``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Tuple

import numpy as np


@dataclass(frozen=True)
class TimeSeriesSplitter:
    """Walk-forward splitter that guarantees no future-leakage.

    Enforces ``step >= test_size`` to prevent overlapping test windows
    (research-grade walk-forward requires non-overlapping evaluation sets).
    """

    mode: Literal["expanding", "rolling"]
    train_size: int
    test_size: int
    step: int | None = None       # defaults to test_size
    drop_last: bool = True

    # ------------------------------------------------------------------
    # Validation (runs once, at construction time via __post_init__)
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        if self.mode not in ("expanding", "rolling"):
            raise ValueError(
                f"mode must be 'expanding' or 'rolling', got {self.mode!r}"
            )
        if self.train_size <= 0:
            raise ValueError(f"train_size must be > 0, got {self.train_size}")
        if self.test_size <= 0:
            raise ValueError(f"test_size must be > 0, got {self.test_size}")

        # Materialise default step
        if self.step is None:
            object.__setattr__(self, "step", self.test_size)

        if self.step <= 0:  # type: ignore[operator]
            raise ValueError(f"step must be > 0, got {self.step}")
        if self.step < self.test_size:  # type: ignore[operator]
            raise ValueError(
                f"step ({self.step}) < test_size ({self.test_size}): "
                "overlapping test windows disallowed for research-grade "
                "walk-forward"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split(self, n_samples: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield (train_idx, test_idx) folds for *n_samples* observations.

        Parameters
        ----------
        n_samples : int
            Total number of data points (indices ``0 .. n_samples-1``).

        Raises
        ------
        ValueError
            If *n_samples* cannot produce at least one fold:
            - drop_last=True  requires ``n_samples >= train_size + test_size``
            - drop_last=False requires ``n_samples >= train_size + 1``
        """
        if self.drop_last:
            min_required = self.train_size + self.test_size
            if n_samples < min_required:
                raise ValueError(
                    f"n_samples ({n_samples}) must be >= train_size + test_size "
                    f"({min_required}) when drop_last=True to guarantee at "
                    f"least one full test fold"
                )
        else:
            min_required = self.train_size + 1
            if n_samples < min_required:
                raise ValueError(
                    f"n_samples ({n_samples}) must be >= train_size + 1 "
                    f"({min_required}) to produce at least one test sample"
                )

        step: int = self.step  # type: ignore[assignment]

        if self.mode == "expanding":
            yield from self._expanding(n_samples, step)
        else:
            yield from self._rolling(n_samples, step)

    def n_splits(self, n_samples: int) -> int:
        """Return the number of folds that *split()* will produce."""
        return sum(1 for _ in self.split(n_samples))

    # ------------------------------------------------------------------
    # Private generators
    # ------------------------------------------------------------------
    def _expanding(
        self, n: int, step: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        train_end = self.train_size           # exclusive upper bound of train

        while train_end < n:
            test_start = train_end
            test_end = min(test_start + self.test_size, n)

            if self.drop_last and (test_end - test_start) < self.test_size:
                break

            yield (
                np.arange(0, train_end),
                np.arange(test_start, test_end),
            )

            train_end += step

    def _rolling(
        self, n: int, step: int
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        train_start = 0

        while train_start + self.train_size < n:
            train_end = train_start + self.train_size
            test_start = train_end
            test_end = min(test_start + self.test_size, n)

            if self.drop_last and (test_end - test_start) < self.test_size:
                break

            yield (
                np.arange(train_start, train_end),
                np.arange(test_start, test_end),
            )

            train_start += step

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"TimeSeriesSplitter(mode={self.mode!r}, train_size={self.train_size}, "
            f"test_size={self.test_size}, step={self.step}, "
            f"drop_last={self.drop_last})"
        )
