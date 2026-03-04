"""Tests for TimeSeriesSplitter — leakage, determinism, fold counts, edges."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.validation import TimeSeriesSplitter


# ── Helpers ──────────────────────────────────────────────────────────────


def _expected_fold_count(
    n: int,
    train_size: int,
    test_size: int,
    step: int,
    mode: str,
    drop_last: bool,
) -> int:
    """Compute expected number of folds analytically."""
    if mode == "expanding":
        # First fold: train_end = train_size.  Last full fold starts when
        # train_end + test_size <= n, i.e. train_end <= n - test_size.
        # train_end for fold k = train_size + k*step.
        available = n - train_size      # samples after initial train window
        if drop_last:
            if available < test_size:
                return 0
            return (available - test_size) // step + 1
        else:
            if available <= 0:
                return 0
            return (available - 1) // step + 1
    else:  # rolling
        # train_start for fold k = k*step.
        # test_start = train_start + train_size.
        # Condition: test_start < n  (at least 1 test sample).
        # Full-test condition: test_start + test_size <= n.
        available = n - train_size
        if drop_last:
            if available < test_size:
                return 0
            return (available - test_size) // step + 1
        else:
            if available <= 0:
                return 0
            return (available - 1) // step + 1


# ── Construction / validation tests ──────────────────────────────────────


class TestConstruction:
    def test_default_step_equals_test_size(self):
        s = TimeSeriesSplitter(mode="expanding", train_size=100, test_size=50)
        assert s.step == 50

    def test_explicit_step(self):
        s = TimeSeriesSplitter(mode="rolling", train_size=100, test_size=50, step=75)
        assert s.step == 75

    @pytest.mark.parametrize("mode", ["expanding", "rolling"])
    def test_frozen(self, mode: str):
        s = TimeSeriesSplitter(mode=mode, train_size=100, test_size=50)
        with pytest.raises(AttributeError):
            s.train_size = 200  # type: ignore[misc]


class TestValidationErrors:
    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            TimeSeriesSplitter(mode="invalid", train_size=100, test_size=50)  # type: ignore[arg-type]

    def test_train_size_zero(self):
        with pytest.raises(ValueError, match="train_size"):
            TimeSeriesSplitter(mode="expanding", train_size=0, test_size=50)

    def test_test_size_negative(self):
        with pytest.raises(ValueError, match="test_size"):
            TimeSeriesSplitter(mode="expanding", train_size=100, test_size=-1)

    def test_step_zero(self):
        with pytest.raises(ValueError, match="step"):
            TimeSeriesSplitter(mode="expanding", train_size=100, test_size=50, step=0)

    def test_step_less_than_test_size(self):
        with pytest.raises(ValueError, match="overlapping"):
            TimeSeriesSplitter(mode="expanding", train_size=100, test_size=50, step=25)

    @pytest.mark.parametrize("mode", ["expanding", "rolling"])
    def test_step_less_than_test_size_both_modes(self, mode):
        with pytest.raises(ValueError, match="overlapping"):
            TimeSeriesSplitter(mode=mode, train_size=100, test_size=50, step=49)

    def test_n_samples_too_small_drop_last_false(self):
        s = TimeSeriesSplitter(mode="expanding", train_size=100, test_size=50, drop_last=False)
        with pytest.raises(ValueError, match="n_samples"):
            list(s.split(100))  # need at least 101

    def test_n_samples_too_small_rolling_drop_last_false(self):
        s = TimeSeriesSplitter(mode="rolling", train_size=100, test_size=50, drop_last=False)
        with pytest.raises(ValueError, match="n_samples"):
            list(s.split(100))

    @pytest.mark.parametrize("mode", ["expanding", "rolling"])
    def test_n_samples_too_small_drop_last_true(self, mode):
        """Policy A: drop_last=True requires n >= train_size + test_size."""
        s = TimeSeriesSplitter(mode=mode, train_size=100, test_size=50, drop_last=True)
        # 149 < 150 → ValueError
        with pytest.raises(ValueError, match="n_samples"):
            list(s.split(149))

    @pytest.mark.parametrize("mode", ["expanding", "rolling"])
    def test_n_samples_exactly_enough_drop_last_true(self, mode):
        """Boundary: n == train_size + test_size should yield exactly 1 fold."""
        s = TimeSeriesSplitter(mode=mode, train_size=100, test_size=50, drop_last=True)
        folds = list(s.split(150))
        assert len(folds) == 1


# ── Leakage invariant ────────────────────────────────────────────────────


N_SAMPLES = 2000


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
@pytest.mark.parametrize(
    "train_size, test_size, step",
    [
        (500, 100, 100),
        (500, 100, 200),   # step > test_size (gaps between test windows)
        (200, 200, 200),
        (800, 50, 50),
    ],
)
class TestLeakage:
    def test_train_before_test(self, mode, train_size, test_size, step):
        s = TimeSeriesSplitter(
            mode=mode, train_size=train_size, test_size=test_size, step=step
        )
        for fold_i, (train, test) in enumerate(s.split(N_SAMPLES)):
            assert train.max() < test.min(), (
                f"Leakage in fold {fold_i}: train.max()={train.max()}, "
                f"test.min()={test.min()}"
            )

    def test_indices_in_bounds(self, mode, train_size, test_size, step):
        s = TimeSeriesSplitter(
            mode=mode, train_size=train_size, test_size=test_size, step=step
        )
        for train, test in s.split(N_SAMPLES):
            assert train.min() >= 0
            assert test.max() < N_SAMPLES

    def test_non_empty_folds(self, mode, train_size, test_size, step):
        s = TimeSeriesSplitter(
            mode=mode, train_size=train_size, test_size=test_size, step=step
        )
        for train, test in s.split(N_SAMPLES):
            assert len(train) > 0
            assert len(test) > 0


# ── Determinism invariant ────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_determinism(mode):
    s = TimeSeriesSplitter(mode=mode, train_size=500, test_size=100, step=100)
    run1 = [(tr.tolist(), te.tolist()) for tr, te in s.split(N_SAMPLES)]
    run2 = [(tr.tolist(), te.tolist()) for tr, te in s.split(N_SAMPLES)]
    assert run1 == run2


# ── Fold count ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize(
    "n, train_size, test_size, step",
    [
        (2000, 500, 100, 100),
        (2000, 500, 100, 150),  # step > test_size
        (1000, 200, 200, 200),
        (1000, 800, 50, 50),
        (150, 100, 50, 50),   # exactly 1 fold
        (149, 100, 50, 50),   # drop_last=True → ValueError; False → 1 partial
    ],
)
def test_fold_count(mode, drop_last, n, train_size, test_size, step):
    s = TimeSeriesSplitter(
        mode=mode,
        train_size=train_size,
        test_size=test_size,
        step=step,
        drop_last=drop_last,
    )
    # Policy A: fail-fast thresholds
    if drop_last and n < train_size + test_size:
        with pytest.raises(ValueError):
            list(s.split(n))
        return
    if not drop_last and n < train_size + 1:
        with pytest.raises(ValueError):
            list(s.split(n))
        return

    folds = list(s.split(n))
    expected = _expected_fold_count(n, train_size, test_size, step, mode, drop_last)
    assert len(folds) == expected, (
        f"mode={mode} drop_last={drop_last} n={n} train={train_size} "
        f"test={test_size} step={step}: got {len(folds)}, expected {expected}"
    )


# ── drop_last=False partial final fold ───────────────────────────────────


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_partial_last_fold(mode):
    """When drop_last=False, the last fold may have fewer test samples."""
    # 501 samples, train=500, test=100, step=100 → only 1 sample for test
    s = TimeSeriesSplitter(
        mode=mode, train_size=500, test_size=100, step=100, drop_last=False
    )
    folds = list(s.split(501))
    assert len(folds) >= 1
    _train, last_test = folds[-1]
    assert 0 < len(last_test) <= 100


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_drop_last_true_skips_partial(mode):
    """When drop_last=True, partial final fold is skipped."""
    # n=650: 1 full fold (500+100) + partial 50 remaining
    s_drop = TimeSeriesSplitter(
        mode=mode, train_size=500, test_size=100, step=100, drop_last=True
    )
    s_keep = TimeSeriesSplitter(
        mode=mode, train_size=500, test_size=100, step=100, drop_last=False
    )
    folds_drop = list(s_drop.split(650))
    folds_keep = list(s_keep.split(650))
    # drop_last should have fewer-or-equal folds
    assert len(folds_drop) <= len(folds_keep)
    # every test fold in drop_last version must be full-size
    for _, test in folds_drop:
        assert len(test) == 100


# ── Expanding mode: train window grows ───────────────────────────────────


def test_expanding_train_grows():
    s = TimeSeriesSplitter(mode="expanding", train_size=200, test_size=50, step=50)
    sizes = [len(tr) for tr, _ in s.split(1000)]
    # each successive train window should be larger than the previous
    for i in range(1, len(sizes)):
        assert sizes[i] > sizes[i - 1]


# ── Rolling mode: train window is fixed ──────────────────────────────────


def test_rolling_train_fixed():
    s = TimeSeriesSplitter(mode="rolling", train_size=200, test_size=50, step=50)
    sizes = [len(tr) for tr, _ in s.split(1000)]
    assert all(sz == 200 for sz in sizes)


# ── No test overlap when step >= test_size ───────────────────────────────


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_no_test_overlap_when_step_gte_test_size(mode):
    s = TimeSeriesSplitter(mode=mode, train_size=300, test_size=100, step=100)
    folds = list(s.split(N_SAMPLES))
    for i in range(1, len(folds)):
        prev_test = folds[i - 1][1]
        curr_test = folds[i][1]
        assert prev_test.max() < curr_test.min()


# ── n_splits helper ─────────────────────────────────────────────────────


def test_n_splits_matches():
    s = TimeSeriesSplitter(mode="expanding", train_size=300, test_size=100, step=100)
    assert s.n_splits(N_SAMPLES) == len(list(s.split(N_SAMPLES)))
