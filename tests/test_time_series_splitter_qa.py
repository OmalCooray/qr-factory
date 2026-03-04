import pytest
import numpy as np
import random
from src.validation.time_series_splitter import TimeSeriesSplitter

# A) Leakage invariants (core)
@pytest.mark.parametrize("mode", ["expanding", "rolling"])
@pytest.mark.parametrize("train_size, test_size, step", [
    (10, 5, 5),   # step = test_size
    (10, 5, 10),  # step > test_size
])
def test_leakage_invariants(mode, train_size, test_size, step):
    splitter = TimeSeriesSplitter(mode=mode, train_size=train_size, test_size=test_size, step=step, drop_last=False)
    n_samples = 50
    folds = list(splitter.split(n_samples))
    assert len(folds) > 0
    for train_idx, test_idx in folds:
        assert train_idx.size > 0
        assert test_idx.size > 0
        assert train_idx.max() < test_idx.min()
        assert np.all(train_idx < n_samples)
        assert np.all(test_idx < n_samples)
        assert train_idx.min() >= 0
        assert test_idx.min() >= 0


@pytest.mark.parametrize("mode", ["expanding", "rolling"])
def test_step_lt_test_size_rejected(mode):
    """Policy B: step < test_size must raise ValueError."""
    with pytest.raises(ValueError, match="overlapping"):
        TimeSeriesSplitter(mode=mode, train_size=10, test_size=5, step=2)

def test_leakage_drop_last_true():
    splitter = TimeSeriesSplitter(mode="expanding", train_size=10, test_size=5, step=5, drop_last=True)
    n_samples = 24
    folds = list(splitter.split(n_samples))
    for train_idx, test_idx in folds:
        assert test_idx.size == 5  # No partial folds if drop_last=True

# B) Determinism
def test_determinism():
    n_samples = 40
    splitter = TimeSeriesSplitter("rolling", 10, 5, 5)
    folds1 = list(splitter.split(n_samples))
    folds2 = list(splitter.split(n_samples))

    assert len(folds1) == len(folds2)
    for (tr1, te1), (tr2, te2) in zip(folds1, folds2):
        np.testing.assert_array_equal(tr1, tr2)
        np.testing.assert_array_equal(te1, te2)

    folds3 = list(TimeSeriesSplitter("rolling", 10, 5, 5).split(n_samples))
    for (tr1, te1), (tr3, te3) in zip(folds1, folds3):
        np.testing.assert_array_equal(tr1, tr3)
        np.testing.assert_array_equal(te1, te3)

# C) Fold count & boundary correctness
def test_fold_boundaries_expanding_drop_last():
    # n=20, train=10, test=5, step=5, drop_last=True
    splitter = TimeSeriesSplitter("expanding", 10, 5, 5, drop_last=True)
    folds = list(splitter.split(20))
    assert len(folds) == 2
    tr0, te0 = folds[0]
    np.testing.assert_array_equal(tr0, np.arange(0, 10))
    np.testing.assert_array_equal(te0, np.arange(10, 15))
    
    tr1, te1 = folds[1]
    np.testing.assert_array_equal(tr1, np.arange(0, 15))
    np.testing.assert_array_equal(te1, np.arange(15, 20))

def test_fold_boundaries_partial():
    # n=21, train=10, test=5, step=5
    splitter_drop = TimeSeriesSplitter("expanding", 10, 5, 5, drop_last=True)
    assert len(list(splitter_drop.split(21))) == 2
    
    splitter_keep = TimeSeriesSplitter("expanding", 10, 5, 5, drop_last=False)
    folds = list(splitter_keep.split(21))
    assert len(folds) == 3
    tr2, te2 = folds[2]
    np.testing.assert_array_equal(tr2, np.arange(0, 20))
    np.testing.assert_array_equal(te2, np.array([20]))

# D) Parameter validation
def test_parameter_validation():
    with pytest.raises(ValueError):
        TimeSeriesSplitter("expanding", 0, 5) # type: ignore
    with pytest.raises(ValueError):
        TimeSeriesSplitter("expanding", 10, 0) # type: ignore
    with pytest.raises(ValueError):
        TimeSeriesSplitter("expanding", 10, 5, step=0)
    with pytest.raises(ValueError):
        TimeSeriesSplitter("invalid_mode", 10, 5) # type: ignore
    # step < test_size rejected (Policy B)
    with pytest.raises(ValueError, match="overlapping"):
        TimeSeriesSplitter("expanding", 10, 5, step=3)

    # drop_last=False: n < train_size + 1 rejected
    splitter = TimeSeriesSplitter("expanding", 10, 5, drop_last=False)
    with pytest.raises(ValueError):
        list(splitter.split(10))
    # drop_last=True: n < train_size + test_size rejected (Policy A)
    splitter_dl = TimeSeriesSplitter("expanding", 10, 5, drop_last=True)
    with pytest.raises(ValueError):
        list(splitter_dl.split(14))

def test_docstring_n_samples_validation():
    """FIXED: drop_last=True now requires n >= train_size + test_size (Policy A).

    Previously n=11, train=10, test=5, drop_last=True silently yielded 0 folds.
    Now it fails fast with ValueError.
    """
    splitter = TimeSeriesSplitter("expanding", 10, 5, drop_last=True)
    # n_samples=11 < train_size+test_size=15 → ValueError
    with pytest.raises(ValueError, match="n_samples"):
        list(splitter.split(11))

    # But drop_last=False with same config should yield 1 partial fold
    splitter_keep = TimeSeriesSplitter("expanding", 10, 5, drop_last=False)
    folds = list(splitter_keep.split(11))
    assert len(folds) == 1
    _train, test = folds[0]
    assert len(test) == 1  # only 1 sample past train window

# E) Rolling-specific correctness
def test_rolling_correctness():
    splitter = TimeSeriesSplitter("rolling", train_size=10, test_size=5, step=5, drop_last=False)
    folds = list(splitter.split(25))
    train_starts = []
    for i, (tr, te) in enumerate(folds):
        if i < len(folds) - 1 or te.size == 5:
            assert len(tr) == 10
            assert len(te) == 5
        train_starts.append(tr[0])

    for i in range(1, len(train_starts)):
        assert train_starts[i] - train_starts[i-1] == 5

# F) Expanding-specific correctness
def test_expanding_correctness():
    splitter = TimeSeriesSplitter("expanding", train_size=10, test_size=5, step=5, drop_last=False)
    folds = list(splitter.split(25))
    for i, (tr, te) in enumerate(folds):
        assert tr[0] == 0
        assert len(tr) == 10 + i * 5

# G) Stress / fuzz tests
def test_fuzz_invariants():
    np.random.seed(123)
    random.seed(123)

    for _ in range(200):
        n_samples = random.randint(50, 500)
        train_size = random.randint(5, min(200, n_samples - 2))
        test_size = random.randint(1, min(50, n_samples - train_size))
        # Policy B: step must be >= test_size
        step = random.randint(test_size, test_size * 2)
        drop_last = random.choice([True, False])
        mode = random.choice(["expanding", "rolling"])

        splitter = TimeSeriesSplitter(
            mode=mode, train_size=train_size, test_size=test_size,
            step=step, drop_last=drop_last,
        )

        # Policy A: may raise ValueError for drop_last=True if n too small
        min_required = (train_size + test_size) if drop_last else (train_size + 1)
        if n_samples < min_required:
            with pytest.raises(ValueError):
                list(splitter.split(n_samples))
            continue

        folds = list(splitter.split(n_samples))
        # With Policy A, at least 1 fold is guaranteed
        assert len(folds) > 0

        for tr, te in folds:
            assert tr.size > 0
            if drop_last:
                assert te.size == test_size
            else:
                assert te.size > 0

            assert tr.max() < te.min()
            assert tr.min() >= 0
            assert te.min() >= 0
            assert tr.max() < n_samples
            assert te.max() < n_samples

        if random.random() < 0.1:
            folds_2 = list(splitter.split(n_samples))
            assert len(folds) == len(folds_2)
            for (tr1, te1), (tr2, te2) in zip(folds, folds_2):
                np.testing.assert_array_equal(tr1, tr2)
                np.testing.assert_array_equal(te1, te2)
