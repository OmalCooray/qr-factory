#!/usr/bin/env python
"""Quick demo: print walk-forward fold boundaries for a synthetic series."""

from src.validation import TimeSeriesSplitter


def main() -> None:
    n = 2000

    for mode in ("expanding", "rolling"):
        splitter = TimeSeriesSplitter(
            mode=mode, train_size=500, test_size=100, step=100
        )
        print(f"\n{'='*60}")
        print(f"  {splitter}")
        print(f"  n_samples={n}  ->  {splitter.n_splits(n)} folds")
        print(f"{'='*60}")
        for i, (train, test) in enumerate(splitter.split(n)):
            print(
                f"  Fold {i:>2d}:  "
                f"train[{train[0]:>5d}..{train[-1]:>5d}] ({len(train):>5d})  "
                f"test[{test[0]:>5d}..{test[-1]:>5d}] ({len(test):>4d})"
            )


if __name__ == "__main__":
    main()
