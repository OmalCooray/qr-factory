"""Per-year CSV writer with atomic file operations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from ._protocols import EXPECTED_COLS


class CsvYearlyWriter:
    """Write bar data into per-year CSV files with atomic renames."""

    def write(self, df: pd.DataFrame, symbol: str, out_dir: Path) -> None:
        if df.empty:
            return

        df = df.loc[:, EXPECTED_COLS].drop_duplicates(subset=["time"])
        df = df.sort_values("time")

        for year, part in df.groupby(df["time"].dt.year):
            part = part.sort_values("time")
            out_path = out_dir / f"{symbol}_M5_{year}.csv"

            if out_path.exists():
                existing = pd.read_csv(out_path, parse_dates=["time"])
                combined = pd.concat([existing, part]).drop_duplicates(subset=["time"])
                combined = combined.sort_values("time")
                self._atomic_write_csv(combined, out_path)
            else:
                self._atomic_write_csv(part, out_path)

    @staticmethod
    def _atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
        """Write *df* to a temp file then rename — crash-safe."""
        fd, tmp = tempfile.mkstemp(
            dir=target.parent, suffix=".tmp", prefix=target.stem,
        )
        try:
            os.close(fd)
            df.to_csv(tmp, index=False)
            os.replace(tmp, target)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
