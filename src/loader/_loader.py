"""Loader orchestrator — coordinates source, writer, and version log."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ._config import LoaderConfig
from ._protocols import DataSource, DataWriter
from ._version_log import VersionLog

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"  # …/qr-factory/data


class Loader:
    """Coordinates source -> writer -> version-log for a single snapshot."""

    def __init__(
        self,
        source: DataSource,
        writer: DataWriter,
        version_log: VersionLog,
    ) -> None:
        self._source = source
        self._writer = writer
        self._version_log = version_log

    # -- Public API --------------------------------------------------------

    def run(self, cfg: LoaderConfig) -> Path | None:
        """Execute the full download pipeline.  Returns the snapshot dir."""
        snapshot_id = self._generate_snapshot_id()
        snapshot_dir = DATA_DIR / snapshot_id

        log.info("Symbol   : %s", cfg.symbol)
        log.info("Range    : %s .. %s", cfg.start, cfg.end)
        log.info("Snapshot : %s", snapshot_id)
        log.info("Output   : %s", snapshot_dir)

        self._source.connect(cfg)
        try:
            broker = self._source.broker_name()
            log.info("Broker   : %s", broker)

            total, first, last = self._fetch_all(cfg, snapshot_dir)

            if total > 0:
                range_start = first.strftime("%Y-%m-%d") if first else cfg.start
                range_end   = last.strftime("%Y-%m-%d")  if last  else cfg.end
                self._write_meta(
                    snapshot_dir, snapshot_id, cfg,
                    broker, range_start, range_end, total,
                )
                self._version_log.append(
                    snapshot_id, cfg.symbol, broker, range_start, range_end,
                )
                log.info("Done — %s rows written to %s", f"{total:,}", snapshot_dir)
                return snapshot_dir
            else:
                log.warning("No data was downloaded.")
                if snapshot_dir.exists() and not any(snapshot_dir.iterdir()):
                    snapshot_dir.rmdir()
                return None
        finally:
            self._source.disconnect()

    # -- Private helpers ---------------------------------------------------

    def _fetch_all(
        self, cfg: LoaderConfig, out_dir: Path,
    ) -> tuple[int, datetime | None, datetime | None]:
        out_dir.mkdir(parents=True, exist_ok=True)

        first_bar = self._source.find_first_bar(
            cfg.symbol, cfg.start_dt, cfg.timeframe,
            cfg.max_empty_advances, cfg.pause_sec,
        )
        if first_bar is None:
            log.warning("No history found after %s", cfg.start)
            return 0, None, None

        chunk = timedelta(days=cfg.days_per_chunk)
        cur_start = first_bar
        end_date = cfg.end_dt
        total_rows = 0
        windows = 0
        actual_first: datetime | None = None
        actual_last: datetime | None = None

        while cur_start < end_date:
            cur_end = min(cur_start + chunk, end_date)

            df = self._source.fetch_range(
                cfg.symbol, cfg.timeframe, cur_start, cur_end,
            )

            if df.empty:
                cur_start = cur_end + timedelta(seconds=1)
                continue

            self._writer.write(df, cfg.symbol, out_dir)

            if actual_first is None:
                actual_first = df["time"].iat[0].to_pydatetime()
            actual_last = df["time"].iat[-1].to_pydatetime()

            total_rows += len(df)
            windows += 1

            cur_start = df["time"].iat[-1].to_pydatetime() + timedelta(seconds=1)

            if windows % 5 == 0:
                log.info(
                    "  [%d windows]  rows so far: %s",
                    windows, f"{total_rows:,}",
                )

            time.sleep(cfg.pause_sec)

        return total_rows, actual_first, actual_last

    @staticmethod
    def _generate_snapshot_id() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def _write_meta(
        snapshot_dir: Path,
        snapshot_id: str,
        cfg: LoaderConfig,
        broker: str,
        range_start: str,
        range_end: str,
        total_rows: int,
    ) -> None:
        meta = {
            "snapshot_id": snapshot_id,
            "symbol": cfg.symbol,
            "timeframe": "M5",
            "broker": broker,
            "range_start": range_start,
            "range_end": range_end,
            "total_rows": total_rows,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta_path = snapshot_dir / "META.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        log.info("Wrote %s", meta_path)


# -- Convenience entry-point ----------------------------------------------

def main(argv: list[str] | None = None) -> None:
    """CLI entry-point: configure logging, build components, run."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    cfg = LoaderConfig.from_cli(argv)

    from ._csv_writer import CsvYearlyWriter
    from ._mt5_source import MT5Source

    version_file = DATA_DIR / "DATA_VERSION.md"

    loader = Loader(
        source=MT5Source(),
        writer=CsvYearlyWriter(),
        version_log=VersionLog(version_file),
    )
    loader.run(cfg)
