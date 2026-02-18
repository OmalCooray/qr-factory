"""Append-only audit log for data snapshots."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

_HEADER = "# Data Version Log"


class VersionLog:
    """Append-only audit log stored in ``DATA_VERSION.md``."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(
        self,
        snapshot_id: str,
        symbol: str,
        broker: str,
        range_start: str,
        range_end: str,
    ) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        entry = (
            f"{today} [{snapshot_id}]: Exported {symbol} M5 from MT5 terminal "
            f"<{broker}>, range {range_start}..{range_end}."
        )

        # Read existing content (strip BOM if present via utf-8-sig)
        if self._path.exists() and self._path.stat().st_size > 0:
            existing = self._path.read_text(encoding="utf-8-sig").rstrip("\r\n")
        else:
            existing = _HEADER

        # Build the full file: header, blank line, then all entries
        lines = existing.split("\n")
        # Ensure header is first line
        if not lines[0].startswith("#"):
            lines.insert(0, _HEADER)
        lines.append(entry)

        # Write back (no BOM, consistent utf-8)
        self._path.write_text(
            "\n".join(lines) + "\n", encoding="utf-8",
        )

        log.info("Updated %s", self._path)
