"""Campaign manifest I/O."""

from __future__ import annotations

import json
from pathlib import Path


def write_manifest(campaign_dir: Path, manifest: dict) -> Path:
    """Write MANIFEST.json to the campaign directory.

    Returns the path to the written file.
    """
    campaign_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = campaign_dir / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )
    return manifest_path


def read_manifest(campaign_dir: Path) -> dict:
    """Read MANIFEST.json from a campaign directory."""
    manifest_path = campaign_dir / "MANIFEST.json"
    return json.loads(manifest_path.read_text(encoding="utf-8"))
