"""Capture git commit hash and dirty flag."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def get_git_info(repo: Path) -> dict:
    """Return git state for the given repository root.

    Returns
    -------
    dict
        ``git_commit`` – HEAD SHA-1 (40-char hex) or ``"unknown"``.
        ``git_dirty``  – *True* when the working tree has uncommitted changes.
    """
    git_commit = "unknown"
    git_dirty = False

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )

        status = (
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        git_dirty = bool(status)
    except (subprocess.CalledProcessError, FileNotFoundError):
        log.warning("Could not read git info — git may not be installed or this is not a repo")

    return {
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
