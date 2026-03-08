"""Centralized repository path resolution.

Single source of truth for repo root — eliminates duplicate path logic.
"""

from pathlib import Path

# Canonical repository root (parent of src/)
REPO_ROOT = Path(__file__).resolve().parent.parent
