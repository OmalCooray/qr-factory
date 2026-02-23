"""Signal — output of the model/strategy layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """What the model believes — NOT an execution instruction.

    The decision layer converts Signal → OrderIntent based on risk and sizing.

    Attributes
    ----------
    direction : float
        +1.0 long, -1.0 short, 0.0 flat.
    strength : float
        0.0–1.0 confidence (for future sizing models).
    reason : str
        Human-readable audit trail.
    """

    direction: float  # +1.0 long, -1.0 short, 0.0 flat
    strength: float = 1.0  # 0.0–1.0 confidence
    reason: str = ""
