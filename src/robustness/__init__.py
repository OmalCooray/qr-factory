"""Robustness campaign harness for Week 8 stress testing.

Provides scenario specification, expansion, orchestration,
regime slicing, summary generation, promotion gating, and report generation.
"""

from src.robustness.spec import CampaignSpec, RobustnessScenario

__all__ = [
    "CampaignSpec",
    "RobustnessScenario",
]
