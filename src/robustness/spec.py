"""Data structures for robustness scenario specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobustnessScenario:
    """A single robustness stress scenario.

    Attributes
    ----------
    scenario_id : str
        Unique identifier, typically ``{family}__{name}``.
    family : str
        Scenario family (e.g. cost_stress, latency_proxy, parameter_sweep).
    name : str
        Human-readable scenario name.
    description : str
        What this scenario tests.
    overrides : dict
        Nested dict of config overrides to deep-merge onto the baseline.
    tags : tuple[str, ...]
        Optional metadata tags for filtering / grouping.
    enabled : bool
        Whether to include this scenario in campaign execution.
    """

    scenario_id: str
    family: str
    name: str
    description: str
    overrides: dict
    tags: tuple[str, ...] = ()
    enabled: bool = True


@dataclass(frozen=True)
class CampaignSpec:
    """Specification for a robustness campaign.

    Attributes
    ----------
    campaign_id : str
        Unique campaign identifier.
    baseline_config : str
        Path to the baseline YAML config (relative to repo root or absolute).
    description : str
        Human-readable campaign description.
    scenarios : tuple[RobustnessScenario, ...]
        Ordered list of scenarios to run.
    baseline_run_id : str | None
        If provided, reference an existing baseline run instead of re-running.
    """

    campaign_id: str
    baseline_config: str
    description: str
    scenarios: tuple[RobustnessScenario, ...]
    baseline_run_id: str | None = None
