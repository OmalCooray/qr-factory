"""Scenario expansion and config patching utilities."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml

from src.robustness.spec import CampaignSpec, RobustnessScenario

# Repo root (three levels up from src/robustness/expand.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base*.  Returns a new dict.

    Dict values are merged recursively; all other types are replaced.
    """
    result = dict(base)
    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _dotpath_to_nested(path: str, value: Any) -> dict:
    """Convert ``'a.b.c'`` + value into ``{'a': {'b': {'c': value}}}``.

    Used to convert grid sweep dot-paths into nested override dicts.
    """
    parts = path.split(".")
    result: dict = {}
    current = result
    for part in parts[:-1]:
        current[part] = {}
        current = current[part]
    current[parts[-1]] = value
    return result


def apply_overrides(base_config: dict, overrides: dict) -> dict:
    """Apply scenario overrides to a base config via deep merge."""
    return deep_merge(base_config, overrides)


def expand_grid_scenarios(
    family: str,
    name_prefix: str,
    description: str,
    grid: dict[str, list],
    tags: tuple[str, ...] = (),
) -> list[RobustnessScenario]:
    """Expand a grid specification into individual scenarios.

    Parameters
    ----------
    family : str
        Scenario family name.
    name_prefix : str
        Prefix for generated scenario names.
    description : str
        Base description (augmented with parameter values).
    grid : dict[str, list]
        Mapping of dot-path config keys to lists of values.
        E.g. ``{"strategy.policy.p_enter": [0.35, 0.45, 0.55]}``.
    tags : tuple[str, ...]
        Tags inherited by all expanded scenarios.

    Returns
    -------
    list[RobustnessScenario]
        Deterministically ordered list (sorted grid keys, then product order).
    """
    keys = sorted(grid.keys())
    value_lists = [grid[k] for k in keys]
    scenarios: list[RobustnessScenario] = []

    for combo in itertools.product(*value_lists):
        name_parts: list[str] = []
        combined_overrides: dict = {}
        for key, val in zip(keys, combo):
            short_key = key.split(".")[-1]
            name_parts.append(f"{short_key}_{val}")
            nested = _dotpath_to_nested(key, val)
            combined_overrides = deep_merge(combined_overrides, nested)

        suffix = "__".join(name_parts)
        name = f"{name_prefix}__{suffix}"
        scenario_id = f"{family}__{name}"

        scenarios.append(
            RobustnessScenario(
                scenario_id=scenario_id,
                family=family,
                name=name,
                description=f"{description} ({suffix})",
                overrides=combined_overrides,
                tags=tags,
            )
        )

    return scenarios


def load_campaign_spec(spec_path: str | Path) -> CampaignSpec:
    """Load a campaign specification from a YAML file.

    Supports two scenario entry formats:

    1. **Direct** — has ``name`` and ``overrides`` keys.
    2. **Grid** — has ``name_prefix`` and ``grid`` keys (expanded via
       :func:`expand_grid_scenarios`).
    """
    spec_path = Path(spec_path)
    if not spec_path.is_absolute():
        spec_path = _REPO_ROOT / spec_path

    with open(spec_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    scenarios: list[RobustnessScenario] = []

    for entry in raw.get("scenarios", []):
        family = entry["family"]
        enabled = entry.get("enabled", True)
        tags = tuple(entry.get("tags", []))

        if "grid" in entry:
            grid_scenarios = expand_grid_scenarios(
                family=family,
                name_prefix=entry["name_prefix"],
                description=entry.get("description", ""),
                grid=entry["grid"],
                tags=tags,
            )
            if not enabled:
                grid_scenarios = [
                    RobustnessScenario(
                        scenario_id=s.scenario_id,
                        family=s.family,
                        name=s.name,
                        description=s.description,
                        overrides=s.overrides,
                        tags=s.tags,
                        enabled=False,
                    )
                    for s in grid_scenarios
                ]
            scenarios.extend(grid_scenarios)
        else:
            name = entry["name"]
            scenario_id = f"{family}__{name}"
            scenarios.append(
                RobustnessScenario(
                    scenario_id=scenario_id,
                    family=family,
                    name=name,
                    description=entry.get("description", ""),
                    overrides=entry.get("overrides", {}),
                    tags=tags,
                    enabled=enabled,
                )
            )

    return CampaignSpec(
        campaign_id=raw["campaign_id"],
        baseline_config=raw["baseline_config"],
        description=raw.get("description", ""),
        scenarios=tuple(scenarios),
        baseline_run_id=raw.get("baseline_run_id"),
    )


def build_scenario_configs(
    campaign: CampaignSpec,
    baseline_config_override: dict | None = None,
) -> list[tuple[RobustnessScenario, dict]]:
    """Build resolved configs for each enabled scenario.

    Loads the baseline config once, then applies each scenario's overrides.

    Parameters
    ----------
    campaign : CampaignSpec
        The campaign specification.
    baseline_config_override : dict | None
        If provided, use this dict as the baseline config instead of
        loading from ``campaign.baseline_config``.  Useful for testing.

    Returns
    -------
    list[tuple[RobustnessScenario, dict]]
        ``(scenario, resolved_config)`` tuples in scenario order.
    """
    if baseline_config_override is not None:
        base_config = baseline_config_override
    else:
        baseline_path = Path(campaign.baseline_config)
        if not baseline_path.is_absolute():
            baseline_path = _REPO_ROOT / baseline_path

        with open(baseline_path, "r", encoding="utf-8") as f:
            base_config = yaml.safe_load(f)

    results: list[tuple[RobustnessScenario, dict]] = []
    for scenario in campaign.scenarios:
        if not scenario.enabled:
            continue
        resolved = apply_overrides(base_config, scenario.overrides)
        results.append((scenario, resolved))

    return results
