"""Tests for the robustness campaign harness.

Verifies scenario expansion, config patching, manifest I/O,
and dry-run orchestration wiring — NOT strategy performance.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from src.robustness.spec import CampaignSpec, RobustnessScenario
from src.robustness.expand import (
    apply_overrides,
    build_scenario_configs,
    deep_merge,
    expand_grid_scenarios,
    load_campaign_spec,
    _dotpath_to_nested,
)
from src.robustness.manifest import read_manifest, write_manifest


# ══════════════════════════════════════════════════════════════════════════════
# deep_merge
# ══════════════════════════════════════════════════════════════════════════════


class TestDeepMerge:

    def test_flat_override(self):
        base = {"a": 1, "b": 2}
        result = deep_merge(base, {"b": 99})
        assert result == {"a": 1, "b": 99}

    def test_nested_merge(self):
        base = {"execution": {"spread_points": 40, "slippage_points": 10}}
        overrides = {"execution": {"spread_points": 80}}
        result = deep_merge(base, overrides)
        assert result == {"execution": {"spread_points": 80, "slippage_points": 10}}

    def test_deep_nested_merge(self):
        base = {"strategy": {"policy": {"p_enter": 0.4, "p_exit": 0.3}}}
        overrides = {"strategy": {"policy": {"p_enter": 0.5}}}
        result = deep_merge(base, overrides)
        assert result["strategy"]["policy"]["p_enter"] == 0.5
        assert result["strategy"]["policy"]["p_exit"] == 0.3

    def test_add_new_key(self):
        base = {"a": 1}
        result = deep_merge(base, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_does_not_mutate_base(self):
        base = {"execution": {"spread_points": 40}}
        deep_merge(base, {"execution": {"spread_points": 80}})
        assert base["execution"]["spread_points"] == 40

    def test_replace_non_dict_with_dict(self):
        """If base has a scalar and override has a dict, replace."""
        base = {"x": 42}
        result = deep_merge(base, {"x": {"nested": True}})
        assert result == {"x": {"nested": True}}

    def test_replace_dict_with_scalar(self):
        """If base has a dict and override has a scalar, replace."""
        base = {"x": {"nested": True}}
        result = deep_merge(base, {"x": 42})
        assert result == {"x": 42}

    def test_empty_overrides(self):
        base = {"a": 1, "b": {"c": 3}}
        result = deep_merge(base, {})
        assert result == base


# ══════════════════════════════════════════════════════════════════════════════
# _dotpath_to_nested
# ══════════════════════════════════════════════════════════════════════════════


class TestDotpathToNested:

    def test_single_key(self):
        assert _dotpath_to_nested("spread", 80) == {"spread": 80}

    def test_two_levels(self):
        assert _dotpath_to_nested("execution.spread_points", 80) == {
            "execution": {"spread_points": 80}
        }

    def test_three_levels(self):
        assert _dotpath_to_nested("strategy.policy.p_enter", 0.5) == {
            "strategy": {"policy": {"p_enter": 0.5}}
        }


# ══════════════════════════════════════════════════════════════════════════════
# apply_overrides
# ══════════════════════════════════════════════════════════════════════════════


class TestApplyOverrides:

    def test_basic_override(self):
        base = {
            "symbol": "XAUUSD",
            "execution": {"spread_points": 40, "slippage_points": 10},
        }
        overrides = {"execution": {"spread_points": 80}}
        result = apply_overrides(base, overrides)
        assert result["execution"]["spread_points"] == 80
        assert result["execution"]["slippage_points"] == 10
        assert result["symbol"] == "XAUUSD"


# ══════════════════════════════════════════════════════════════════════════════
# expand_grid_scenarios
# ══════════════════════════════════════════════════════════════════════════════


class TestExpandGrid:

    def test_single_key_grid(self):
        scenarios = expand_grid_scenarios(
            family="parameter_sweep",
            name_prefix="p_enter",
            description="Sweep",
            grid={"strategy.policy.p_enter": [0.35, 0.45, 0.55]},
        )
        assert len(scenarios) == 3
        assert scenarios[0].scenario_id == "parameter_sweep__p_enter__p_enter_0.35"
        assert scenarios[1].scenario_id == "parameter_sweep__p_enter__p_enter_0.45"
        assert scenarios[2].scenario_id == "parameter_sweep__p_enter__p_enter_0.55"

    def test_grid_overrides_correct(self):
        scenarios = expand_grid_scenarios(
            family="sweep",
            name_prefix="test",
            description="",
            grid={"execution.spread_points": [40, 80]},
        )
        assert scenarios[0].overrides == {"execution": {"spread_points": 40}}
        assert scenarios[1].overrides == {"execution": {"spread_points": 80}}

    def test_multi_key_grid_cartesian(self):
        """Multi-key grid produces Cartesian product."""
        scenarios = expand_grid_scenarios(
            family="sweep",
            name_prefix="multi",
            description="",
            grid={
                "strategy.policy.p_enter": [0.4, 0.5],
                "execution.spread_points": [40, 80],
            },
        )
        assert len(scenarios) == 4  # 2 x 2

    def test_deterministic_order(self):
        """Same input → same output order."""
        for _ in range(5):
            result = expand_grid_scenarios(
                family="sweep",
                name_prefix="det",
                description="",
                grid={"a.b": [1, 2], "c.d": [3, 4]},
            )
            ids = [s.scenario_id for s in result]
            assert ids == [
                "sweep__det__b_1__d_3",
                "sweep__det__b_1__d_4",
                "sweep__det__b_2__d_3",
                "sweep__det__b_2__d_4",
            ]

    def test_tags_propagated(self):
        scenarios = expand_grid_scenarios(
            family="sweep",
            name_prefix="t",
            description="",
            grid={"x": [1]},
            tags=("stress", "w8"),
        )
        assert scenarios[0].tags == ("stress", "w8")

    def test_all_scenarios_enabled_by_default(self):
        scenarios = expand_grid_scenarios(
            family="sweep",
            name_prefix="t",
            description="",
            grid={"x": [1, 2]},
        )
        assert all(s.enabled for s in scenarios)


# ══════════════════════════════════════════════════════════════════════════════
# Scenario ID uniqueness
# ══════════════════════════════════════════════════════════════════════════════


class TestScenarioIDUniqueness:

    def test_direct_scenarios_unique(self):
        scenarios = [
            RobustnessScenario(
                scenario_id="cost_stress__2x", family="cost_stress",
                name="2x", description="", overrides={},
            ),
            RobustnessScenario(
                scenario_id="cost_stress__3x", family="cost_stress",
                name="3x", description="", overrides={},
            ),
        ]
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids))

    def test_grid_scenarios_unique(self):
        scenarios = expand_grid_scenarios(
            family="sweep",
            name_prefix="p",
            description="",
            grid={"a": [1, 2, 3], "b": [4, 5]},
        )
        ids = [s.scenario_id for s in scenarios]
        assert len(ids) == len(set(ids))


# ══════════════════════════════════════════════════════════════════════════════
# load_campaign_spec
# ══════════════════════════════════════════════════════════════════════════════


class TestLoadCampaignSpec:

    def test_load_direct_and_grid(self, tmp_path: Path):
        spec = {
            "campaign_id": "test_campaign",
            "baseline_config": "configs/test.yaml",
            "description": "Test",
            "scenarios": [
                {
                    "family": "cost_stress",
                    "name": "2x",
                    "description": "Double spread",
                    "overrides": {"execution": {"spread_points": 80}},
                },
                {
                    "family": "parameter_sweep",
                    "name_prefix": "p",
                    "description": "Sweep",
                    "grid": {"strategy.policy.p_enter": [0.3, 0.5]},
                },
            ],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        campaign = load_campaign_spec(str(spec_path))
        assert campaign.campaign_id == "test_campaign"
        assert len(campaign.scenarios) == 3  # 1 direct + 2 grid
        assert campaign.scenarios[0].scenario_id == "cost_stress__2x"
        assert campaign.scenarios[1].family == "parameter_sweep"

    def test_disabled_scenarios_preserved(self, tmp_path: Path):
        spec = {
            "campaign_id": "test",
            "baseline_config": "test.yaml",
            "scenarios": [
                {
                    "family": "cost_stress",
                    "name": "disabled_one",
                    "description": "Should be disabled",
                    "overrides": {},
                    "enabled": False,
                },
            ],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        campaign = load_campaign_spec(str(spec_path))
        assert len(campaign.scenarios) == 1
        assert not campaign.scenarios[0].enabled

    def test_baseline_run_id_optional(self, tmp_path: Path):
        spec = {
            "campaign_id": "test",
            "baseline_config": "test.yaml",
            "baseline_run_id": "20260307_120000",
            "scenarios": [],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        campaign = load_campaign_spec(str(spec_path))
        assert campaign.baseline_run_id == "20260307_120000"

    def test_tags_loaded(self, tmp_path: Path):
        spec = {
            "campaign_id": "test",
            "baseline_config": "test.yaml",
            "scenarios": [
                {
                    "family": "cost_stress",
                    "name": "tagged",
                    "description": "",
                    "overrides": {},
                    "tags": ["stress", "w8"],
                },
            ],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        campaign = load_campaign_spec(str(spec_path))
        assert campaign.scenarios[0].tags == ("stress", "w8")


# ══════════════════════════════════════════════════════════════════════════════
# build_scenario_configs
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildScenarioConfigs:

    def _make_campaign(self) -> CampaignSpec:
        return CampaignSpec(
            campaign_id="test",
            baseline_config="unused",
            description="Test",
            scenarios=(
                RobustnessScenario(
                    scenario_id="cost__2x",
                    family="cost_stress",
                    name="2x",
                    description="Double spread",
                    overrides={"execution": {"spread_points": 80}},
                ),
                RobustnessScenario(
                    scenario_id="cost__disabled",
                    family="cost_stress",
                    name="disabled",
                    description="Disabled",
                    overrides={"execution": {"spread_points": 200}},
                    enabled=False,
                ),
            ),
        )

    def test_applies_overrides(self):
        campaign = self._make_campaign()
        base = {"execution": {"spread_points": 40, "slippage_points": 10}}
        results = build_scenario_configs(campaign, baseline_config_override=base)

        assert len(results) == 1  # disabled scenario excluded
        scenario, config = results[0]
        assert config["execution"]["spread_points"] == 80
        assert config["execution"]["slippage_points"] == 10

    def test_disabled_excluded(self):
        campaign = self._make_campaign()
        base = {"execution": {"spread_points": 40}}
        results = build_scenario_configs(campaign, baseline_config_override=base)
        ids = [s.scenario_id for s, _ in results]
        assert "cost__disabled" not in ids

    def test_base_config_not_mutated(self):
        campaign = CampaignSpec(
            campaign_id="test",
            baseline_config="unused",
            description="",
            scenarios=(
                RobustnessScenario(
                    scenario_id="s1",
                    family="test",
                    name="s1",
                    description="",
                    overrides={"x": 99},
                ),
            ),
        )
        base = {"x": 1, "y": 2}
        build_scenario_configs(campaign, baseline_config_override=base)
        assert base["x"] == 1  # not mutated


# ══════════════════════════════════════════════════════════════════════════════
# Manifest I/O
# ══════════════════════════════════════════════════════════════════════════════


class TestManifestIO:

    def test_round_trip(self, tmp_path: Path):
        manifest = {
            "campaign_id": "test",
            "baseline_run_id": "20260307_120000",
            "scenarios": {
                "cost__2x": {"run_id": "20260307_120001", "status": "completed"},
            },
        }
        write_manifest(tmp_path, manifest)
        loaded = read_manifest(tmp_path)
        assert loaded == manifest

    def test_creates_directory(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c"
        write_manifest(nested, {"test": True})
        assert (nested / "MANIFEST.json").exists()


# ══════════════════════════════════════════════════════════════════════════════
# Dry-run scaffold
# ══════════════════════════════════════════════════════════════════════════════


class TestDryRunScaffold:
    """Verify dry-run writes scenario configs + manifest without running backtests."""

    def test_dry_run_writes_files(self, tmp_path: Path):
        """Dry run creates scenario YAMLs, MANIFEST.json, and no run dirs."""
        # Create a baseline config
        baseline = {
            "symbol": "XAUUSD",
            "timeframe": "M5",
            "snapshot_dir": "data/test",
            "starting_capital": 1000.0,
            "output_dir": "runs",
            "execution": {"spread_points": 40, "slippage_points": 10, "point_value": 0.01},
            "strategy": {"type": "ma_crossover", "fast": 10, "slow": 50},
        }
        baseline_path = tmp_path / "baseline.yaml"
        baseline_path.write_text(yaml.dump(baseline), encoding="utf-8")

        # Create a campaign spec
        spec = {
            "campaign_id": "dry_test",
            "baseline_config": str(baseline_path),
            "description": "Dry run test",
            "scenarios": [
                {
                    "family": "cost_stress",
                    "name": "2x_spread",
                    "description": "Double spread",
                    "overrides": {"execution": {"spread_points": 80}},
                },
                {
                    "family": "latency_proxy",
                    "name": "high_slip",
                    "description": "More slippage",
                    "overrides": {"execution": {"slippage_points": 30}},
                },
            ],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        # Import and run
        from src.robustness.campaign import run_campaign

        campaign_dir = run_campaign(str(spec_path), dry_run=True)

        # Verify manifest
        manifest_path = campaign_dir / "MANIFEST.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["campaign_id"] == "dry_test"
        assert manifest["dry_run"] is True
        assert len(manifest["scenarios"]) == 2
        for entry in manifest["scenarios"].values():
            assert entry["status"] == "pending"
            assert entry["run_id"] is None

        # Verify scenario configs written
        scenarios_dir = campaign_dir / "scenarios"
        assert (scenarios_dir / "cost_stress__2x_spread.yaml").exists()
        assert (scenarios_dir / "latency_proxy__high_slip.yaml").exists()

        # Verify overrides applied in scenario config
        resolved = yaml.safe_load(
            (scenarios_dir / "cost_stress__2x_spread.yaml").read_text(encoding="utf-8")
        )
        assert resolved["execution"]["spread_points"] == 80
        assert resolved["execution"]["slippage_points"] == 10  # preserved from baseline

    def test_dry_run_grid_expansion(self, tmp_path: Path):
        """Grid scenarios are expanded and written in dry-run mode."""
        baseline = {
            "strategy": {"policy": {"p_enter": 0.4, "p_exit": 0.3}},
            "execution": {"spread_points": 40},
        }
        baseline_path = tmp_path / "baseline.yaml"
        baseline_path.write_text(yaml.dump(baseline), encoding="utf-8")

        spec = {
            "campaign_id": "grid_test",
            "baseline_config": str(baseline_path),
            "scenarios": [
                {
                    "family": "parameter_sweep",
                    "name_prefix": "p_enter",
                    "description": "Sweep",
                    "grid": {"strategy.policy.p_enter": [0.35, 0.45, 0.55]},
                },
            ],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        from src.robustness.campaign import run_campaign

        campaign_dir = run_campaign(str(spec_path), dry_run=True)
        manifest = json.loads(
            (campaign_dir / "MANIFEST.json").read_text(encoding="utf-8")
        )
        assert len(manifest["scenarios"]) == 3

        # Verify each override is correct
        for val in [0.35, 0.45, 0.55]:
            sid = f"parameter_sweep__p_enter__p_enter_{val}"
            cfg_path = campaign_dir / "scenarios" / f"{sid}.yaml"
            assert cfg_path.exists(), f"Missing config for {sid}"
            resolved = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            assert resolved["strategy"]["policy"]["p_enter"] == val
            assert resolved["strategy"]["policy"]["p_exit"] == 0.3  # baseline preserved


# ══════════════════════════════════════════════════════════════════════════════
# Expansion determinism
# ══════════════════════════════════════════════════════════════════════════════


class TestExpansionDeterminism:
    """Same spec → same scenarios in same order, every time."""

    def test_repeated_expansion_identical(self, tmp_path: Path):
        baseline = {"x": 1, "y": {"z": 2}}
        baseline_path = tmp_path / "baseline.yaml"
        baseline_path.write_text(yaml.dump(baseline), encoding="utf-8")

        spec = {
            "campaign_id": "det",
            "baseline_config": str(baseline_path),
            "scenarios": [
                {"family": "a", "name": "s1", "overrides": {"x": 10}},
                {
                    "family": "b",
                    "name_prefix": "g",
                    "grid": {"y.z": [3, 4, 5]},
                },
            ],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        results = []
        for _ in range(5):
            campaign = load_campaign_spec(str(spec_path))
            configs = build_scenario_configs(campaign, baseline_config_override=baseline)
            ids = [s.scenario_id for s, _ in configs]
            results.append(ids)

        assert all(r == results[0] for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# Frozen dataclass contracts
# ══════════════════════════════════════════════════════════════════════════════


class TestFrozenDataclasses:

    def test_scenario_immutable(self):
        s = RobustnessScenario(
            scenario_id="test", family="f", name="n",
            description="d", overrides={"a": 1},
        )
        with pytest.raises(AttributeError):
            s.family = "changed"  # type: ignore[misc]

    def test_campaign_immutable(self):
        c = CampaignSpec(
            campaign_id="test", baseline_config="x",
            description="d", scenarios=(),
        )
        with pytest.raises(AttributeError):
            c.campaign_id = "changed"  # type: ignore[misc]
