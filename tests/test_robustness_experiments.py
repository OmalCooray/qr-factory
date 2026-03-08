"""Tests for Week 8 robustness experiment families.

Covers: cost stress, latency proxy, parameter sweeps, regime slicing,
and machine-readable summary generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.robustness.expand import (
    build_scenario_configs,
    deep_merge,
    load_campaign_spec,
)
from src.robustness.regimes import (
    aggregate_slice_metrics,
    assign_session,
    compute_volatility_regimes,
)
from src.robustness.spec import CampaignSpec, RobustnessScenario
from src.robustness.summarize import (
    build_campaign_summary,
    build_scenario_metrics_table,
)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


_BASELINE_CONFIG = {
    "symbol": "XAUUSD",
    "timeframe": "M5",
    "snapshot_dir": "data/test",
    "starting_capital": 1000.0,
    "output_dir": "runs",
    "execution": {
        "spread_points": 40,
        "slippage_points": 10,
        "point_value": 0.01,
    },
    "strategy": {
        "type": "ml_prob_threshold",
        "policy": {
            "type": "fixed_threshold",
            "p_enter": 0.40,
            "p_exit": 0.30,
            "min_hold_bars": 5,
        },
    },
}


def _make_trades_df(n: int = 20, seed: int = 42) -> pd.DataFrame:
    """Synthetic trades for regime slicing tests."""
    rng = np.random.RandomState(seed)
    base_ts = pd.Timestamp("2025-06-01", tz="UTC")
    entries = [base_ts + pd.Timedelta(hours=i * 4) for i in range(n)]
    pnl = rng.randn(n) * 10
    gross_pnl = pnl + rng.rand(n) * 2  # gross > net
    return pd.DataFrame({
        "entry_ts": [t.isoformat() for t in entries],
        "exit_ts": [(t + pd.Timedelta(hours=1)).isoformat() for t in entries],
        "side": "long",
        "qty": 1.0,
        "entry_price": 2000.0,
        "exit_price": 2001.0,
        "pnl": pnl,
        "gross_pnl": gross_pnl,
    })


# ══════════════════════════════════════════════════════════════════════════════
# A. COST STRESS — scenario generation
# ══════════════════════════════════════════════════════════════════════════════


class TestCostStressScenarios:

    def test_cost_multiplier_overrides(self):
        """Cost stress scenarios apply correct spread/slippage values."""
        campaign = CampaignSpec(
            campaign_id="cost_test",
            baseline_config="unused",
            description="",
            scenarios=(
                RobustnessScenario(
                    scenario_id="cost_stress__0.5x",
                    family="cost_stress",
                    name="0.5x",
                    description="Half cost",
                    overrides={"execution": {"spread_points": 20, "slippage_points": 5}},
                ),
                RobustnessScenario(
                    scenario_id="cost_stress__2.0x",
                    family="cost_stress",
                    name="2.0x",
                    description="Double cost",
                    overrides={"execution": {"spread_points": 80, "slippage_points": 20}},
                ),
            ),
        )
        configs = build_scenario_configs(campaign, baseline_config_override=_BASELINE_CONFIG)

        # 0.5x
        _, cfg_half = configs[0]
        assert cfg_half["execution"]["spread_points"] == 20
        assert cfg_half["execution"]["slippage_points"] == 5
        assert cfg_half["execution"]["point_value"] == 0.01  # preserved

        # 2.0x
        _, cfg_double = configs[1]
        assert cfg_double["execution"]["spread_points"] == 80
        assert cfg_double["execution"]["slippage_points"] == 20

    def test_cost_stress_preserves_strategy(self):
        """Cost overrides don't change strategy/policy."""
        campaign = CampaignSpec(
            campaign_id="cost_test",
            baseline_config="unused",
            description="",
            scenarios=(
                RobustnessScenario(
                    scenario_id="cost_stress__2x",
                    family="cost_stress",
                    name="2x",
                    description="",
                    overrides={"execution": {"spread_points": 80}},
                ),
            ),
        )
        configs = build_scenario_configs(campaign, baseline_config_override=_BASELINE_CONFIG)
        _, cfg = configs[0]
        assert cfg["strategy"]["policy"]["p_enter"] == 0.40
        assert cfg["strategy"]["policy"]["type"] == "fixed_threshold"

    def test_cost_stress_deterministic_ids(self):
        """Same spec → same scenario IDs every time."""
        results = []
        for _ in range(5):
            campaign = CampaignSpec(
                campaign_id="det",
                baseline_config="unused",
                description="",
                scenarios=(
                    RobustnessScenario(
                        scenario_id="cost_stress__0.5x",
                        family="cost_stress",
                        name="0.5x",
                        description="",
                        overrides={"execution": {"spread_points": 20}},
                    ),
                    RobustnessScenario(
                        scenario_id="cost_stress__2.0x",
                        family="cost_stress",
                        name="2.0x",
                        description="",
                        overrides={"execution": {"spread_points": 80}},
                    ),
                ),
            )
            configs = build_scenario_configs(campaign, baseline_config_override=_BASELINE_CONFIG)
            results.append([s.scenario_id for s, _ in configs])
        assert all(r == results[0] for r in results)


# ══════════════════════════════════════════════════════════════════════════════
# B. LATENCY PROXY — scenario generation + engine wiring
# ══════════════════════════════════════════════════════════════════════════════


class TestLatencyProxy:

    def test_latency_override_applied(self):
        """Latency scenario adds latency_bars to execution config."""
        campaign = CampaignSpec(
            campaign_id="latency_test",
            baseline_config="unused",
            description="",
            scenarios=(
                RobustnessScenario(
                    scenario_id="latency_proxy__1bar",
                    family="latency_proxy",
                    name="1bar",
                    description="1-bar delay",
                    overrides={"execution": {"latency_bars": 1}},
                ),
            ),
        )
        configs = build_scenario_configs(campaign, baseline_config_override=_BASELINE_CONFIG)
        _, cfg = configs[0]
        assert cfg["execution"]["latency_bars"] == 1
        assert cfg["execution"]["spread_points"] == 40  # preserved

    def test_engine_latency_delays_execution(self):
        """With latency_bars=1, intent executes 2 bars later instead of 1."""
        from src.engine.core import TradingEngine
        from src.execution.costs import CostConfig
        from src.risk import RiskConfig, RiskManager
        from src.strategy.signal import Signal

        class GoLongOnce:
            name = "go_long_once"
            warmup_bars = 0
            _called = 0

            def on_bar(self, ctx):
                self._called += 1
                if self._called == 1:
                    return Signal(direction=1.0, strength=1.0, reason="enter")
                return Signal(direction=0.0, strength=0.0, reason="flat")

        rm = RiskManager(RiskConfig(), 10000.0)

        # No latency — intent from bar 0 executes at bar 1
        s0 = GoLongOnce()
        e0 = TradingEngine(s0, rm, 10000.0, latency_bars=0)

        bars = pd.DataFrame({
            "time": pd.date_range("2025-01-01", periods=5, freq="5min", tz="UTC"),
            "open": [100.0] * 5,
            "high": [101.0] * 5,
            "low": [99.0] * 5,
            "close": [100.0] * 5,
            "volume": [1000] * 5,
        })
        features = pd.DataFrame({"f1": [0.0] * 5})

        for i in range(5):
            e0.process_bar(bars.iloc[i], features.iloc[i], i, is_last=(i == 4))

        # With latency_bars=0, position should be entered at bar 1
        positions_0 = [r["position"] for r in e0.equity_records]
        assert positions_0[0] == 0.0  # bar 0: no intent yet
        assert positions_0[1] == 1.0  # bar 1: intent from bar 0 executes

        # With latency_bars=1 — intent from bar 0 executes at bar 2
        rm1 = RiskManager(RiskConfig(), 10000.0)
        s1 = GoLongOnce()
        e1 = TradingEngine(s1, rm1, 10000.0, latency_bars=1)

        for i in range(5):
            e1.process_bar(bars.iloc[i], features.iloc[i], i, is_last=(i == 4))

        positions_1 = [r["position"] for r in e1.equity_records]
        assert positions_1[0] == 0.0  # bar 0: no intent
        assert positions_1[1] == 0.0  # bar 1: intent queued but not old enough
        assert positions_1[2] == 1.0  # bar 2: intent from bar 0 now executes

    def test_engine_default_latency_zero(self):
        """Default latency_bars=0 preserves standard 1-bar execution delay."""
        from src.engine.core import TradingEngine
        from src.risk import RiskConfig, RiskManager

        rm = RiskManager(RiskConfig(), 10000.0)

        class Stub:
            name = "stub"
            warmup_bars = 0
            def on_bar(self, ctx):
                from src.strategy.signal import Signal
                return Signal(direction=0.0, strength=0.0, reason="stub")

        engine = TradingEngine(Stub(), rm, 10000.0)
        assert engine._latency_bars == 0


# ══════════════════════════════════════════════════════════════════════════════
# C. BOUNDED PARAMETER SWEEPS
# ══════════════════════════════════════════════════════════════════════════════


class TestParameterSweeps:

    def test_p_enter_sweep_generates_correct_configs(self):
        """p_enter grid produces configs with correct overrides."""
        campaign = CampaignSpec(
            campaign_id="sweep_test",
            baseline_config="unused",
            description="",
            scenarios=tuple(
                RobustnessScenario(
                    scenario_id=f"parameter_sweep__p_enter__{v}",
                    family="parameter_sweep",
                    name=f"p_enter_{v}",
                    description="",
                    overrides={"strategy": {"policy": {"p_enter": v}}},
                )
                for v in [0.35, 0.45]
            ),
        )
        configs = build_scenario_configs(campaign, baseline_config_override=_BASELINE_CONFIG)
        assert len(configs) == 2
        _, cfg_low = configs[0]
        _, cfg_high = configs[1]
        assert cfg_low["strategy"]["policy"]["p_enter"] == 0.35
        assert cfg_high["strategy"]["policy"]["p_enter"] == 0.45
        # Other policy params preserved
        assert cfg_low["strategy"]["policy"]["p_exit"] == 0.30
        assert cfg_low["strategy"]["policy"]["min_hold_bars"] == 5

    def test_min_hold_sweep(self):
        """min_hold_bars sweep applies correctly."""
        campaign = CampaignSpec(
            campaign_id="sweep_test",
            baseline_config="unused",
            description="",
            scenarios=tuple(
                RobustnessScenario(
                    scenario_id=f"parameter_sweep__min_hold__{v}",
                    family="parameter_sweep",
                    name=f"min_hold_{v}",
                    description="",
                    overrides={"strategy": {"policy": {"min_hold_bars": v}}},
                )
                for v in [3, 7]
            ),
        )
        configs = build_scenario_configs(campaign, baseline_config_override=_BASELINE_CONFIG)
        _, cfg_3 = configs[0]
        _, cfg_7 = configs[1]
        assert cfg_3["strategy"]["policy"]["min_hold_bars"] == 3
        assert cfg_7["strategy"]["policy"]["min_hold_bars"] == 7

    def test_sweep_preserves_execution_config(self):
        """Parameter sweeps don't touch execution config."""
        campaign = CampaignSpec(
            campaign_id="sweep_test",
            baseline_config="unused",
            description="",
            scenarios=(
                RobustnessScenario(
                    scenario_id="sweep__p_enter",
                    family="parameter_sweep",
                    name="p_enter_0.5",
                    description="",
                    overrides={"strategy": {"policy": {"p_enter": 0.50}}},
                ),
            ),
        )
        configs = build_scenario_configs(campaign, baseline_config_override=_BASELINE_CONFIG)
        _, cfg = configs[0]
        assert cfg["execution"]["spread_points"] == 40
        assert cfg["execution"]["slippage_points"] == 10

    def test_sweep_from_yaml(self, tmp_path: Path):
        """Grid sweep in YAML produces expected scenario set."""
        baseline_path = tmp_path / "baseline.yaml"
        baseline_path.write_text(yaml.dump(_BASELINE_CONFIG), encoding="utf-8")

        spec = {
            "campaign_id": "sweep_yaml",
            "baseline_config": str(baseline_path),
            "scenarios": [
                {
                    "family": "parameter_sweep",
                    "name_prefix": "p_enter",
                    "description": "Entry sweep",
                    "grid": {"strategy.policy.p_enter": [0.35, 0.45]},
                },
            ],
        }
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(spec), encoding="utf-8")

        campaign = load_campaign_spec(str(spec_path))
        assert len(campaign.scenarios) == 2
        configs = build_scenario_configs(campaign)
        assert configs[0][1]["strategy"]["policy"]["p_enter"] == 0.35
        assert configs[1][1]["strategy"]["policy"]["p_enter"] == 0.45

    def test_probsized_p_full_sweep(self):
        """p_full sweep (probsized-specific) applies correctly."""
        probsized_base = deep_merge(_BASELINE_CONFIG, {
            "strategy": {"policy": {
                "type": "probability_sized",
                "p_full": 0.50,
                "min_size": 0.25,
                "max_size": 1.0,
            }},
        })
        campaign = CampaignSpec(
            campaign_id="pfull_test",
            baseline_config="unused",
            description="",
            scenarios=tuple(
                RobustnessScenario(
                    scenario_id=f"sweep__p_full_{v}",
                    family="parameter_sweep",
                    name=f"p_full_{v}",
                    description="",
                    overrides={"strategy": {"policy": {"p_full": v}}},
                )
                for v in [0.45, 0.55]
            ),
        )
        configs = build_scenario_configs(campaign, baseline_config_override=probsized_base)
        _, cfg_45 = configs[0]
        _, cfg_55 = configs[1]
        assert cfg_45["strategy"]["policy"]["p_full"] == 0.45
        assert cfg_55["strategy"]["policy"]["p_full"] == 0.55
        # Other params preserved
        assert cfg_45["strategy"]["policy"]["p_enter"] == 0.40
        assert cfg_45["strategy"]["policy"]["min_size"] == 0.25


# ══════════════════════════════════════════════════════════════════════════════
# D. REGIME SLICING
# ══════════════════════════════════════════════════════════════════════════════


class TestVolatilityRegimes:

    def test_regime_labels_valid(self):
        """Output contains only expected labels."""
        close = pd.Series(np.random.RandomState(42).randn(500).cumsum() + 2000)
        regimes = compute_volatility_regimes(close, window=50)
        assert set(regimes.unique()) <= {"warmup", "low_vol", "high_vol"}

    def test_warmup_bars_labelled(self):
        """First `window` bars are labelled warmup."""
        close = pd.Series(np.random.RandomState(42).randn(200).cumsum() + 2000)
        regimes = compute_volatility_regimes(close, window=50)
        # First 50 bars should be warmup (rolling window needs 50 points + 1 for diff)
        assert (regimes.iloc[:50] == "warmup").all()

    def test_regime_deterministic(self):
        """Same input → same regime assignments."""
        close = pd.Series(np.random.RandomState(42).randn(300).cumsum() + 2000)
        r1 = compute_volatility_regimes(close, window=50)
        r2 = compute_volatility_regimes(close, window=50)
        assert (r1 == r2).all()

    def test_median_split_balanced(self):
        """Roughly equal number of low_vol and high_vol bars (after warmup)."""
        close = pd.Series(np.random.RandomState(42).randn(1000).cumsum() + 2000)
        regimes = compute_volatility_regimes(close, window=50)
        valid = regimes[regimes != "warmup"]
        counts = valid.value_counts()
        # Should be roughly 50/50 (within 10% tolerance)
        assert abs(counts.get("low_vol", 0) - counts.get("high_vol", 0)) < len(valid) * 0.1


class TestSessionSlicing:

    def test_session_assignment(self):
        """Correct session assignment for known UTC hours."""
        timestamps = pd.Series([
            "2025-06-01T02:00:00+00:00",  # Asia (hour 2)
            "2025-06-01T10:00:00+00:00",  # London (hour 10)
            "2025-06-01T20:00:00+00:00",  # New York (hour 20)
        ])
        labels = assign_session(timestamps)
        assert labels.iloc[0] == "asia"
        assert labels.iloc[1] == "london"
        assert labels.iloc[2] == "new_york"

    def test_session_boundaries(self):
        """Boundary hours assigned correctly."""
        timestamps = pd.Series([
            "2025-06-01T00:00:00+00:00",  # Asia boundary
            "2025-06-01T07:59:00+00:00",  # Still Asia
            "2025-06-01T08:00:00+00:00",  # London boundary
            "2025-06-01T15:59:00+00:00",  # Still London
            "2025-06-01T16:00:00+00:00",  # New York boundary
            "2025-06-01T23:59:00+00:00",  # Still New York
        ])
        labels = assign_session(timestamps)
        assert list(labels) == ["asia", "asia", "london", "london", "new_york", "new_york"]

    def test_session_deterministic(self):
        """Same timestamps → same session labels."""
        ts = pd.Series(["2025-06-01T05:00:00+00:00", "2025-06-01T12:00:00+00:00"])
        l1 = assign_session(ts)
        l2 = assign_session(ts)
        assert (l1 == l2).all()


class TestSliceAggregation:

    def test_per_slice_metrics(self):
        """Aggregate metrics computed correctly per group."""
        trades = _make_trades_df(20)
        labels = pd.Series(["A"] * 10 + ["B"] * 10)
        result = aggregate_slice_metrics(trades, labels, "group")

        assert "A" in result
        assert "B" in result
        assert result["A"]["n_trades"] == 10
        assert result["B"]["n_trades"] == 10

    def test_metrics_fields(self):
        """Each slice has all expected metric fields."""
        trades = _make_trades_df(10)
        labels = pd.Series(["X"] * 10)
        result = aggregate_slice_metrics(trades, labels, "group")

        expected_keys = {
            "n_trades", "gross_pnl", "net_pnl", "cost_pnl",
            "avg_cost_per_trade", "win_rate", "avg_trade_pnl",
        }
        assert set(result["X"].keys()) == expected_keys

    def test_pnl_consistency(self):
        """gross_pnl - net_pnl == cost_pnl."""
        trades = _make_trades_df(15)
        labels = pd.Series(["Z"] * 15)
        result = aggregate_slice_metrics(trades, labels, "group")
        z = result["Z"]
        assert abs(z["gross_pnl"] - z["net_pnl"] - z["cost_pnl"]) < 1e-4

    def test_deterministic_aggregation(self):
        """Same trades + labels → same metrics."""
        trades = _make_trades_df(20)
        labels = pd.Series(["A"] * 10 + ["B"] * 10)
        r1 = aggregate_slice_metrics(trades, labels, "g")
        r2 = aggregate_slice_metrics(trades, labels, "g")
        assert r1 == r2


# ══════════════════════════════════════════════════════════════════════════════
# E. MACHINE-READABLE SUMMARIES
# ══════════════════════════════════════════════════════════════════════════════


class TestCampaignSummary:

    def _setup_campaign_dir(self, tmp_path: Path) -> Path:
        """Create a minimal campaign dir with manifest and mock runs."""
        campaign_dir = tmp_path / "campaign"
        campaign_dir.mkdir()

        # Create mock run directories with evaluation.json
        runs_dir = tmp_path / "runs"
        for run_id in ("run_001", "run_002"):
            run_dir = runs_dir / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "evaluation.json").write_text(json.dumps({
                "run_id": run_id,
                "strategy_type": "test",
                "prediction_metrics": {"brier": 0.25, "roc_auc": 0.55},
                "trading_metrics": {
                    "total_net_pnl": 50.0 if run_id == "run_001" else -10.0,
                    "total_gross_pnl": 60.0,
                    "total_cost_pnl": 10.0,
                    "sharpe_per_bar": 0.001,
                    "max_drawdown_pct": 5.0,
                    "num_trades": 100,
                    "win_rate": 0.52,
                    "avg_trade_pnl": 0.5,
                    "profit_factor": 1.2,
                    "trades_per_day": 5.0,
                },
                "calibration_metrics": {},
                "metadata": {},
            }), encoding="utf-8")

        # Write manifest
        manifest = {
            "campaign_id": "test_campaign",
            "baseline_config": "configs/test.yaml",
            "baseline_run_id": "run_baseline",
            "description": "Test campaign",
            "dry_run": False,
            "scenarios": {
                "cost_stress__2x": {
                    "family": "cost_stress",
                    "name": "2x",
                    "description": "Double cost",
                    "tags": [],
                    "status": "completed",
                    "run_id": "run_001",
                },
                "latency__1bar": {
                    "family": "latency_proxy",
                    "name": "1bar",
                    "description": "1-bar delay",
                    "tags": [],
                    "status": "completed",
                    "run_id": "run_002",
                },
            },
        }
        (campaign_dir / "MANIFEST.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8",
        )

        return campaign_dir

    def test_campaign_summary_written(self, tmp_path: Path, monkeypatch):
        """campaign_summary.json is produced with correct structure."""
        campaign_dir = self._setup_campaign_dir(tmp_path)
        monkeypatch.setattr(
            "src.robustness.summarize.REPO_ROOT", tmp_path,
        )

        path = build_campaign_summary(campaign_dir)
        assert path.exists()

        summary = json.loads(path.read_text(encoding="utf-8"))
        assert summary["campaign_id"] == "test_campaign"
        assert summary["n_scenarios"] == 2
        assert summary["n_completed"] == 2
        assert "cost_stress" in summary["families"]
        assert "latency_proxy" in summary["families"]
        assert "cost_stress__2x" in summary["scenario_results"]

    def test_scenario_metrics_populated(self, tmp_path: Path, monkeypatch):
        """Per-scenario key metrics are populated from evaluation.json."""
        campaign_dir = self._setup_campaign_dir(tmp_path)
        monkeypatch.setattr(
            "src.robustness.summarize.REPO_ROOT", tmp_path,
        )

        build_campaign_summary(campaign_dir)
        summary = json.loads(
            (campaign_dir / "campaign_summary.json").read_text(encoding="utf-8")
        )
        metrics = summary["scenario_results"]["cost_stress__2x"]["key_metrics"]
        assert metrics["total_net_pnl"] == 50.0
        assert metrics["sharpe_per_bar"] == 0.001
        assert metrics["num_trades"] == 100

    def test_scenario_metrics_csv(self, tmp_path: Path, monkeypatch):
        """scenario_metrics.csv has one row per completed scenario."""
        campaign_dir = self._setup_campaign_dir(tmp_path)
        monkeypatch.setattr(
            "src.robustness.summarize.REPO_ROOT", tmp_path,
        )

        path = build_scenario_metrics_table(campaign_dir)
        assert path.exists()

        df = pd.read_csv(path)
        assert len(df) == 2
        assert "scenario_id" in df.columns
        assert "total_net_pnl" in df.columns
        assert set(df["family"]) == {"cost_stress", "latency_proxy"}


# ══════════════════════════════════════════════════════════════════════════════
# F. CAMPAIGN CONFIG LOADING — canonical w8 configs parse correctly
# ══════════════════════════════════════════════════════════════════════════════


class TestCanonicalCampaignConfigs:

    def test_fixed_campaign_loads(self):
        """w8_robustness_fixed.yaml loads and expands correctly."""
        campaign = load_campaign_spec("configs/w8_robustness_fixed.yaml")
        assert campaign.campaign_id == "w8_fixed"
        assert campaign.baseline_config == "configs/w7_policy_fixed.yaml"

        families = {s.family for s in campaign.scenarios}
        assert "cost_stress" in families
        assert "latency_proxy" in families
        assert "parameter_sweep" in families

        # Count: 3 cost + 1 latency + 2 p_enter + 2 p_exit + 2 min_hold = 10
        assert len(campaign.scenarios) == 10

    def test_probsized_campaign_loads(self):
        """w8_robustness_probsized.yaml loads and expands correctly."""
        campaign = load_campaign_spec("configs/w8_robustness_probsized.yaml")
        assert campaign.campaign_id == "w8_probsized"
        assert campaign.baseline_config == "configs/w7_policy_probsized.yaml"

        families = {s.family for s in campaign.scenarios}
        assert "cost_stress" in families
        assert "latency_proxy" in families
        assert "parameter_sweep" in families

        # Count: 3 cost + 1 latency + 2 p_enter + 2 p_full + 2 min_hold = 10
        assert len(campaign.scenarios) == 10

    def test_fixed_scenarios_differ_only_in_intended_variable(self):
        """Each scenario in the fixed campaign changes only the intended config key."""
        campaign = load_campaign_spec("configs/w8_robustness_fixed.yaml")
        configs = build_scenario_configs(campaign)

        for scenario, cfg in configs:
            # All scenarios must preserve the baseline strategy type
            assert cfg["strategy"]["type"] == "ml_prob_threshold"
            assert cfg["strategy"]["policy"]["type"] == "fixed_threshold"
            # All scenarios must preserve data/model/label/calibration
            assert cfg["symbol"] == "XAUUSD"
            assert cfg["model"]["type"] == "logistic"
            assert cfg["label"]["type"] == "return_sign_cost_aware"
            assert cfg["calibration"]["method"] == "platt"

    def test_probsized_scenarios_preserve_baseline(self):
        """Probsized campaign scenarios preserve non-overridden baseline values."""
        campaign = load_campaign_spec("configs/w8_robustness_probsized.yaml")
        configs = build_scenario_configs(campaign)

        for scenario, cfg in configs:
            assert cfg["strategy"]["policy"]["type"] == "probability_sized"
            assert cfg["model"]["type"] == "logistic"
            assert cfg["calibration"]["method"] == "platt"


# ══════════════════════════════════════════════════════════════════════════════
# G. DRY-RUN WITH FULL CAMPAIGN SPEC
# ══════════════════════════════════════════════════════════════════════════════


class TestDryRunFullCampaign:

    def test_dry_run_fixed_campaign(self, tmp_path: Path):
        """Dry-run of the fixed campaign writes all expected scenario configs."""
        # Create a baseline config at the expected location
        baseline_path = tmp_path / "configs" / "w7_policy_fixed.yaml"
        baseline_path.parent.mkdir(parents=True, exist_ok=True)

        import shutil
        shutil.copy2(
            Path(__file__).resolve().parent.parent / "configs" / "w7_policy_fixed.yaml",
            baseline_path,
        )

        # Create campaign spec pointing to the tmp baseline
        campaign_spec = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "configs" / "w8_robustness_fixed.yaml")
            .read_text(encoding="utf-8")
        )
        campaign_spec["baseline_config"] = str(baseline_path)
        spec_path = tmp_path / "campaign.yaml"
        spec_path.write_text(yaml.dump(campaign_spec), encoding="utf-8")

        from src.robustness.campaign import run_campaign
        campaign_dir = run_campaign(str(spec_path), dry_run=True)

        # Verify manifest
        manifest = json.loads(
            (campaign_dir / "MANIFEST.json").read_text(encoding="utf-8")
        )
        assert manifest["dry_run"] is True
        assert len(manifest["scenarios"]) == 10

        # All scenarios should have pending status
        for entry in manifest["scenarios"].values():
            assert entry["status"] == "pending"

        # Verify scenario config files exist
        scenarios_dir = campaign_dir / "scenarios"
        yamls = list(scenarios_dir.glob("*.yaml"))
        assert len(yamls) == 10
