"""Tests for StructureGraph, Kosaraju SCC, and bias derivation."""

import pytest

from src.structure.graph import (
    Edge,
    GraphState,
    StructureGraph,
    _kosaraju_sccs,
)
from src.structure.levels import Level, LevelType


class TestKosarajuSCC:
    def test_single_cycle(self):
        # 0 → 1 → 2 → 0
        adj = {0: [1], 1: [2], 2: [0]}
        adj_rev = {0: [2], 1: [0], 2: [1]}
        sccs = _kosaraju_sccs(3, adj, adj_rev)
        assert len(sccs) == 1
        assert sorted(sccs[0]) == [0, 1, 2]

    def test_two_separate_cycles(self):
        # Cycle 1: 0 → 1 → 0;  Cycle 2: 2 → 3 → 2
        adj = {0: [1], 1: [0], 2: [3], 3: [2]}
        adj_rev = {0: [1], 1: [0], 2: [3], 3: [2]}
        sccs = _kosaraju_sccs(4, adj, adj_rev)
        assert len(sccs) == 2
        scc_sets = [frozenset(s) for s in sccs]
        assert frozenset({0, 1}) in scc_sets
        assert frozenset({2, 3}) in scc_sets

    def test_dag_singletons(self):
        # 0 → 1 → 2 (no cycles → 3 singleton SCCs)
        adj = {0: [1], 1: [2], 2: []}
        adj_rev = {0: [], 1: [0], 2: [1]}
        sccs = _kosaraju_sccs(3, adj, adj_rev)
        assert len(sccs) == 3
        for scc in sccs:
            assert len(scc) == 1

    def test_determinism(self):
        adj = {0: [1, 2], 1: [0], 2: [0]}
        adj_rev = {0: [1, 2], 1: [0], 2: [0]}
        result1 = _kosaraju_sccs(3, adj, adj_rev)
        result2 = _kosaraju_sccs(3, adj, adj_rev)
        assert result1 == result2

    def test_disconnected_nodes(self):
        adj = {0: [], 1: [], 2: []}
        adj_rev = {0: [], 1: [], 2: []}
        sccs = _kosaraju_sccs(3, adj, adj_rev)
        assert len(sccs) == 3


class TestStructureGraphFIFO:
    def test_eviction_at_max_levels(self):
        g = StructureGraph(max_levels=3)
        levels = [
            Level(bar_index=i, level_type=LevelType.BSL,
                  price_low=100.0 + i, price_high=100.0 + i)
            for i in range(5)
        ]
        g.add_levels(levels)
        assert len(g.levels) == 3
        # Should keep the last 3
        assert g.levels[0].bar_index == 2
        assert g.levels[2].bar_index == 4

    def test_no_eviction_under_limit(self):
        g = StructureGraph(max_levels=10)
        levels = [
            Level(bar_index=i, level_type=LevelType.SSL,
                  price_low=100.0, price_high=100.0)
            for i in range(5)
        ]
        g.add_levels(levels)
        assert len(g.levels) == 5


class TestSweepMarking:
    def test_bsl_swept_by_price_above(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        bsl = Level(bar_index=0, level_type=LevelType.BSL,
                    price_low=100.0, price_high=100.0)
        g.add_levels([bsl])
        # Price above BSL → swept
        state = g.update(current_price=101.0, atr=1.0, bar_index=5)
        # The internal level should be swept now
        assert g.levels[0].swept is True

    def test_ssl_swept_by_price_below(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        ssl = Level(bar_index=0, level_type=LevelType.SSL,
                    price_low=100.0, price_high=100.0)
        g.add_levels([ssl])
        # Price below SSL → swept
        state = g.update(current_price=99.0, atr=1.0, bar_index=5)
        assert g.levels[0].swept is True

    def test_not_swept_when_price_not_through(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        bsl = Level(bar_index=0, level_type=LevelType.BSL,
                    price_low=100.0, price_high=100.0)
        g.add_levels([bsl])
        state = g.update(current_price=99.0, atr=1.0, bar_index=5)
        assert g.levels[0].swept is False


class TestActiveFiltering:
    def test_swept_excluded(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        bsl = Level(bar_index=0, level_type=LevelType.BSL,
                    price_low=100.0, price_high=100.0)
        g.add_levels([bsl])
        # Sweep it
        g.update(current_price=101.0, atr=1.0, bar_index=1)
        # Now update again — swept level should be excluded from active
        state = g.update(current_price=101.0, atr=1.0, bar_index=2)
        assert state.n_active == 0

    def test_far_levels_excluded(self):
        g = StructureGraph(max_levels=100, proximity_atr=2.0)
        far_level = Level(bar_index=0, level_type=LevelType.FVG_BULL,
                          price_low=50.0, price_high=55.0)
        near_level = Level(bar_index=1, level_type=LevelType.FVG_BULL,
                           price_low=99.0, price_high=101.0)
        g.add_levels([far_level, near_level])
        # ATR=1.0, proximity=2.0 → only levels within 2.0 of price=100
        state = g.update(current_price=100.0, atr=1.0, bar_index=5)
        assert state.n_active == 1


class TestEdgeCreation:
    def test_sequential_same_direction(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        l1 = Level(bar_index=0, level_type=LevelType.BSL,
                   price_low=100.0, price_high=100.0)
        l2 = Level(bar_index=5, level_type=LevelType.BSL,
                   price_low=102.0, price_high=102.0)
        g.add_levels([l1, l2])
        state = g.update(current_price=99.0, atr=1.0, bar_index=10)
        assert state.n_edges >= 1

    def test_defended_by_ssl_ob_bull(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        ssl = Level(bar_index=0, level_type=LevelType.SSL,
                    price_low=100.0, price_high=100.0)
        ob = Level(bar_index=1, level_type=LevelType.OB_BULL,
                   price_low=98.0, price_high=99.0)  # OB below SSL
        g.add_levels([ssl, ob])
        state = g.update(current_price=101.0, atr=1.0, bar_index=5)
        assert state.n_edges >= 1


class TestBiasDerivation:
    def test_all_bull_levels_trend_positive(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        levels = [
            Level(bar_index=i, level_type=LevelType.BSL,
                  price_low=100.0, price_high=100.0)
            for i in range(3)
        ]
        g.add_levels(levels)
        state = g.update(current_price=99.0, atr=1.0, bar_index=10)
        assert state.regime == "TREND"
        assert state.bias == 1.0

    def test_all_bear_levels_trend_negative(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        levels = [
            Level(bar_index=i, level_type=LevelType.SSL,
                  price_low=100.0, price_high=100.0)
            for i in range(3)
        ]
        g.add_levels(levels)
        state = g.update(current_price=101.0, atr=1.0, bar_index=10)
        assert state.regime == "TREND"
        assert state.bias == -1.0

    def test_no_active_levels_range(self):
        g = StructureGraph(max_levels=100, proximity_atr=100.0)
        state = g.update(current_price=100.0, atr=1.0, bar_index=0)
        assert state.regime == "RANGE"
        assert state.bias == 0.0
        assert state.n_active == 0

    def test_graph_state_frozen(self):
        gs = GraphState(regime="TREND", bias=1.0, bias_strength=0.8,
                        n_active=5, n_sccs=2, n_edges=3)
        with pytest.raises(AttributeError):
            gs.regime = "RANGE"  # type: ignore[misc]
