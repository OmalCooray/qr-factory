"""Directed structure graph with Kosaraju SCC and bias derivation.

Builds a graph of causal relationships between structural levels (BSL, SSL,
FVG, OB) and derives regime (RANGE vs TREND) + directional bias from the
strongly-connected components of the condensation DAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .levels import Level, LevelType


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Edge:
    from_idx: int
    to_idx: int
    edge_type: str
    weight: float = 1.0


@dataclass(frozen=True)
class GraphState:
    """Per-bar snapshot of the structure graph's derived state."""

    regime: str           # "TREND" or "RANGE"
    bias: float           # +1.0 bullish, -1.0 bearish, 0.0 neutral
    bias_strength: float  # 0.0–1.0
    n_active: int         # number of active (non-swept, proximal) levels
    n_sccs: int           # number of SCCs in condensation
    n_edges: int          # number of edges in the graph


# ---------------------------------------------------------------------------
# Helpers: directional classification of level types
# ---------------------------------------------------------------------------

_BULL_TYPES = frozenset({LevelType.BSL, LevelType.FVG_BULL, LevelType.OB_BULL})
_BEAR_TYPES = frozenset({LevelType.SSL, LevelType.FVG_BEAR, LevelType.OB_BEAR})


def _is_bull(lt: LevelType) -> bool:
    return lt in _BULL_TYPES


def _is_bear(lt: LevelType) -> bool:
    return lt in _BEAR_TYPES


# ---------------------------------------------------------------------------
# Kosaraju's SCC (iterative)
# ---------------------------------------------------------------------------

def _kosaraju_sccs(
    n_nodes: int,
    adj: dict[int, list[int]],
    adj_rev: dict[int, list[int]],
) -> list[list[int]]:
    """Compute SCCs using iterative Kosaraju's algorithm.

    Processes nodes in sorted order for determinism.

    Returns a list of SCCs, each SCC being a sorted list of node indices.
    """
    # Pass 1: compute finish order on the forward graph
    visited: set[int] = set()
    finish_order: list[int] = []

    for start in range(n_nodes):
        if start in visited:
            continue
        # Iterative DFS
        stack: list[tuple[int, int]] = [(start, 0)]
        visited.add(start)
        while stack:
            node, idx = stack[-1]
            neighbors = adj.get(node, [])
            if idx < len(neighbors):
                stack[-1] = (node, idx + 1)
                nb = neighbors[idx]
                if nb not in visited:
                    visited.add(nb)
                    stack.append((nb, 0))
            else:
                stack.pop()
                finish_order.append(node)

    # Pass 2: process nodes in reverse finish order on the reversed graph
    visited.clear()
    sccs: list[list[int]] = []

    for start in reversed(finish_order):
        if start in visited:
            continue
        component: list[int] = []
        stack_2: list[int] = [start]
        visited.add(start)
        while stack_2:
            node = stack_2.pop()
            component.append(node)
            for nb in adj_rev.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    stack_2.append(nb)
        component.sort()
        sccs.append(component)

    return sccs


# ---------------------------------------------------------------------------
# StructureGraph
# ---------------------------------------------------------------------------

class StructureGraph:
    """Maintains a rolling set of structural levels, builds edges, runs
    Kosaraju SCC, and derives regime + directional bias.

    Parameters
    ----------
    max_levels : int
        FIFO capacity for stored levels.
    proximity_atr : float
        Only levels within ``proximity_atr * ATR`` of the current price are
        considered active.
    recompute_every : int
        Expensive graph recomputation (edges + SCC) runs every *n* calls to
        ``update()``.  Sweep marking (cheap) still runs every call.  Set to 1
        for exact results on every bar; higher values trade accuracy for speed
        on large datasets.
    """

    def __init__(
        self,
        max_levels: int = 200,
        proximity_atr: float = 3.0,
        recompute_every: int = 1,
    ) -> None:
        self.max_levels = max_levels
        self.proximity_atr = proximity_atr
        self.recompute_every = max(1, recompute_every)
        self._levels: list[Level] = []
        self._update_counter: int = 0
        self._cached_state: GraphState = GraphState(
            regime="RANGE", bias=0.0, bias_strength=0.0,
            n_active=0, n_sccs=0, n_edges=0,
        )
        self._dirty: bool = True  # force first computation

    @property
    def levels(self) -> list[Level]:
        return list(self._levels)

    def add_levels(self, new_levels: Sequence[Level]) -> None:
        """Add levels with FIFO eviction when exceeding ``max_levels``."""
        if new_levels:
            self._dirty = True
        self._levels.extend(new_levels)
        if len(self._levels) > self.max_levels:
            self._levels = self._levels[-self.max_levels:]

    def update(
        self,
        current_price: float,
        atr: float,
        bar_index: int,
    ) -> GraphState:
        """Recompute (or return cached) graph state for the current bar.

        Sweep marking runs every call.  The expensive edge-building + SCC
        pass runs only every ``recompute_every`` calls or when the level set
        has been modified since the last full computation.
        """
        # Always mark swept (cheap O(n) pass)
        swept_count = self._mark_swept(current_price)
        if swept_count > 0:
            self._dirty = True

        self._update_counter += 1

        need_full = self._dirty or (self._update_counter % self.recompute_every == 0)
        if not need_full:
            return self._cached_state

        # Full recompute
        state = self._full_update(current_price, atr)
        self._cached_state = state
        self._dirty = False
        return state

    def _full_update(self, current_price: float, atr: float) -> GraphState:
        """Run the full edge + SCC + bias pipeline."""
        proximity = self.proximity_atr * atr if atr > 0 else float("inf")
        active: list[Level] = []
        for lv in self._levels:
            if lv.swept:
                continue
            mid = (lv.price_low + lv.price_high) / 2.0
            if abs(mid - current_price) <= proximity:
                active.append(lv)

        n_active = len(active)
        if n_active == 0:
            return GraphState(
                regime="RANGE",
                bias=0.0,
                bias_strength=0.0,
                n_active=0,
                n_sccs=0,
                n_edges=0,
            )

        # Build edges
        edges = self._build_edges(active)

        # Kosaraju SCC
        n_nodes = len(active)
        adj: dict[int, list[int]] = {i: [] for i in range(n_nodes)}
        adj_rev: dict[int, list[int]] = {i: [] for i in range(n_nodes)}

        for e in edges:
            adj[e.from_idx].append(e.to_idx)
            adj_rev[e.to_idx].append(e.from_idx)

        for k in adj:
            adj[k].sort()
        for k in adj_rev:
            adj_rev[k].sort()

        sccs = _kosaraju_sccs(n_nodes, adj, adj_rev)

        regime, bias, bias_strength = self._derive_bias(active, sccs)

        return GraphState(
            regime=regime,
            bias=bias,
            bias_strength=bias_strength,
            n_active=n_active,
            n_sccs=len(sccs),
            n_edges=len(edges),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mark_swept(self, current_price: float) -> int:
        """Mark levels as swept if price has traded through them.

        Returns the number of newly swept levels.
        """
        updated: list[Level] = []
        swept_count = 0
        for lv in self._levels:
            if lv.swept:
                updated.append(lv)
                continue
            # BSL swept when price trades above it
            if lv.level_type == LevelType.BSL and current_price > lv.price_high:
                updated.append(lv.mark_swept())
                swept_count += 1
            # SSL swept when price trades below it
            elif lv.level_type == LevelType.SSL and current_price < lv.price_low:
                updated.append(lv.mark_swept())
                swept_count += 1
            else:
                updated.append(lv)
        self._levels = updated
        return swept_count

    def _build_edges(self, active: list[Level]) -> list[Edge]:
        """Build directed edges between active levels.

        Uses type-indexed buckets so only relevant pairs are checked,
        avoiding O(N²) all-pairs comparison.

        Edge rules:
        - ``sequential``: same-direction, older → newer
        - ``defended_by``: SSL → OB_BULL (below), BSL → OB_BEAR (above)
        - ``sweep_to_fvg``: liquidity level → FVG in sweep direction
        """
        edges: list[Edge] = []

        # Index by type → list of (original_index, Level)
        by_type: dict[LevelType, list[tuple[int, Level]]] = {}
        for idx, lv in enumerate(active):
            by_type.setdefault(lv.level_type, []).append((idx, lv))

        # Index by direction → list of (original_index, Level)
        bulls: list[tuple[int, Level]] = []
        bears: list[tuple[int, Level]] = []
        for idx, lv in enumerate(active):
            if _is_bull(lv.level_type):
                bulls.append((idx, lv))
            elif _is_bear(lv.level_type):
                bears.append((idx, lv))

        # Sequential edges: same-direction, older → newer
        for group in (bulls, bears):
            n = len(group)
            if n < 2:
                continue
            # Only connect consecutive pairs (sorted by bar_index) to keep
            # edge count O(N) instead of O(N²) while preserving the chain.
            sorted_group = sorted(group, key=lambda t: t[1].bar_index)
            for k in range(n - 1):
                idx_i, li = sorted_group[k]
                idx_j, lj = sorted_group[k + 1]
                if li.bar_index < lj.bar_index:
                    edges.append(Edge(
                        from_idx=idx_i, to_idx=idx_j,
                        edge_type="sequential",
                        weight=(li.strength + lj.strength) / 2.0,
                    ))

        # Defended_by: SSL → OB_BULL
        for idx_i, li in by_type.get(LevelType.SSL, []):
            for idx_j, lj in by_type.get(LevelType.OB_BULL, []):
                if lj.price_high <= li.price_high:
                    edges.append(Edge(
                        from_idx=idx_i, to_idx=idx_j,
                        edge_type="defended_by", weight=lj.strength,
                    ))

        # Defended_by: BSL → OB_BEAR
        for idx_i, li in by_type.get(LevelType.BSL, []):
            for idx_j, lj in by_type.get(LevelType.OB_BEAR, []):
                if lj.price_low >= li.price_low:
                    edges.append(Edge(
                        from_idx=idx_i, to_idx=idx_j,
                        edge_type="defended_by", weight=lj.strength,
                    ))

        # Sweep_to_fvg: BSL → FVG_BEAR
        for idx_i, li in by_type.get(LevelType.BSL, []):
            for idx_j, lj in by_type.get(LevelType.FVG_BEAR, []):
                if lj.price_high >= li.price_low:
                    edges.append(Edge(
                        from_idx=idx_i, to_idx=idx_j,
                        edge_type="sweep_to_fvg", weight=lj.strength,
                    ))

        # Sweep_to_fvg: SSL → FVG_BULL
        for idx_i, li in by_type.get(LevelType.SSL, []):
            for idx_j, lj in by_type.get(LevelType.FVG_BULL, []):
                if lj.price_low <= li.price_high:
                    edges.append(Edge(
                        from_idx=idx_i, to_idx=idx_j,
                        edge_type="sweep_to_fvg", weight=lj.strength,
                    ))

        return edges

    def _derive_bias(
        self,
        active: list[Level],
        sccs: list[list[int]],
    ) -> tuple[str, float, float]:
        """Derive regime and bias from SCC composition.

        - If any SCC contains both bull and bear levels → RANGE.
        - Otherwise, count directional weight → TREND + bias sign.

        Returns (regime, bias, bias_strength).
        """
        bull_weight = 0.0
        bear_weight = 0.0
        has_mixed_scc = False

        for scc in sccs:
            scc_bull = 0.0
            scc_bear = 0.0
            for node_idx in scc:
                lv = active[node_idx]
                if _is_bull(lv.level_type):
                    scc_bull += lv.strength
                elif _is_bear(lv.level_type):
                    scc_bear += lv.strength

            if scc_bull > 0 and scc_bear > 0:
                has_mixed_scc = True

            bull_weight += scc_bull
            bear_weight += scc_bear

        total_weight = bull_weight + bear_weight
        if total_weight == 0:
            return "RANGE", 0.0, 0.0

        if has_mixed_scc:
            return "RANGE", 0.0, abs(bull_weight - bear_weight) / total_weight

        # Pure directional
        bias = 1.0 if bull_weight > bear_weight else (-1.0 if bear_weight > bull_weight else 0.0)
        bias_strength = abs(bull_weight - bear_weight) / total_weight

        if bias == 0.0:
            return "RANGE", 0.0, 0.0

        return "TREND", bias, bias_strength
