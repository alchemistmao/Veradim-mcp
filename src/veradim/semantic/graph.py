"""Join graph built from ``dimensions.yaml`` and fan-out analysis.

Every declared join is an edge in both directions; the reverse edge gets the
flipped cardinality. A path is *safe* when no edge along it is one_to_many in
the direction of travel: joining along it can never duplicate rows of the
measure. A dimension reachable only through a one_to_many edge needs a
``fanout_allocation`` rule, otherwise it is rejected.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from veradim.errors import FanoutJoinRejected, SchemaViolation
from veradim.semantic.models import Cardinality, Dimension, DimensionCatalog
from veradim.semantic.sqlexpr import referenced_tables


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    on: tuple[tuple[str, str], ...]
    cardinality: Cardinality

    def condition(self) -> str:
        return " AND ".join(f"{self.src}.{a} = {self.dst}.{b}" for a, b in self.on)

    def describe(self) -> str:
        return f"{self.src} -> {self.dst} ({self.cardinality.value})"


@dataclass(frozen=True)
class DirectReach:
    """Dimension reachable from the measure grain through safe joins only."""

    edges: tuple[Edge, ...]


@dataclass(frozen=True)
class AllocatedReach:
    """Dimension reached across a one_to_many join, split by allocation weights."""

    at_grain: str
    at_table: str
    at_key: tuple[str, ...]
    prefix_edges: tuple[Edge, ...]  # measure grain table -> allocation grain table (safe)
    suffix_edges: tuple[Edge, ...]  # allocation grain table -> dimension tables (may fan out)
    weight: str
    missing_label: str


@dataclass
class JoinGraph:
    catalog: DimensionCatalog
    _adj: dict[str, list[Edge]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for join in self.catalog.joins:
            fwd = Edge(join.from_table, join.to_table, tuple(join.on), join.cardinality)
            rev = Edge(
                join.to_table,
                join.from_table,
                tuple((b, a) for a, b in join.on),
                join.cardinality.reversed(),
            )
            self._adj.setdefault(fwd.src, []).append(fwd)
            self._adj.setdefault(rev.src, []).append(rev)

    @property
    def tables(self) -> set[str]:
        names = set(self._adj)
        names |= {g.table for g in self.catalog.grains.values()}
        names |= {d.table for d in self.catalog.dimensions.values()}
        return names

    def path(self, src: str, dst: str) -> tuple[Edge, ...]:
        """Shortest join path (BFS, declaration order breaks ties)."""
        if src == dst:
            return ()
        prev: dict[str, Edge] = {}
        seen = {src}
        queue = deque([src])
        while queue:
            node = queue.popleft()
            for edge in self._adj.get(node, []):
                if edge.dst in seen:
                    continue
                seen.add(edge.dst)
                prev[edge.dst] = edge
                if edge.dst == dst:
                    out = []
                    cur = dst
                    while cur != src:
                        out.append(prev[cur])
                        cur = prev[cur].src
                    return tuple(reversed(out))
                queue.append(edge.dst)
        raise SchemaViolation(
            f"no join path from {src!r} to {dst!r}",
            hint="Declare the join in metrics/dimensions.yaml.",
        )

    def safe_path(self, src: str, dst: str, *, context: str) -> tuple[Edge, ...]:
        path = self.path(src, dst)
        fan = [e for e in path if e.cardinality.fans_out]
        if fan:
            raise FanoutJoinRejected(
                f"{context}: reaching {dst!r} from {src!r} crosses a one_to_many join "
                f"({fan[0].describe()}), which would multiply rows of the measure",
                hint="Only use columns reachable through many_to_one or one_to_one joins.",
                details={"path": [e.describe() for e in path]},
            )
        return path

    def dimension_tables(self, dim_name: str, dim: Dimension) -> set[str]:
        tables = {dim.table} | referenced_tables(dim.expression, context=f"dimension {dim_name}")
        if dim.fanout_allocation:
            tables.add(dim.fanout_allocation.weight.split(".")[0])
        return tables

    def reach_dimension(self, grain_table: str, dim_name: str) -> DirectReach | AllocatedReach:
        dim = self.catalog.dimensions.get(dim_name)
        if dim is None:
            raise SchemaViolation(f"unknown dimension {dim_name!r}")
        tables = {dim.table} | referenced_tables(dim.expression, context=f"dimension {dim_name}")
        paths = {t: self.path(grain_table, t) for t in sorted(tables)}
        fanning = {t: p for t, p in paths.items() if any(e.cardinality.fans_out for e in p)}
        if not fanning:
            return DirectReach(edges=merge_paths(paths.values()))

        alloc = dim.fanout_allocation
        if alloc is None:
            path = next(iter(fanning.values()))
            edge = next(e for e in path if e.cardinality.fans_out)
            raise FanoutJoinRejected(
                f"dimension {dim_name!r} is reachable from grain table {grain_table!r} only through "
                f"a one_to_many join ({edge.describe()}); grouping by it would inflate the measure",
                hint=f"Declare fanout_allocation for {dim_name!r} in dimensions.yaml, or use a "
                "dimension reachable through many_to_one joins.",
                details={"path": [e.describe() for e in path]},
            )
        grain = self.catalog.grains.get(alloc.at)
        if grain is None:
            raise SchemaViolation(f"dimension {dim_name!r}: fanout_allocation.at {alloc.at!r} is not a grain")
        prefix = self.safe_path(grain_table, grain.table, context=f"dimension {dim_name} allocation")
        weight_table = alloc.weight.split(".")[0]
        suffix = merge_paths(self.path(grain.table, t) for t in sorted(tables | {weight_table}))
        return AllocatedReach(
            at_grain=alloc.at,
            at_table=grain.table,
            at_key=tuple(grain.key),
            prefix_edges=prefix,
            suffix_edges=suffix,
            weight=alloc.weight,
            missing_label=alloc.missing,
        )


def merge_paths(paths) -> tuple[Edge, ...]:
    """Union of paths that all start at the same table, in first-seen order.

    Raises if two paths reach the same table through different edges, which
    would make the join ambiguous.
    """
    out: list[Edge] = []
    reached: dict[str, Edge] = {}
    for path in paths:
        for edge in path:
            existing = reached.get(edge.dst)
            if existing is None:
                reached[edge.dst] = edge
                out.append(edge)
            elif existing != edge:
                raise SchemaViolation(
                    f"table {edge.dst!r} is reached through two different joins "
                    f"({existing.describe()} and {edge.describe()})"
                )
    return tuple(out)
