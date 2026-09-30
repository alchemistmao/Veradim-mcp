"""Compile an approved metric plus a request into one read-only SELECT.

The LLM never writes SQL. It picks a metric, dimensions, filters and a time
range; this module turns that into SQL using only the approved YAML and the
join graph, with every user-supplied value bound as a parameter.

Fan-out safety
--------------
Dimensions reachable through many_to_one / one_to_one joins are joined
directly. A dimension that sits across a one_to_many join (payment_type: one
order, several payments) is handled by *allocation*: the measure is first
aggregated to the allocation grain (one row per order), then split across the
members in proportion to their weights. Shares are exact decimals, so the
split always sums back to the unsplit total.

Composite metrics
-----------------
Each component is compiled on its own, the results are stacked with
UNION ALL and pivoted by the output keys, and the formula (or ratio) is
applied on top. Components never share a join, so one cannot inflate another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

import sqlglot
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlglot import exp

from veradim.config import DEFAULT_LIMIT, MAX_LIMIT
from veradim.errors import InvalidDimension, InvalidFilter, InvalidTimeRange
from veradim.semantic.graph import AllocatedReach, DirectReach, JoinGraph, merge_paths
from veradim.semantic.models import DerivedMetric, RatioMetric, SimpleMetric
from veradim.semantic.registry import Registry
from veradim.semantic.sqlexpr import DIALECT, referenced_tables

TimeGrain = Literal["day", "week", "month", "quarter", "year"]
RESERVED_COLUMNS = {"period", "value"}
SHARE_TYPE = "DECIMAL(18, 15)"


class DimFilter(BaseModel):
    """Filter on an allowed dimension of the metric."""

    model_config = ConfigDict(extra="forbid")

    dimension: str
    op: Literal["eq", "neq", "in", "not_in"] = "in"
    values: list[str | int | float] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def _single_value_ops(self) -> DimFilter:
        if self.op in ("eq", "neq") and len(self.values) != 1:
            raise ValueError(f"op {self.op!r} takes exactly one value")
        return self


class TimeRange(BaseModel):
    """Half-open date range: start inclusive, end exclusive. Either side may be omitted."""

    model_config = ConfigDict(extra="forbid")

    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def _ordered(self) -> TimeRange:
        if self.start and self.end and self.start >= self.end:
            raise ValueError("time_range.start must be before time_range.end (end is exclusive)")
        return self


@dataclass
class QuerySpec:
    time_range: TimeRange = field(default_factory=TimeRange)
    filters: list[DimFilter] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    time_grain: TimeGrain | None = None
    limit: int = DEFAULT_LIMIT


@dataclass
class CompiledQuery:
    sql: str
    params: dict[str, Any]
    key_columns: list[str]
    value_columns: list[str]
    notes: list[str]


class SqlBuilder:
    def __init__(self, registry: Registry):
        self.registry = registry
        self.catalog = registry.catalog
        self.graph = JoinGraph(registry.catalog)

    # -- public ------------------------------------------------------------

    def compile(self, metric_name: str, spec: QuerySpec) -> CompiledQuery:
        loaded = self.registry.get(metric_name)
        metric = loaded.definition
        self._check_spec(metric, spec)
        params: dict[str, Any] = {}
        notes: list[str] = []

        if isinstance(metric, SimpleMetric):
            select = self._simple(metric, spec, params, notes)
            value_columns = ["value"]
        else:
            select, value_columns = self._composite(metric, spec, params, notes)

        keys = self._key_columns(spec)
        order = (
            (["period"] if spec.time_grain else []) + ['"value" DESC NULLS LAST'] + [f'"{d}"' for d in spec.group_by]
        )
        select = select.order_by(*order).limit(spec.limit)
        return CompiledQuery(
            sql=select.sql(dialect=DIALECT, pretty=True),
            params=params,
            key_columns=keys,
            value_columns=value_columns,
            notes=sorted(set(notes)),
        )

    # -- request validation -----------------------------------------------

    def _check_spec(self, metric, spec: QuerySpec) -> None:
        allowed = set(metric.dimensions)
        seen: set[str] = set()
        for dim in spec.group_by:
            if dim not in allowed:
                raise InvalidDimension(
                    f"metric {metric.name!r} cannot be grouped by {dim!r}",
                    hint=f"Allowed dimensions: {', '.join(metric.dimensions) or 'none'}",
                )
            if dim in seen:
                raise InvalidDimension(f"dimension {dim!r} is listed twice in group_by")
            if dim in RESERVED_COLUMNS:
                raise InvalidDimension(f"dimension name {dim!r} is reserved")
            seen.add(dim)
        for f in spec.filters:
            if f.dimension not in allowed:
                raise InvalidFilter(
                    f"metric {metric.name!r} cannot be filtered by {f.dimension!r}",
                    hint=f"Allowed dimensions: {', '.join(metric.dimensions) or 'none'}",
                )
        if not 1 <= spec.limit <= MAX_LIMIT:
            raise InvalidFilter(f"limit must be between 1 and {MAX_LIMIT}")
        tr = spec.time_range
        if tr.start and tr.end and tr.start >= tr.end:
            raise InvalidTimeRange("time_range.start must be before time_range.end (end is exclusive)")

    def _key_columns(self, spec: QuerySpec) -> list[str]:
        return (["period"] if spec.time_grain else []) + list(spec.group_by)

    # -- simple metrics ----------------------------------------------------

    def _simple(self, metric: SimpleMetric, spec: QuerySpec, params: dict[str, Any], notes: list[str]) -> exp.Select:
        grain = self.catalog.grains[metric.measure_grain]
        gt = grain.table
        tc = metric.time_column

        needed = list(dict.fromkeys(spec.group_by + [f.dimension for f in spec.filters]))
        reaches = {d: self.graph.reach_dimension(gt, d) for d in needed}
        allocated = {d: r for d, r in reaches.items() if isinstance(r, AllocatedReach)}
        if len(allocated) > 1:
            raise InvalidDimension(
                f"only one allocated dimension per query; got {', '.join(allocated)}",
                hint="Query them separately.",
            )

        # Tables used by the metric itself (never fan out: checked at approval time).
        own_tables = {tc.split(".")[0]} | referenced_tables(metric.expression, context=metric.name)
        for f in metric.filters:
            own_tables |= referenced_tables(f, context=metric.name)
        paths = [self.graph.safe_path(gt, t, context=metric.name) for t in sorted(own_tables)]
        paths += [r.edges for r in reaches.values() if isinstance(r, DirectReach)]
        paths += [r.prefix_edges for r in allocated.values()]
        edges = merge_paths(paths)

        conditions = [f"({f})" for f in metric.filters]
        conditions += self._time_conditions(tc, spec.time_range, params)
        direct_filters = [f for f in spec.filters if f.dimension not in allocated]
        conditions += [self._filter_condition(f, self._dim_expr(f.dimension), spec, params) for f in direct_filters]

        key_selects = []
        if spec.time_grain:
            key_selects.append(f"CAST(DATE_TRUNC('{spec.time_grain}', {tc}) AS DATE) AS period")
        direct_groups = [d for d in spec.group_by if d not in allocated]
        key_selects += [f'{self._dim_expr(d)} AS "{d}"' for d in direct_groups]

        if not allocated:
            select = exp.select(*key_selects, f"{metric.expression} AS value").from_(gt)
            select = self._apply_joins(select, edges)
            if conditions:
                select = select.where(*conditions)
            return select.group_by("ALL")

        # --- allocation across a one_to_many join ---
        ((dim_name, reach),) = allocated.items()
        notes.append(
            f"{dim_name}: measure pre-aggregated per {reach.at_grain} and split by share of "
            f"{reach.weight}; members sum to the unsplit total. Rows with no {dim_name} are labelled "
            f"'{reach.missing_label}'."
        )
        key_aliases = [f"__k{i}" for i in range(len(reach.at_key))]
        base = exp.select(
            *[f"{reach.at_table}.{k} AS {a}" for k, a in zip(reach.at_key, key_aliases, strict=True)],
            *key_selects,
            f"{metric.expression} AS __m",
        ).from_(gt)
        base = self._apply_joins(base, edges)
        if conditions:
            base = base.where(*conditions)
        base = base.group_by("ALL")

        weights_raw = exp.select(
            *[f"{reach.at_table}.{k} AS {a}" for k, a in zip(reach.at_key, key_aliases, strict=True)],
            f"{self._dim_expr(dim_name)} AS __member",
            f"SUM({reach.weight}) AS __w",
        ).from_(reach.at_table)
        weights_raw = self._apply_joins(weights_raw, reach.suffix_edges, join_type="inner").group_by("ALL")

        part = ", ".join(key_aliases)
        weights = exp.select(
            *key_aliases,
            "__member",
            f"CAST(CASE WHEN SUM(__w) OVER (PARTITION BY {part}) = 0 "
            f"THEN 1.0 / COUNT(*) OVER (PARTITION BY {part}) "
            f"ELSE __w / SUM(__w) OVER (PARTITION BY {part}) END AS {SHARE_TYPE}) AS __share",
        ).from_("__weights_raw")

        member = f"COALESCE(__weights.__member, '{_escape(reach.missing_label)}')"
        final_keys = (["__base.period AS period"] if spec.time_grain else []) + [
            (f'{member} AS "{d}"' if d == dim_name else f'__base."{d}" AS "{d}"') for d in spec.group_by
        ]
        final = (
            exp.select(*final_keys, f"SUM(__base.__m * COALESCE(__weights.__share, CAST(1 AS {SHARE_TYPE}))) AS value")
            .from_("__base")
            .join(
                "__weights",
                on=" AND ".join(f"__base.{a} = __weights.{a}" for a in key_aliases),
                join_type="left",
            )
            .with_("__base", as_=base)
            .with_("__weights_raw", as_=weights_raw)
            .with_("__weights", as_=weights)
        )
        alloc_filters = [
            self._filter_condition(f, member, spec, params) for f in spec.filters if f.dimension == dim_name
        ]
        if alloc_filters:
            final = final.where(*alloc_filters)
        return final.group_by("ALL")

    # -- composite metrics -------------------------------------------------

    def _composite(
        self,
        metric: DerivedMetric | RatioMetric,
        spec: QuerySpec,
        params: dict[str, Any],
        notes: list[str],
    ) -> tuple[exp.Select, list[str]]:
        comps = list(metric.depends_on)
        keys = self._key_columns(spec)
        key_cols = [f'"{k}"' for k in keys]
        cte_names = {c: f"__c_{c}" for c in comps}

        branches = []
        for i, comp in enumerate(comps):
            cols = key_cols + [
                (f"value AS {cte_names[c]}" if j == i else f"NULL AS {cte_names[c]}") for j, c in enumerate(comps)
            ]
            branches.append(f"SELECT {', '.join(cols)} FROM {cte_names[comp]}")
        stacked = sqlglot.parse_one(" UNION ALL ".join(branches), dialect=DIALECT)

        pivot = (
            exp.select(*key_cols, *[f"SUM({cte_names[c]}) AS {cte_names[c]}" for c in comps])
            .from_(exp.alias_(exp.Subquery(this=stacked), "__stacked"))
            .group_by("ALL")
        )

        if isinstance(metric, RatioMetric):
            value = (
                f"COALESCE({cte_names[metric.numerator]}, 0) / NULLIF(COALESCE({cte_names[metric.denominator]}, 0), 0)"
            )
        else:
            value = self._formula_sql(metric.formula, cte_names)

        final = exp.select(
            *key_cols,
            f"{value} AS value",
            *[f'COALESCE({cte_names[c]}, 0) AS "{c}"' for c in comps],
        ).from_("__pivot")
        for comp in comps:
            sub = self._simple(self.registry.simple(comp), spec, params, notes)
            final = final.with_(cte_names[comp], as_=_hoist_ctes(sub, final, prefix=f"{cte_names[comp]}_"))
        final = final.with_("__pivot", as_=pivot)
        return final, ["value", *comps]

    def _formula_sql(self, formula: str, cte_names: dict[str, str]) -> str:
        tree = sqlglot.parse_one(formula, dialect=DIALECT)

        def rewrite(node: exp.Expression) -> exp.Expression:
            if isinstance(node, exp.Column) and node.name in cte_names:
                return sqlglot.parse_one(f"COALESCE({cte_names[node.name]}, 0)", dialect=DIALECT)
            if isinstance(node, exp.Div):
                return exp.Div(
                    this=node.this, expression=exp.Nullif(this=node.expression, expression=exp.Literal.number(0))
                )
            return node

        return tree.transform(rewrite).sql(dialect=DIALECT)

    # -- helpers -----------------------------------------------------------

    def _dim_expr(self, dim: str) -> str:
        return f"({self.catalog.dimensions[dim].expression})"

    @staticmethod
    def _apply_joins(select: exp.Select, edges, *, join_type: str = "left") -> exp.Select:
        for edge in edges:
            select = select.join(edge.dst, on=edge.condition(), join_type=join_type)
        return select

    @staticmethod
    def _time_conditions(time_column: str, tr: TimeRange, params: dict[str, Any]) -> list[str]:
        out = []
        if tr.start:
            params["start"] = tr.start.isoformat()
            out.append(f"{time_column} >= CAST($start AS TIMESTAMP)")
        if tr.end:
            params["end"] = tr.end.isoformat()
            out.append(f"{time_column} < CAST($end AS TIMESTAMP)")
        return out

    @staticmethod
    def _filter_condition(f: DimFilter, expr: str, spec: QuerySpec, params: dict[str, Any]) -> str:
        idx = spec.filters.index(f)
        names = []
        for j, v in enumerate(f.values):
            name = f"f{idx}_{j}"
            params[name] = v
            names.append(f"${name}")
        if f.op == "eq":
            return f"{expr} = {names[0]}"
        if f.op == "neq":
            return f"{expr} IS DISTINCT FROM {names[0]}"
        joined = ", ".join(names)
        if f.op == "in":
            return f"{expr} IN ({joined})"
        return f"({expr} IS NULL OR {expr} NOT IN ({joined}))"


def _escape(value: str) -> str:
    return value.replace("'", "''")


def _hoist_ctes(sub: exp.Select, outer: exp.Select, *, prefix: str) -> exp.Select:
    """Move a component's own CTEs to the outer query under unique names.

    DuckDB accepts nested WITH, but hoisting keeps the statement flat and
    avoids name clashes when two components both use allocation.
    """
    with_ = sub.args.get("with_") or sub.args.get("with")
    if not with_:
        return sub
    renames = {cte.alias_or_name: f"{prefix}{cte.alias_or_name}" for cte in with_.expressions}
    sub = sub.copy()
    sub.set("with_" if "with_" in sub.args else "with", None)

    def rename(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Table) and node.name in renames:
            return exp.alias_(exp.to_table(renames[node.name]), node.alias_or_name, table=True)
        return node

    for cte in with_.expressions:
        body = cte.this.transform(rename)
        outer.with_(renames[cte.alias_or_name], as_=body, copy=False)
    return sub.transform(rename)
