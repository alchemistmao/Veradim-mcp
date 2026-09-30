"""Cross-file rules for metric definitions.

Shape is checked by the Pydantic models; this module checks meaning:

* simple metrics: grain exists, every referenced table is reachable without
  fan-out, every dimension is reachable (directly, or across a one_to_many join
  with an allocation rule and an additive measure), simulatable entries are
  consistent, and (when a schema is given) every column exists.
* derived / ratio metrics: components exist, are simple, share the same
  time column and the same filters, and expose every declared dimension.
"""

from __future__ import annotations

from collections.abc import Mapping

from veradim.errors import FanoutJoinRejected, IncompatibleComponents, SchemaViolation
from veradim.semantic.graph import AllocatedReach, JoinGraph
from veradim.semantic.models import (
    DerivedMetric,
    DimensionCatalog,
    RatioMetric,
    SimpleMetric,
    SimulatableColumn,
)
from veradim.semantic.sqlexpr import (
    additive_aggregate,
    formula_metric_names,
    has_aggregate,
    normalize_condition,
    referenced_columns,
)

# table -> set of column names, from the live database
Schema = Mapping[str, set[str]]


def validate_catalog(catalog: DimensionCatalog, schema: Schema | None = None) -> None:
    graph = JoinGraph(catalog)
    for join in catalog.joins:
        _check_columns(schema, {(join.from_table, a) for a, _ in join.on}, context="join")
        _check_columns(schema, {(join.to_table, b) for _, b in join.on}, context="join")
    for name, grain in catalog.grains.items():
        _check_columns(schema, {(grain.table, k) for k in grain.key}, context=f"grain {name}")
    for name, dim in catalog.dimensions.items():
        context = f"dimension {name}"
        if has_aggregate(dim.expression, context=context):
            raise SchemaViolation(f"{context}: expression must not aggregate")
        _check_columns(schema, referenced_columns(dim.expression, context=context), context=context)
        if dim.fanout_allocation:
            alloc = dim.fanout_allocation
            if alloc.at not in catalog.grains:
                raise SchemaViolation(f"{context}: fanout_allocation.at {alloc.at!r} is not a declared grain")
            wt, wc = alloc.weight.split(".")
            _check_columns(schema, {(wt, wc)}, context=context)
        # every table the dimension touches must be connected to the graph
        for table in graph.dimension_tables(name, dim):
            if table not in graph.tables:
                raise SchemaViolation(f"{context}: table {table!r} is not declared in any join")


def validate_simple(metric: SimpleMetric, catalog: DimensionCatalog, schema: Schema | None = None) -> None:
    ctx = f"metric {metric.name}"
    graph = JoinGraph(catalog)
    grain = catalog.grains.get(metric.measure_grain)
    if grain is None:
        raise SchemaViolation(
            f"{ctx}: measure_grain {metric.measure_grain!r} is not declared",
            hint=f"Declared grains: {', '.join(sorted(catalog.grains))}",
        )

    if not has_aggregate(metric.expression, context=f"{ctx} expression"):
        raise SchemaViolation(f"{ctx}: expression must be an aggregation, e.g. SUM(order_items.price)")

    time_table, time_col = metric.time_column.split(".")
    referenced = {(time_table, time_col)}
    referenced |= referenced_columns(metric.expression, context=f"{ctx} expression")
    for f in metric.filters:
        if has_aggregate(f, context=f"{ctx} filter"):
            raise SchemaViolation(f"{ctx}: filter {f!r} must not aggregate")
        referenced |= referenced_columns(f, context=f"{ctx} filter")
    _check_columns(schema, referenced, context=ctx)

    # Expression, filters and time column must not fan out the measure.
    for table in sorted({t for t, _ in referenced}):
        graph.safe_path(grain.table, table, context=ctx)

    additive = additive_aggregate(metric.expression, context=f"{ctx} expression")
    for dim_name in metric.dimensions:
        if dim_name not in catalog.dimensions:
            raise SchemaViolation(
                f"{ctx}: unknown dimension {dim_name!r}",
                hint=f"Declared dimensions: {', '.join(sorted(catalog.dimensions))}",
            )
        reach = graph.reach_dimension(grain.table, dim_name)
        if isinstance(reach, AllocatedReach) and additive is None:
            raise FanoutJoinRejected(
                f"{ctx}: dimension {dim_name!r} needs allocation across a one_to_many join, "
                "which is only valid for a single SUM(...) or COUNT(...) measure",
                hint=f"Remove {dim_name!r} from this metric's dimensions.",
            )

    expr_cols = referenced_columns(metric.expression, context=f"{ctx} expression")
    for entry in metric.simulatable:
        if isinstance(entry, SimulatableColumn):
            table, col = entry.column.split(".")
            if (table, col) not in expr_cols:
                raise SchemaViolation(
                    f"{ctx}: simulatable column {entry.column!r} is not used by the expression",
                )
        elif entry.dimension not in metric.dimensions:
            raise SchemaViolation(
                f"{ctx}: simulatable dimension {entry.dimension!r} is not one of the metric's dimensions",
            )


def validate_composite(
    metric: DerivedMetric | RatioMetric,
    components: Mapping[str, SimpleMetric],
) -> None:
    ctx = f"metric {metric.name}"
    if isinstance(metric, DerivedMetric):
        used = formula_metric_names(metric.formula, context=ctx)
        if used != set(metric.depends_on):
            raise SchemaViolation(
                f"{ctx}: depends_on {sorted(metric.depends_on)} must match the metrics used "
                f"in the formula {sorted(used)}"
            )
    missing = [c for c in metric.depends_on if c not in components]
    if missing:
        raise IncompatibleComponents(
            f"{ctx}: component metric(s) {', '.join(missing)} are not approved simple metrics",
            hint="Approve the components first. Components must be simple metrics.",
        )

    parts = [components[c] for c in metric.depends_on]
    time_columns = {p.time_column for p in parts}
    if len(time_columns) > 1:
        raise IncompatibleComponents(
            f"{ctx}: components use different time columns: " + ", ".join(f"{p.name}={p.time_column}" for p in parts),
            hint="Components of a derived or ratio metric must share the same time column.",
        )
    filter_sets = {
        p.name: frozenset(normalize_condition(f, context=f"metric {p.name} filter") for f in p.filters) for p in parts
    }
    if len(set(filter_sets.values())) > 1:
        raise IncompatibleComponents(
            f"{ctx}: components have incompatible filters: "
            + "; ".join(f"{n}={sorted(fs)}" for n, fs in filter_sets.items()),
            hint="Components must apply exactly the same filters, otherwise the result mixes populations.",
        )
    for dim in metric.dimensions:
        lacking = [p.name for p in parts if dim not in p.dimensions]
        if lacking:
            raise IncompatibleComponents(
                f"{ctx}: dimension {dim!r} is not available on component(s) {', '.join(lacking)}",
                hint="A composite metric can only use dimensions shared by all of its components.",
            )


def _check_columns(schema: Schema | None, columns: set[tuple[str, str]], *, context: str) -> None:
    if schema is None:
        return
    for table, col in sorted(columns):
        if table not in schema:
            raise SchemaViolation(f"{context}: table {table!r} does not exist in the database")
        if col not in schema[table]:
            raise SchemaViolation(f"{context}: column {table}.{col} does not exist in the database")
