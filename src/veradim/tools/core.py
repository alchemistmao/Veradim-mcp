"""Tool implementations, independent of the MCP transport.

``VeradimApp`` owns the read-only database, the receipt store and the metric
registry. The registry is reloaded automatically when files in metrics/ change,
so a metric approved by a human while the server is running becomes queryable
without a restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from veradim.config import DEFAULT_LIMIT, MAX_LIMIT, Settings
from veradim.engine.connection import ReadOnlyDatabase
from veradim.engine.sql_builder import CompiledQuery, DimFilter, QuerySpec, SqlBuilder, TimeGrain, TimeRange
from veradim.engine.validator import validate_select
from veradim.errors import DatabaseUnavailable
from veradim.receipts.store import Receipt, ReceiptStore, build_receipt, to_jsonable
from veradim.semantic.graph import AllocatedReach, JoinGraph
from veradim.semantic.models import SimpleMetric
from veradim.semantic.registry import LoadedMetric, Registry
from veradim.tools.schemas import (
    ComponentInfo,
    DescribeMetricOutput,
    DimensionInfo,
    ListMetricsOutput,
    MetricSummary,
    QueryMetricOutput,
)


@dataclass
class ExecutedQuery:
    compiled: CompiledQuery
    rows: list[dict[str, Any]]
    receipt: Receipt


@dataclass
class VeradimApp:
    settings: Settings = field(default_factory=Settings)

    def __post_init__(self) -> None:
        self.db = ReadOnlyDatabase(self.settings.db_path)
        self.receipts = ReceiptStore(self.settings.receipts_db_path)
        self._registry: Registry | None = None
        self._registry_key: tuple | None = None

    # -- registry ------------------------------------------------------------

    def _metrics_fingerprint(self) -> tuple:
        root = self.settings.metrics_dir
        files = [root / "dimensions.yaml", *sorted((root / "approved").glob("*.yaml"))]
        files += sorted((root / "proposed").glob("*.yaml"))
        return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in files if p.exists())

    def _schema(self) -> dict[str, set[str]] | None:
        try:
            return self.db.column_sets()
        except DatabaseUnavailable:
            return None

    @property
    def registry(self) -> Registry:
        key = self._metrics_fingerprint()
        if self._registry is None or key != self._registry_key:
            self._registry = Registry.load(self.settings.metrics_dir, schema=self._schema())
            self._registry_key = key
        return self._registry

    def whitelist(self) -> dict[str, set[str]]:
        tables = JoinGraph(self.registry.catalog).tables
        return {t: cols for t, cols in self.db.column_sets().items() if t in tables}

    # -- execution -------------------------------------------------------------

    def run(self, metric: str, spec: QuerySpec, *, tool: str) -> ExecutedQuery:
        registry = self.registry
        loaded = registry.get(metric)
        compiled = SqlBuilder(registry).compile(metric, spec)
        validate_select(compiled.sql, self.whitelist(), max_limit=MAX_LIMIT)
        columns, raw_rows = self.db.execute(compiled.sql, compiled.params)
        rows = [{c: to_jsonable(v) for c, v in zip(columns, r, strict=True)} for r in raw_rows]
        receipt = build_receipt(
            metric=metric,
            metric_version=loaded.version_hash,
            component_versions={c.name: c.version_hash for c in registry.components(metric)},
            sql=compiled.sql,
            parameters=compiled.params,
            rows=rows,
            data_snapshot_hash=self.db.snapshot_hash(),
        )
        self.receipts.save(receipt, tool=tool, rows=rows)
        return ExecutedQuery(compiled=compiled, rows=rows, receipt=receipt)

    # -- tools -----------------------------------------------------------------

    def list_metrics(self) -> ListMetricsOutput:
        registry = self.registry
        return ListMetricsOutput(
            metrics=[_summary(m) for m in registry.metrics.values()],
            invalid=dict(registry.errors),
        )

    def describe_metric(self, name: str) -> DescribeMetricOutput:
        registry = self.registry
        loaded = registry.get(name)
        metric = loaded.definition
        graph = JoinGraph(registry.catalog)

        if isinstance(metric, SimpleMetric):
            base = metric
            components: list[ComponentInfo] = []
        else:
            comps = registry.components(name)
            base = comps[0].definition
            components = [
                ComponentInfo(
                    name=c.name,
                    version_hash=c.version_hash,
                    expression=c.definition.expression,
                    measure_grain=c.definition.measure_grain,
                )
                for c in comps
            ]

        dims = []
        grain_table = registry.catalog.grains[base.measure_grain].table
        for d in metric.dimensions:
            dim = registry.catalog.dimensions[d]
            reach = graph.reach_dimension(grain_table, d)
            allocation = None
            if isinstance(reach, AllocatedReach):
                allocation = (
                    f"one {reach.at_grain} can have several {d} values; the measure is split by share of "
                    f"{reach.weight}, so members sum to the total"
                )
            dims.append(DimensionInfo(name=d, description=" ".join(dim.description.split()), allocation=allocation))

        example_group = metric.dimensions[:1]
        example_spec = QuerySpec(
            time_range=TimeRange.model_validate({"start": "2018-01-01", "end": "2018-04-01"}),
            group_by=example_group,
            time_grain="month",
            limit=100,
        )
        example_sql = SqlBuilder(registry).compile(name, example_spec).sql
        return DescribeMetricOutput(
            summary=_summary(loaded),
            definition=metric.model_dump(mode="json", exclude_none=True),
            time_column=base.time_column,
            filters=list(base.filters),
            measure_grain=metric.measure_grain if isinstance(metric, SimpleMetric) else None,
            components=components,
            dimensions=dims,
            simulatable=[s.model_dump() for s in (metric.simulatable if isinstance(metric, SimpleMetric) else [])],
            example_request={
                "metric": name,
                "time_range": {"start": "2018-01-01", "end": "2018-04-01"},
                "group_by": example_group,
                "time_grain": "month",
                "limit": 100,
            },
            example_sql=example_sql,
        )

    def query_metric(
        self,
        metric: str,
        time_range: TimeRange | None = None,
        filters: list[DimFilter] | None = None,
        group_by: list[str] | None = None,
        time_grain: TimeGrain | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> QueryMetricOutput:
        spec = QuerySpec(
            time_range=time_range or TimeRange(),
            filters=filters or [],
            group_by=group_by or [],
            time_grain=time_grain,
            limit=limit,
        )
        result = self.run(metric, spec, tool="query_metric")
        loaded = self.registry.get(metric)
        rid = result.receipt.receipt_id
        columns = result.compiled.key_columns + result.compiled.value_columns
        return QueryMetricOutput(
            metric=metric,
            metric_version=loaded.version_hash,
            format=loaded.definition.format,
            columns=columns,
            rows=[{**row, "receipt_id": rid} for row in result.rows],
            row_count=len(result.rows),
            truncated=len(result.rows) >= limit,
            receipt_id=rid,
            receipt=result.receipt,
            notes=result.compiled.notes,
        )


def _summary(loaded: LoadedMetric) -> MetricSummary:
    m = loaded.definition
    return MetricSummary(
        name=m.name,
        kind=m.kind,
        description=" ".join(m.description.split()),
        owner=m.owner,
        version=m.version,
        version_hash=loaded.version_hash,
        dimensions=list(m.dimensions),
        format=m.format,
        depends_on=list(getattr(m, "depends_on", [])),
        approved_by=m.approval.approved_by if m.approval else None,
        approved_at=m.approval.approved_at.isoformat() if m.approval else None,
    )


def default_app(metrics_dir: Path | None = None) -> VeradimApp:
    settings = Settings() if metrics_dir is None else Settings(metrics_dir=metrics_dir)
    return VeradimApp(settings)
