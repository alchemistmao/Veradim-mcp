"""Pydantic output models for the MCP tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from veradim.receipts.store import Receipt
from veradim.semantic.models import Format


class MetricSummary(BaseModel):
    name: str
    kind: str
    description: str
    owner: str
    version: int
    version_hash: str = Field(description="git blob hash of the approved YAML")
    dimensions: list[str]
    format: Format
    depends_on: list[str] = Field(default_factory=list)
    approved_by: str | None = None
    approved_at: str | None = None


class ListMetricsOutput(BaseModel):
    metrics: list[MetricSummary]
    invalid: dict[str, str] = Field(
        default_factory=dict,
        description="Approved files that failed validation and are therefore not queryable",
    )
    note: str = (
        "Only approved metrics are queryable. New metrics are proposals until a human "
        "approves them with the `veradim approve` CLI; no tool can approve a metric."
    )


class DimensionInfo(BaseModel):
    name: str
    description: str
    allocation: str | None = Field(None, description="How the measure is split when this dimension fans out")


class ComponentInfo(BaseModel):
    name: str
    version_hash: str
    expression: str
    measure_grain: str


class DescribeMetricOutput(BaseModel):
    summary: MetricSummary
    definition: dict[str, Any] = Field(description="The approved YAML definition")
    time_column: str
    filters: list[str]
    measure_grain: str | None = None
    components: list[ComponentInfo] = Field(default_factory=list)
    dimensions: list[DimensionInfo]
    simulatable: list[dict[str, Any]] = Field(default_factory=list)
    example_request: dict[str, Any]
    example_sql: str


class QueryMetricOutput(BaseModel):
    metric: str
    metric_version: str
    format: Format
    columns: list[str]
    rows: list[dict[str, Any]] = Field(description="Each row carries the receipt_id that backs its numbers")
    row_count: int
    truncated: bool = Field(description="True if the row count hit the limit; more rows may exist")
    receipt_id: str
    receipt: Receipt
    notes: list[str] = Field(default_factory=list)
