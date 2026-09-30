"""Veradim MCP server (stdio).

The agent can discover, describe and query *approved* metrics. It never writes
SQL and never computes numbers: every number comes back with a receipt_id.

Deliberately absent: any tool that approves, rejects or edits approved
metrics. Approval is a human-only action through the `veradim` CLI.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from veradim.config import DEFAULT_LIMIT, MAX_LIMIT
from veradim.engine.sql_builder import DimFilter, TimeGrain, TimeRange
from veradim.errors import VeradimError
from veradim.tools.core import VeradimApp
from veradim.tools.schemas import DescribeMetricOutput, ListMetricsOutput, QueryMetricOutput

INSTRUCTIONS = """\
Veradim is a governed BI analyst. Rules:
1. Never compute or estimate numbers yourself. Every number you report must come
   from a Veradim tool and be quoted with its receipt_id.
2. Only approved metrics can be queried. Call list_metrics first, then
   describe_metric to see allowed dimensions and filters.
3. Time ranges are half-open: start is inclusive, end is exclusive.
4. Errors are JSON with a code, a message and a hint; follow the hint and retry.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)

mcp = MCPServer(name="veradim", instructions=INSTRUCTIONS)
app = VeradimApp()


def structured_errors[T](fn: Callable[..., T]) -> Callable[..., T]:
    """Turn VeradimError into a ToolError whose text is the JSON error payload."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except VeradimError as exc:
            raise ToolError(exc.to_json()) from exc

    return wrapper


@mcp.tool(annotations=READ_ONLY)
@structured_errors
def list_metrics() -> ListMetricsOutput:
    """List approved metrics with their descriptions, allowed dimensions and version hashes."""
    return app.list_metrics()


@mcp.tool(annotations=READ_ONLY)
@structured_errors
def describe_metric(name: Annotated[str, Field(description="Approved metric name")]) -> DescribeMetricOutput:
    """Full definition of an approved metric: components, filters, dimensions, an example request and its SQL."""
    return app.describe_metric(name)


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=False))
@structured_errors
def query_metric(
    metric: Annotated[str, Field(description="Approved metric name (see list_metrics)")],
    time_range: Annotated[
        TimeRange | None,
        Field(description="Half-open range {start, end} in YYYY-MM-DD; end is exclusive. Omit for all time."),
    ] = None,
    filters: Annotated[
        list[DimFilter] | None,
        Field(description="Filters on allowed dimensions, e.g. [{dimension: customer_state, op: in, values: [SP]}]"),
    ] = None,
    group_by: Annotated[list[str] | None, Field(description="Allowed dimensions to group by")] = None,
    time_grain: Annotated[TimeGrain | None, Field(description="Bucket results by day/week/month/quarter/year")] = None,
    limit: Annotated[int, Field(ge=1, le=MAX_LIMIT)] = DEFAULT_LIMIT,
) -> QueryMetricOutput:
    """Compute an approved metric. Veradim builds and runs the SQL read-only and returns rows plus a receipt.

    Quote the receipt_id next to every number you report.
    """
    return app.query_metric(metric, time_range, filters, group_by, time_grain, limit)


def main() -> None:
    mcp.run("stdio")


if __name__ == "__main__":
    main()
