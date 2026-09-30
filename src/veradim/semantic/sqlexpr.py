"""Helpers to inspect the small SQL snippets that appear in the YAML files.

Metric expressions, filters and dimension expressions are written by humans and
approved by humans, but we still parse them with sqlglot so the registry can
check which tables they touch, and reject anything that is not a plain
expression over qualified columns.
"""

from __future__ import annotations

from functools import lru_cache

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from veradim.errors import SchemaViolation

DIALECT = "duckdb"

AGGREGATES: tuple[type[exp.Expression], ...] = (
    exp.Sum,
    exp.Count,
    exp.Avg,
    exp.Min,
    exp.Max,
    exp.Median,
    exp.Quantile,
    exp.Stddev,
    exp.Variance,
)
# Statement node types: never allowed inside a snippet, and never allowed
# anywhere in generated SQL other than the single root SELECT.
STATEMENT_TYPES: tuple[type[exp.Expression], ...] = tuple(
    getattr(exp, name)
    for name in (
        "Insert",
        "Update",
        "Delete",
        "Drop",
        "Create",
        "Alter",
        "Command",
        "Pragma",
        "Set",
        "Use",
        "Copy",
        "Merge",
        "TruncateTable",
        "Attach",
        "Detach",
        "Transaction",
        "Commit",
        "Rollback",
        "LoadData",
        "Export",
        "Install",
        "Grant",
        "Describe",
        "Summarize",
        "Show",
        "Analyze",
        "Kill",
        "Cache",
        "Uncache",
        "Refresh",
        "Declare",
        "Execute",
    )
    if hasattr(exp, name)
)

_FORBIDDEN_IN_SNIPPETS: tuple[type[exp.Expression], ...] = (
    exp.Query,
    exp.Subquery,
    exp.Window,
    exp.Placeholder,
    exp.Table,
    exp.Anonymous,
    *STATEMENT_TYPES,
)


@lru_cache(maxsize=1024)
def parse_snippet(sql: str, *, context: str) -> exp.Expression:
    try:
        statements = sqlglot.parse(sql, read=DIALECT)
    except ParseError as exc:
        raise SchemaViolation(f"{context}: cannot parse SQL {sql!r}: {exc}") from exc
    if len(statements) != 1 or statements[0] is None:
        raise SchemaViolation(f"{context}: expected exactly one expression in {sql!r}")
    node = statements[0]
    for forbidden in _FORBIDDEN_IN_SNIPPETS:
        if isinstance(node, forbidden) or node.find(forbidden) is not None:
            raise SchemaViolation(
                f"{context}: {forbidden.__name__} is not allowed in {sql!r}",
                hint="Use a plain expression over qualified columns (table.column).",
            )
    for col in node.find_all(exp.Column):
        if not col.table:
            raise SchemaViolation(
                f"{context}: column {col.name!r} must be qualified with its table",
                hint="Write table.column, e.g. orders.order_status.",
            )
    return node


def referenced_columns(sql: str, *, context: str) -> set[tuple[str, str]]:
    node = parse_snippet(sql, context=context)
    return {(c.table, c.name) for c in node.find_all(exp.Column)}


def referenced_tables(sql: str, *, context: str) -> set[str]:
    return {t for t, _ in referenced_columns(sql, context=context)}


def has_aggregate(sql: str, *, context: str) -> bool:
    return parse_snippet(sql, context=context).find(*AGGREGATES) is not None


def additive_aggregate(sql: str, *, context: str) -> str | None:
    """Return 'sum' or 'count' if the expression is one additive aggregate, else None.

    Only such expressions can be pre-aggregated to a coarser grain and then
    split by allocation weights without changing their meaning.
    """
    node = parse_snippet(sql, context=context)
    if isinstance(node, exp.Sum):
        return "sum"
    if isinstance(node, exp.Count) and not isinstance(node.this, exp.Distinct):
        return "count"
    return None


def normalize_condition(sql: str, *, context: str) -> str:
    """Canonical text of a filter, used to compare filters across metrics."""
    return parse_snippet(sql, context=context).sql(dialect=DIALECT, normalize=True, pretty=False)


def formula_metric_names(formula: str, *, context: str) -> set[str]:
    """Metric names used in a derived-metric formula. Only arithmetic is allowed."""
    try:
        node = sqlglot.parse_one(formula, dialect=DIALECT)
    except ParseError as exc:
        raise SchemaViolation(f"{context}: cannot parse formula {formula!r}: {exc}") from exc
    allowed = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Paren, exp.Neg, exp.Column, exp.Identifier, exp.Literal)
    for sub in node.walk():
        if not isinstance(sub, allowed):
            raise SchemaViolation(
                f"{context}: formula may only use metric names, numbers and + - * / ( ); "
                f"found {sub.key!r} in {formula!r}"
            )
        if isinstance(sub, exp.Literal) and sub.is_string:
            raise SchemaViolation(f"{context}: string literals are not allowed in a formula")
    names = set()
    for col in node.find_all(exp.Column):
        if col.table:
            raise SchemaViolation(f"{context}: formula must reference metric names, not table columns ({col.sql()})")
        names.add(col.name)
    return names
