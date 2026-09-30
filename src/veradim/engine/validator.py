"""Last line of defense: validate every SQL statement before it runs.

Even though Veradim generates all SQL itself from approved definitions, every
statement is parsed again with sqlglot and must satisfy:

* exactly one statement, and it is a SELECT (or a set operation of SELECTs)
* no statement nodes anywhere inside it (DDL, DML, ATTACH, COPY, PRAGMA...)
* no table functions (read_csv, read_parquet, glob...) and no unknown functions
* every table is a whitelisted base table or a CTE defined in the statement
* every column qualified with a base table exists in that table
* a top-level LIMIT no larger than the configured maximum
"""

from __future__ import annotations

from collections.abc import Mapping

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from veradim.errors import UnsafeSQL
from veradim.semantic.sqlexpr import DIALECT, STATEMENT_TYPES


def validate_select(sql: str, allowed: Mapping[str, set[str]], *, max_limit: int) -> exp.Expression:
    try:
        statements = sqlglot.parse(sql, read=DIALECT)
    except ParseError as exc:
        raise UnsafeSQL(f"SQL does not parse: {exc}") from exc
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise UnsafeSQL(f"expected exactly one statement, found {len(statements)}")
    root = statements[0]
    if not isinstance(root, (exp.Select, exp.SetOperation)):
        raise UnsafeSQL(f"only SELECT statements are allowed, got {root.key.upper()}")

    for node in root.walk():
        if isinstance(node, STATEMENT_TYPES):
            raise UnsafeSQL(f"{node.key.upper()} is not allowed inside a query")
        if isinstance(node, exp.Anonymous):
            raise UnsafeSQL(f"function {node.name!r} is not allowed")
        if type(node).__name__.startswith("Read"):
            raise UnsafeSQL(f"table function {node.key!r} is not allowed")

    cte_names = {cte.alias_or_name for cte in root.find_all(exp.CTE)}
    table_aliases: dict[str, str] = {}
    for table in root.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise UnsafeSQL(f"table expression {table.sql(dialect=DIALECT)!r} is not allowed")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQL(f"qualified table names are not allowed: {table.sql(dialect=DIALECT)!r}")
        name = table.name
        if name in cte_names:
            continue
        if name not in allowed:
            raise UnsafeSQL(f"table {name!r} is not whitelisted", details={"allowed": sorted(allowed)})
        table_aliases[table.alias_or_name] = name

    for col in root.find_all(exp.Column):
        qualifier = col.table
        if not qualifier or qualifier in cte_names:
            continue
        base = table_aliases.get(qualifier)
        if base is None:
            # A qualifier that is neither a base table nor a CTE (e.g. a subquery alias).
            if qualifier in allowed:
                raise UnsafeSQL(f"column {col.sql()!r} references table {qualifier!r} that is not in FROM")
            continue
        if col.name not in allowed[base]:
            raise UnsafeSQL(f"column {base}.{col.name} is not whitelisted")

    limit = root.args.get("limit")
    if limit is None:
        raise UnsafeSQL("query must have a LIMIT")
    limit_value = limit.expression
    if not (isinstance(limit_value, exp.Literal) and limit_value.is_int):
        raise UnsafeSQL("LIMIT must be an integer literal")
    if int(limit_value.this) > max_limit:
        raise UnsafeSQL(f"LIMIT {limit_value.this} exceeds the maximum of {max_limit}")
    return root
