"""SQL validation and read-only guarantees."""

from __future__ import annotations

import duckdb
import pytest

from veradim.engine.validator import validate_select
from veradim.errors import QueryFailed, UnsafeSQL

ALLOWED = {"orders": {"order_id", "order_status"}, "order_items": {"order_id", "price"}}


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "SELECT 1 LIMIT 1; DROP TABLE orders",
        "INSERT INTO orders VALUES ('x', 'y')",
        "ATTACH 'other.db' AS other",
        "COPY orders TO '/tmp/x.csv'",
        "PRAGMA database_list",
        "INSTALL httpfs",
        "SELECT * FROM read_csv('/etc/passwd') LIMIT 1",
        "SELECT * FROM glob('*') LIMIT 1",
        "SELECT getenv('HOME') LIMIT 1",
        "SELECT * FROM customers LIMIT 1",  # not whitelisted
        "SELECT orders.secret FROM orders LIMIT 1",  # unknown column
        "SELECT * FROM main.orders LIMIT 1",  # qualified table
        "SELECT order_id FROM orders",  # no LIMIT
        "SELECT order_id FROM orders LIMIT 1000000",  # over max
        "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x LIMIT 1",
    ],
)
def test_validator_rejects(sql):
    with pytest.raises(UnsafeSQL):
        validate_select(sql, ALLOWED, max_limit=10_000)


def test_validator_accepts_cte_select():
    sql = (
        "WITH t AS (SELECT orders.order_id AS k FROM orders) "
        "SELECT t.k, SUM(order_items.price) AS v FROM t JOIN order_items ON t.k = order_items.order_id "
        "GROUP BY ALL LIMIT 10"
    )
    validate_select(sql, ALLOWED, max_limit=10_000)


def test_generated_sql_passes_validation(app):
    for metric, group_by in [("revenue", ["payment_type"]), ("avg_ticket", ["customer_state"]), ("margin", [])]:
        out = app.query_metric(metric, group_by=group_by)
        validate_select(out.receipt.sql, app.whitelist(), max_limit=10_000)


def test_connection_is_read_only(app):
    with pytest.raises(QueryFailed):
        app.db.execute("CREATE TABLE hack AS SELECT 1")
    with pytest.raises(QueryFailed):
        app.db.execute("DELETE FROM orders")
    with pytest.raises(QueryFailed):
        app.db.execute("SELECT * FROM read_csv('/etc/hosts')")
    with pytest.raises(QueryFailed):
        app.db.execute("SET enable_external_access = true")


def test_analytical_db_is_untouched_by_queries(app, workspace):
    app.query_metric("revenue", group_by=["payment_type"])
    app.db.close()
    con = duckdb.connect(str(workspace.db_path), read_only=True)
    tables = {r[0] for r in con.execute("SHOW TABLES").fetchall()}
    con.close()
    assert "veradim_receipts" not in tables
