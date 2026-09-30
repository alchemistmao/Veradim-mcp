"""Receipts are complete, reproducible and tamper-evident."""

from __future__ import annotations

import duckdb

from veradim.receipts.store import compute_receipt_id


def test_receipt_fields_and_storage(app):
    out = app.query_metric("revenue", group_by=["customer_state"])
    r = out.receipt
    assert r.metric == "revenue"
    assert r.metric_version == app.registry.get("revenue").version_hash
    assert r.row_count == len(out.rows)
    assert r.sql.startswith("SELECT") and "LIMIT" in r.sql
    assert len(r.data_snapshot_hash) == 64 and len(r.result_hash) == 64
    assert r.receipt_id == compute_receipt_id(r.model_dump(exclude={"receipt_id"}))

    stored = app.receipts.get(r.receipt_id)
    assert stored is not None and stored.intact
    assert stored.receipt == r
    assert stored.tool == "query_metric"
    assert [row["value"] for row in stored.rows] == [row["value"] for row in out.rows]


def test_same_query_same_result_hash(app):
    a = app.query_metric("revenue", group_by=["payment_type"]).receipt
    b = app.query_metric("revenue", group_by=["payment_type"]).receipt
    assert a.result_hash == b.result_hash
    assert a.sql == b.sql and a.data_snapshot_hash == b.data_snapshot_hash
    assert a.receipt_id != b.receipt_id  # executed_at differs


def test_tampered_value_is_detected(app, workspace):
    rid = app.query_metric("revenue").receipt_id
    con = duckdb.connect(str(workspace.receipts_db_path))
    con.execute("UPDATE veradim_receipts SET result = replace(result, '490.0', '590.0') WHERE receipt_id = ?", [rid])
    con.close()
    stored = app.receipts.get(rid)
    assert stored.rows[0]["value"] == 590.0
    assert not stored.intact


def test_tampered_sql_is_detected(app, workspace):
    rid = app.query_metric("revenue").receipt_id
    con = duckdb.connect(str(workspace.receipts_db_path))
    con.execute("UPDATE veradim_receipts SET sql = sql || ' ' WHERE receipt_id = ?", [rid])
    con.close()
    assert not app.receipts.get(rid).intact


def test_snapshot_hash_changes_with_data(app, workspace):
    before = app.db.snapshot_hash()
    app.db.close()
    con = duckdb.connect(str(workspace.db_path))
    con.execute("INSERT INTO orders (order_id, customer_id, order_status) VALUES ('o9', 'c1', 'created')")
    con.close()
    assert app.db.snapshot_hash() != before
