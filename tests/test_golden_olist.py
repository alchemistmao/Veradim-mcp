"""Golden tests against the real Olist dataset (skipped if not ingested).

Reference figures are the published row counts of the Kaggle dataset
(olistbr/brazilian-ecommerce) and totals computed directly from the raw
tables, independently of the semantic layer.
"""

from __future__ import annotations

import pytest

from veradim.engine.connection import ReadOnlyDatabase
from veradim.semantic.graph import JoinGraph

pytestmark = pytest.mark.olist

ROW_COUNTS = {
    "orders": 99_441,
    "order_items": 112_650,
    "order_payments": 103_886,
    "order_reviews": 99_224,
    "customers": 99_441,
    "products": 32_951,
    "sellers": 3_095,
    "geolocation": 1_000_163,
    "product_category_name_translation": 71,
}


def scalar(app, sql):
    return app.db.execute(sql)[1][0][0]


def test_row_counts(olist_app):
    for table, expected in ROW_COUNTS.items():
        assert scalar(olist_app, f"SELECT COUNT(*) FROM {table}") == expected, table


def test_known_totals(olist_app):
    assert float(scalar(olist_app, "SELECT SUM(price) FROM order_items")) == pytest.approx(13_591_643.70, abs=0.01)
    assert float(scalar(olist_app, "SELECT SUM(freight_value) FROM order_items")) == pytest.approx(
        2_251_909.54, abs=0.01
    )
    assert float(scalar(olist_app, "SELECT SUM(payment_value) FROM order_payments")) == pytest.approx(
        16_008_872.12, abs=0.01
    )
    assert scalar(olist_app, "SELECT COUNT(*) FROM orders WHERE order_status = 'delivered'") == 96_478


def test_revenue_matches_raw_sql(olist_app):
    raw = float(
        scalar(
            olist_app,
            "SELECT SUM(i.price) FROM order_items i JOIN orders o USING (order_id) WHERE o.order_status = 'delivered'",
        )
    )
    out = olist_app.query_metric("revenue")
    assert out.rows[0]["value"] == pytest.approx(raw, abs=0.01)
    assert olist_app.query_metric("order_count").rows[0]["value"] == 96_478


@pytest.mark.parametrize("dimension", ["payment_type", "product_category", "customer_state", "seller_state"])
def test_revenue_split_sums_to_total(olist_app, dimension):
    """No dimension may inflate or lose revenue: the split always sums to the total, to the cent."""
    total = olist_app.query_metric("revenue").rows[0]["value"]
    rows = olist_app.query_metric("revenue", group_by=[dimension], limit=10_000).rows
    assert sum(r["value"] for r in rows) == pytest.approx(total, abs=0.01)


def test_product_category_is_english_with_portuguese_fallback(olist_app):
    rows = olist_app.query_metric("revenue", group_by=["product_category"], limit=10_000).rows
    names = {r["product_category"] for r in rows}
    assert "health_beauty" in names and "beleza_saude" not in names
    # These two categories have no English translation in the source data.
    assert {"pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos"} <= names


def test_avg_ticket_is_revenue_over_orders(olist_app):
    row = olist_app.query_metric("avg_ticket").rows[0]
    assert row["value"] == pytest.approx(row["revenue"] / row["order_count"])


def test_declared_cardinalities_hold(olist_workspace):
    """Every many_to_one target key must be unique; otherwise the fan-out analysis is wrong."""
    from veradim.semantic.registry import load_catalog

    catalog = load_catalog(olist_workspace.metrics_dir / "dimensions.yaml")
    db = ReadOnlyDatabase(olist_workspace.db_path)
    try:
        for join in catalog.joins:
            if join.cardinality.value == "one_to_many":
                unique_table, cols = join.from_table, [a for a, _ in join.on]
            else:
                unique_table, cols = join.to_table, [b for _, b in join.on]
            keys = ", ".join(cols)
            dupes = db.execute(
                f"SELECT COUNT(*) FROM (SELECT {keys} FROM {unique_table} GROUP BY {keys} HAVING COUNT(*) > 1)"
            )[1][0][0]
            assert dupes == 0, f"{unique_table}({keys}) is not unique but {join.cardinality.value} was declared"
        assert JoinGraph(catalog).tables <= set(db.column_sets())
    finally:
        db.close()
