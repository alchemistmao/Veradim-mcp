"""query_metric on the synthetic database: totals, fan-out safety, composites."""

from __future__ import annotations

import pytest

from veradim.engine.sql_builder import DimFilter, TimeRange
from veradim.errors import InvalidDimension, InvalidFilter, MetricNotApproved, MetricNotFound

TOTAL_REVENUE = 490.0


def by(rows, key):
    return {r[key]: r["value"] for r in rows}


def test_revenue_total(app):
    out = app.query_metric("revenue")
    assert out.rows[0]["value"] == pytest.approx(TOTAL_REVENUE)
    assert out.rows[0]["receipt_id"] == out.receipt_id
    assert out.receipt.row_count == 1


def test_revenue_by_payment_type_sums_to_total(app):
    """An order paid with two methods must not have its revenue counted twice."""
    out = app.query_metric("revenue", group_by=["payment_type"])
    values = by(out.rows, "payment_type")
    assert sum(values.values()) == pytest.approx(TOTAL_REVENUE, abs=0.005)
    assert values["credit_card"] == pytest.approx(150 * 120 / 165 + 40 + 30, abs=1e-6)
    assert values["voucher"] == pytest.approx(150 * 45 / 165 + 40, abs=1e-6)
    assert values["boleto"] == pytest.approx(200)
    assert values["unknown"] == pytest.approx(30)  # o5 has no payment rows
    assert any("payment_type" in n for n in out.notes)


def test_naive_join_would_inflate(app):
    """Sanity check that the fixture actually exercises fan-out."""
    _, rows = app.db.execute(
        "SELECT SUM(i.price) FROM order_items i JOIN orders o USING (order_id) "
        "JOIN order_payments p USING (order_id) WHERE o.order_status = 'delivered'"
    )
    assert float(rows[0][0]) > TOTAL_REVENUE


def test_revenue_by_category_uses_english_with_fallback(app):
    values = by(app.query_metric("revenue", group_by=["product_category"]).rows, "product_category")
    assert values == {"health_beauty": 180.0, "pc_gamer": 80.0, "unknown": 230.0}
    assert sum(values.values()) == pytest.approx(TOTAL_REVENUE)


def test_order_count_by_payment_type_is_fractional_but_sums(app):
    out = app.query_metric("order_count", group_by=["payment_type"])
    assert sum(r["value"] for r in out.rows) == pytest.approx(5)


def test_filters_and_time_range(app):
    out = app.query_metric(
        "revenue",
        time_range=TimeRange.model_validate({"start": "2018-02-01", "end": "2018-03-01"}),
        filters=[DimFilter(dimension="customer_state", op="eq", values=["SP"])],
    )
    assert out.rows[0]["value"] == pytest.approx(80.0)  # o3 only; o2 is RJ, o4 canceled
    assert out.receipt.parameters == {"start": "2018-02-01", "end": "2018-03-01", "f0_0": "SP"}


def test_filter_on_allocated_dimension(app):
    out = app.query_metric("revenue", filters=[DimFilter(dimension="payment_type", values=["voucher"])])
    assert out.rows[0]["value"] == pytest.approx(150 * 45 / 165 + 40, abs=1e-6)


def test_monthly_grain(app):
    out = app.query_metric("revenue", time_grain="month")
    assert [(r["period"], r["value"]) for r in out.rows] == [
        ("2018-01-01", 150.0),
        ("2018-02-01", 280.0),
        ("2018-03-01", 60.0),
    ]


def test_derived_margin(app):
    out = app.query_metric("margin", group_by=["customer_state"])
    rows = {r["customer_state"]: r for r in out.rows}
    assert rows["SP"]["value"] == pytest.approx((100 + 50 + 80 + 30) - (10 + 5 + 8 + 3))
    assert rows["RJ"]["value"] == pytest.approx((200 + 30) - (20 + 3))
    assert set(out.receipt.component_versions) == {"revenue", "freight"}


def test_ratio_avg_ticket(app):
    out = app.query_metric("avg_ticket")
    row = out.rows[0]
    assert row["value"] == pytest.approx(TOTAL_REVENUE / 5)
    assert row["revenue"] == pytest.approx(TOTAL_REVENUE)
    assert row["order_count"] == 5


def test_ratio_by_allocated_dimension(app):
    out = app.query_metric("avg_ticket", group_by=["payment_type"])
    for row in out.rows:
        assert row["value"] == pytest.approx(row["revenue"] / row["order_count"])


def test_dimension_not_allowed(app):
    with pytest.raises(InvalidDimension) as err:
        app.query_metric("avg_ticket", group_by=["product_category"])
    assert "customer_state" in err.value.hint


def test_filter_dimension_not_allowed(app):
    with pytest.raises(InvalidFilter):
        app.query_metric("order_count", filters=[DimFilter(dimension="seller_state", values=["SP"])])


def test_unknown_metric(app):
    with pytest.raises(MetricNotFound):
        app.query_metric("profit")


def test_proposed_metric_is_not_queryable(unapproved_workspace):
    """An agent cannot query a metric that only exists as a proposal."""
    from veradim.tools.core import VeradimApp

    app = VeradimApp(unapproved_workspace)
    assert app.list_metrics().metrics == []
    with pytest.raises(MetricNotApproved) as err:
        app.query_metric("revenue")
    assert "veradim approve revenue" in err.value.hint
    assert err.value.code == "METRIC_NOT_APPROVED"


def test_describe_metric(app):
    out = app.describe_metric("avg_ticket")
    assert [c.name for c in out.components] == ["revenue", "order_count"]
    assert out.filters == ["orders.order_status = 'delivered'"]
    alloc = {d.name: d.allocation for d in out.dimensions}
    assert alloc["payment_type"] and alloc["customer_state"] is None
    assert "LIMIT 100" in out.example_sql


def test_list_metrics(app):
    out = app.list_metrics()
    names = {m.name for m in out.metrics}
    assert names == {"revenue", "freight", "order_count", "margin", "avg_ticket"}
    assert all(m.approved_by == "Test Approver <approver@example.com>" for m in out.metrics)
    assert out.invalid == {}
