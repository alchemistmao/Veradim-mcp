"""Metric schema, registry rules and fan-out analysis."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

from veradim.errors import FanoutJoinRejected, IncompatibleComponents, SchemaViolation
from veradim.semantic.graph import AllocatedReach, DirectReach, JoinGraph
from veradim.semantic.models import DimensionCatalog
from veradim.semantic.registry import Registry, git_blob_hash, load_catalog, load_yaml, parse_metric
from veradim.semantic.validation import validate_composite, validate_simple

ROOT = Path(__file__).resolve().parents[1]
CATALOG = load_catalog(ROOT / "metrics" / "dimensions.yaml")


def proposal(name: str) -> dict:
    return load_yaml(ROOT / "metrics" / "proposed" / f"{name}.yaml")


# -- shape -------------------------------------------------------------------


@pytest.mark.parametrize("field", ["expression", "filters", "base_table", "time_column", "measure_grain"])
def test_composite_metrics_cannot_define_their_own_query_fields(field):
    data = proposal("avg_ticket") | {field: "x"}
    with pytest.raises(SchemaViolation, match="inherit"):
        parse_metric(data, source="avg_ticket.yaml")


def test_simple_metric_requires_measure_grain():
    data = proposal("revenue")
    del data["measure_grain"]
    with pytest.raises(SchemaViolation, match="measure_grain"):
        parse_metric(data, source="revenue.yaml")


def test_ratio_depends_on_must_match_numerator_and_denominator():
    data = proposal("avg_ticket") | {"depends_on": ["revenue", "freight"]}
    with pytest.raises(SchemaViolation, match="depends_on"):
        parse_metric(data, source="avg_ticket.yaml")


def test_unknown_fields_are_rejected():
    with pytest.raises(SchemaViolation, match="Extra inputs"):
        parse_metric(proposal("revenue") | {"sql": "SELECT 1"}, source="revenue.yaml")


def test_yaml_12_booleans():
    """`on:` must stay a string key (YAML 1.1 would turn it into True)."""
    raw = load_yaml(ROOT / "metrics" / "dimensions.yaml")
    assert "on" in raw["joins"][0]


# -- simple metric validation --------------------------------------------------


def test_all_proposals_are_valid():
    simple = {}
    for name in ["revenue", "freight", "order_count"]:
        metric = parse_metric(proposal(name), source=name)
        validate_simple(metric, CATALOG)
        simple[name] = metric
    for name in ["margin", "avg_ticket"]:
        validate_composite(parse_metric(proposal(name), source=name), simple)


def test_expression_must_aggregate():
    metric = parse_metric(proposal("revenue") | {"expression": "order_items.price"}, source="x")
    with pytest.raises(SchemaViolation, match="aggregation"):
        validate_simple(metric, CATALOG)


@pytest.mark.parametrize(
    "expression",
    [
        "SUM(price)",  # unqualified
        "(SELECT SUM(price) FROM order_items)",  # subquery
        "SUM(order_items.price); DROP TABLE orders",  # second statement
        "read_csv('/etc/passwd')",
    ],
)
def test_unsafe_expressions_are_rejected(expression):
    metric = parse_metric(proposal("revenue") | {"expression": expression}, source="x")
    with pytest.raises(SchemaViolation):
        validate_simple(metric, CATALOG)


def test_filter_across_fanout_is_rejected():
    """A metric filter on payments would duplicate order items."""
    data = proposal("revenue") | {"filters": ["order_payments.payment_type = 'boleto'"]}
    with pytest.raises(FanoutJoinRejected):
        validate_simple(parse_metric(data, source="x"), CATALOG)


def test_allocated_dimension_requires_additive_measure():
    data = proposal("order_count") | {"expression": "COUNT(DISTINCT orders.customer_id)"}
    with pytest.raises(FanoutJoinRejected, match="SUM"):
        validate_simple(parse_metric(data, source="x"), CATALOG)


def test_dimension_across_fanout_without_allocation_is_rejected():
    raw = load_yaml(ROOT / "metrics" / "dimensions.yaml")
    raw["dimensions"]["review_score"] = {"table": "order_reviews", "expression": "order_reviews.review_score"}
    catalog = DimensionCatalog.model_validate(raw)
    data = proposal("revenue") | {"dimensions": ["review_score"], "simulatable": []}
    with pytest.raises(FanoutJoinRejected, match="one_to_many"):
        validate_simple(parse_metric(data, source="x"), catalog)


def test_reach_classification():
    graph = JoinGraph(CATALOG)
    assert isinstance(graph.reach_dimension("order_items", "product_category"), DirectReach)
    assert isinstance(graph.reach_dimension("order_items", "customer_state"), DirectReach)
    reach = graph.reach_dimension("order_items", "payment_type")
    assert isinstance(reach, AllocatedReach)
    assert reach.at_table == "orders" and reach.at_key == ("order_id",)
    with pytest.raises(FanoutJoinRejected):
        graph.reach_dimension("orders", "product_category")


def test_simulatable_column_must_be_in_expression():
    data = proposal("revenue") | {"simulatable": [{"column": "order_items.freight_value", "operations": ["scale"]}]}
    with pytest.raises(SchemaViolation, match="not used by the expression"):
        validate_simple(parse_metric(data, source="x"), CATALOG)


# -- composite validation ----------------------------------------------------------


def _simple(name: str, **overrides):
    return parse_metric(proposal(name) | overrides, source=name)


def test_ratio_with_incompatible_filters_is_rejected():
    components = {
        "revenue": _simple("revenue"),
        "order_count": _simple("order_count", filters=["orders.order_status <> 'canceled'"]),
    }
    with pytest.raises(IncompatibleComponents, match="filters"):
        validate_composite(_simple("avg_ticket"), components)


def test_equivalent_filters_are_compatible():
    components = {
        "revenue": _simple("revenue"),
        "order_count": _simple("order_count", filters=["orders.order_status   =   'delivered'"]),
    }
    validate_composite(_simple("avg_ticket"), components)


def test_ratio_with_different_time_columns_is_rejected():
    components = {
        "revenue": _simple("revenue"),
        "order_count": _simple("order_count", time_column="orders.order_approved_at"),
    }
    with pytest.raises(IncompatibleComponents, match="time columns"):
        validate_composite(_simple("avg_ticket"), components)


def test_composite_dimension_must_exist_on_every_component():
    components = {"revenue": _simple("revenue"), "order_count": _simple("order_count")}
    ratio = _simple("avg_ticket", dimensions=["product_category"])
    with pytest.raises(IncompatibleComponents, match="product_category"):
        validate_composite(ratio, components)


def test_formula_must_match_depends_on_and_be_arithmetic():
    components = {"revenue": _simple("revenue"), "freight": _simple("freight")}
    with pytest.raises(SchemaViolation, match="depends_on"):
        validate_composite(_simple("margin", formula="revenue"), components)
    with pytest.raises(SchemaViolation):
        validate_composite(_simple("margin", formula="revenue - freight + (SELECT 1)"), components)


# -- registry ----------------------------------------------------------------------


def test_registry_ignores_proposals_and_unapproved_files(unapproved_workspace):
    metrics_dir = unapproved_workspace.metrics_dir
    # A file dropped into approved/ without going through the CLI is not trusted.
    (metrics_dir / "approved" / "revenue.yaml").write_text((metrics_dir / "proposed" / "revenue.yaml").read_text())
    registry = Registry.load(metrics_dir)
    assert registry.metrics == {}
    assert "no approval record" in registry.errors["revenue"]


def test_version_hash_matches_git_hash_object(workspace):
    path = workspace.approved_dir / "revenue.yaml"
    expected = subprocess.run(["git", "hash-object", str(path)], capture_output=True, text=True, check=True).stdout
    assert git_blob_hash(path.read_bytes()) == expected.strip()
    assert Registry.load(workspace.metrics_dir).get("revenue").version_hash == expected.strip()


def test_catalog_file_is_well_formed():
    raw = yaml.safe_load((ROOT / "metrics" / "dimensions.yaml").read_text())
    assert raw  # plain safe_load still parses the file
    assert {j.cardinality.value for j in CATALOG.joins} <= {"one_to_one", "many_to_one", "one_to_many"}
