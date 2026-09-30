"""Pydantic models for the semantic layer YAML files.

Two kinds of files:

* ``metrics/dimensions.yaml`` -> :class:`DimensionCatalog` (grains, joins with
  cardinality, dimensions with optional fan-out allocation).
* ``metrics/{approved,proposed}/<name>.yaml`` -> :data:`Metric` (simple,
  derived or ratio).

These models only check *shape*. Cross-file rules (components exist, joins are
fan-out safe, filters are compatible...) live in :mod:`veradim.semantic.validation`.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

IDENT = r"^[a-z][a-z0-9_]*$"
QUALIFIED_COLUMN = r"^[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*$"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


# ---------------------------------------------------------------------------
# dimensions.yaml
# ---------------------------------------------------------------------------


class Cardinality(StrEnum):
    ONE_TO_ONE = "one_to_one"
    MANY_TO_ONE = "many_to_one"
    ONE_TO_MANY = "one_to_many"

    def reversed(self) -> Cardinality:
        return {
            Cardinality.ONE_TO_ONE: Cardinality.ONE_TO_ONE,
            Cardinality.MANY_TO_ONE: Cardinality.ONE_TO_MANY,
            Cardinality.ONE_TO_MANY: Cardinality.MANY_TO_ONE,
        }[self]

    @property
    def fans_out(self) -> bool:
        return self is Cardinality.ONE_TO_MANY


class Grain(_Strict):
    table: str = Field(pattern=IDENT)
    key: list[Annotated[str, Field(pattern=IDENT)]] = Field(min_length=1)


class Join(_Strict):
    from_table: str = Field(alias="from", pattern=IDENT)
    to_table: str = Field(alias="to", pattern=IDENT)
    on: list[tuple[str, str]] = Field(min_length=1, description="[[from_column, to_column], ...]")
    cardinality: Cardinality

    @field_validator("on")
    @classmethod
    def _identifiers(cls, value: list[tuple[str, str]]) -> list[tuple[str, str]]:
        import re

        for left, right in value:
            for col in (left, right):
                if not re.match(IDENT, col):
                    raise ValueError(f"join column {col!r} must be a plain identifier")
        return value


class FanoutAllocation(_Strict):
    """How to split an additive measure across members of a one_to_many dimension."""

    at: str = Field(pattern=IDENT, description="Grain the measure is pre-aggregated to")
    weight: str = Field(pattern=QUALIFIED_COLUMN, description="Column whose share splits the measure")
    zero_total: Literal["equal_split"] = "equal_split"
    missing: str = Field("unknown", description="Label for grain rows with no member at all")


class Dimension(_Strict):
    description: str = ""
    table: str = Field(pattern=IDENT)
    expression: str
    fanout_allocation: FanoutAllocation | None = None


class DimensionCatalog(_Strict):
    grains: dict[str, Grain]
    joins: list[Join]
    dimensions: dict[str, Dimension]

    @field_validator("grains", "dimensions")
    @classmethod
    def _names(cls, value: dict[str, Any]) -> dict[str, Any]:
        import re

        for name in value:
            if not re.match(IDENT, name):
                raise ValueError(f"name {name!r} must match {IDENT}")
        return value


# ---------------------------------------------------------------------------
# metric YAML
# ---------------------------------------------------------------------------


class Format(_Strict):
    type: Literal["currency", "number", "integer", "percent"]
    currency: str | None = None
    decimals: int = Field(2, ge=0, le=6)

    @model_validator(mode="after")
    def _currency_required(self) -> Format:
        if self.type == "currency" and not self.currency:
            raise ValueError("format.currency is required when format.type is 'currency'")
        return self


class Approval(_Strict):
    approved_by: str
    approved_at: datetime
    proposal_hash: str = Field(description="Git blob hash of the proposal file that was approved")
    previous_version: str | None = Field(None, description="Git blob hash of the approved file it replaced")


class SimulatableColumn(_Strict):
    column: str = Field(pattern=QUALIFIED_COLUMN)
    operations: list[Literal["scale"]] = Field(min_length=1)


class SimulatableDimension(_Strict):
    dimension: str = Field(pattern=IDENT)
    operations: list[Literal["exclude"]] = Field(min_length=1)


class _MetricBase(_Strict):
    name: str = Field(pattern=IDENT)
    description: str = Field(min_length=10)
    owner: str
    version: int = Field(ge=1, description="Human-facing version; the git blob hash is canonical")
    dimensions: list[Annotated[str, Field(pattern=IDENT)]] = Field(default_factory=list)
    format: Format
    approval: Approval | None = None


class SimpleMetric(_MetricBase):
    kind: Literal["simple"]
    measure_grain: str = Field(pattern=IDENT)
    expression: str
    time_column: str = Field(pattern=QUALIFIED_COLUMN)
    filters: list[str] = Field(default_factory=list)
    simulatable: list[SimulatableColumn | SimulatableDimension] = Field(default_factory=list)


# Fields a composite metric must not define: it inherits them from its components.
_INHERITED = ("base_table", "measure_grain", "expression", "time_column", "filters", "simulatable")


class _CompositeBase(_MetricBase):
    depends_on: list[Annotated[str, Field(pattern=IDENT)]] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def _no_own_query_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            present = [f for f in _INHERITED if f in data]
            if present:
                kind = data.get("kind", "composite")
                raise ValueError(
                    f"{kind} metrics inherit {', '.join(present)} from their component metrics; "
                    f"remove {'it' if len(present) == 1 else 'them'} from the YAML"
                )
        return data


class DerivedMetric(_CompositeBase):
    kind: Literal["derived"]
    formula: str = Field(description="Arithmetic over component metric names, e.g. 'revenue - freight'")


class RatioMetric(_CompositeBase):
    kind: Literal["ratio"]
    numerator: str = Field(pattern=IDENT)
    denominator: str = Field(pattern=IDENT)

    @model_validator(mode="after")
    def _depends_on_matches(self) -> RatioMetric:
        if set(self.depends_on) != {self.numerator, self.denominator}:
            raise ValueError("depends_on must list exactly the numerator and denominator")
        if self.numerator == self.denominator:
            raise ValueError("numerator and denominator must be different metrics")
        return self


Metric = Annotated[SimpleMetric | DerivedMetric | RatioMetric, Field(discriminator="kind")]
