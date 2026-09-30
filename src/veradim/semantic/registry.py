"""Loads the approved metric catalog.

Only ``metrics/approved/`` is ever loaded for querying. Proposals are read only
to give a helpful error ("this metric exists but is not approved") and by the
human approval CLI.

Every approved metric is identified by the git blob hash of its YAML file
(the same value ``git hash-object <file>`` prints), so a receipt pins the exact
definition that produced a number.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import TypeAdapter, ValidationError

from veradim.errors import MetricNotApproved, MetricNotFound, SchemaViolation, VeradimError
from veradim.semantic.models import (
    DerivedMetric,
    DimensionCatalog,
    Metric,
    RatioMetric,
    SimpleMetric,
)
from veradim.semantic.validation import Schema, validate_catalog, validate_composite, validate_simple

_METRIC_ADAPTER: TypeAdapter[SimpleMetric | DerivedMetric | RatioMetric] = TypeAdapter(Metric)


def git_blob_hash(content: bytes) -> str:
    """Identical to ``git hash-object`` for a file with this content."""
    header = f"blob {len(content)}\0".encode()
    return hashlib.sha1(header + content).hexdigest()


class _Yaml12Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false.

    PyYAML follows YAML 1.1, where `on`, `off`, `yes`, `no` (and `NO`, a
    Brazilian state code in some datasets) silently become booleans. The join
    key `on:` in dimensions.yaml would otherwise load as `True:`.
    """


_Yaml12Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def load_yaml(path: Path) -> dict:
    try:
        data = yaml.load(path.read_text(encoding="utf-8"), Loader=_Yaml12Loader)
    except yaml.YAMLError as exc:
        raise SchemaViolation(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaViolation(f"{path.name}: expected a mapping at the top level")
    return data


def parse_metric(data: dict, *, source: str) -> SimpleMetric | DerivedMetric | RatioMetric:
    try:
        return _METRIC_ADAPTER.validate_python(data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'][1:]) or '<root>'}: {err['msg']}" for err in exc.errors()
        )
        raise SchemaViolation(f"{source}: {problems}") from exc


def load_catalog(path: Path, schema: Schema | None = None) -> DimensionCatalog:
    try:
        catalog = DimensionCatalog.model_validate(load_yaml(path))
    except ValidationError as exc:
        raise SchemaViolation(f"{path.name}: {exc}") from exc
    validate_catalog(catalog, schema)
    return catalog


@dataclass(frozen=True)
class LoadedMetric:
    definition: SimpleMetric | DerivedMetric | RatioMetric
    path: Path
    version_hash: str

    @property
    def name(self) -> str:
        return self.definition.name


@dataclass
class Registry:
    catalog: DimensionCatalog
    metrics: dict[str, LoadedMetric]
    errors: dict[str, str] = field(default_factory=dict)
    proposed_dir: Path | None = None

    @classmethod
    def load(
        cls,
        metrics_dir: Path,
        *,
        schema: Schema | None = None,
        require_approval: bool = True,
    ) -> Registry:
        """Load dimensions.yaml and every metric in ``metrics_dir/approved``.

        Invalid metrics are skipped and reported in ``errors`` instead of
        breaking the whole catalog. With ``require_approval`` (the default), a
        file in approved/ without an approval block is treated as invalid: it
        did not go through the human approval CLI.
        """
        catalog = load_catalog(metrics_dir / "dimensions.yaml", schema)
        parsed: dict[str, LoadedMetric] = {}
        errors: dict[str, str] = {}
        for path in sorted((metrics_dir / "approved").glob("*.yaml")):
            try:
                raw = path.read_bytes()
                metric = parse_metric(load_yaml(path), source=path.name)
                if metric.name != path.stem:
                    raise SchemaViolation(f"{path.name}: metric name {metric.name!r} must match the file name")
                if require_approval and metric.approval is None:
                    raise SchemaViolation(
                        f"{path.name}: approved metric has no approval record",
                        hint="Metrics can only be approved with the `veradim approve` CLI.",
                    )
                parsed[metric.name] = LoadedMetric(metric, path, git_blob_hash(raw))
            except VeradimError as exc:
                errors[path.stem] = exc.message

        valid = validate_all(parsed, catalog, schema, errors)
        return cls(catalog=catalog, metrics=valid, errors=errors, proposed_dir=metrics_dir / "proposed")

    # -- lookup ------------------------------------------------------------

    def get(self, name: str) -> LoadedMetric:
        found = self.metrics.get(name)
        if found is not None:
            return found
        if name in self.errors:
            raise MetricNotFound(
                f"metric {name!r} is approved but its definition is invalid: {self.errors[name]}",
                hint="Ask a human owner to fix the YAML file.",
            )
        if self.proposed_dir is not None and (self.proposed_dir / f"{name}.yaml").exists():
            raise MetricNotApproved(
                f"metric {name!r} is only a proposal and cannot be queried until a human approves it",
                hint=f"A human must run `veradim approve {name}`. Approved metrics: "
                + (", ".join(sorted(self.metrics)) or "none"),
            )
        raise MetricNotFound(
            f"unknown metric {name!r}",
            hint="Call list_metrics to see approved metrics. Approved metrics: "
            + (", ".join(sorted(self.metrics)) or "none"),
        )

    def simple(self, name: str) -> SimpleMetric:
        definition = self.get(name).definition
        assert isinstance(definition, SimpleMetric)
        return definition

    def components(self, name: str) -> list[LoadedMetric]:
        definition = self.get(name).definition
        if isinstance(definition, SimpleMetric):
            return []
        return [self.metrics[c] for c in definition.depends_on]


def validate_all(
    parsed: Mapping[str, LoadedMetric],
    catalog: DimensionCatalog,
    schema: Schema | None,
    errors: dict[str, str],
) -> dict[str, LoadedMetric]:
    """Validate simple metrics first, then composites against the valid simple ones."""
    valid: dict[str, LoadedMetric] = {}
    for name, loaded in parsed.items():
        if isinstance(loaded.definition, SimpleMetric):
            try:
                validate_simple(loaded.definition, catalog, schema)
                valid[name] = loaded
            except VeradimError as exc:
                errors[name] = exc.message
    simple = {n: m.definition for n, m in valid.items() if isinstance(m.definition, SimpleMetric)}
    for name, loaded in parsed.items():
        if not isinstance(loaded.definition, SimpleMetric):
            try:
                validate_composite(loaded.definition, simple)
                valid[name] = loaded
            except VeradimError as exc:
                errors[name] = exc.message
    return dict(sorted(valid.items()))
