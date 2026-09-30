"""Structured errors that an agent can read and recover from.

Every error carries a stable machine-readable ``code``, a human ``message`` and
a ``hint`` describing how to fix the request. Tools serialize them to JSON.
"""

from __future__ import annotations

import json
from typing import Any


class VeradimError(Exception):
    code: str = "VERADIM_ERROR"

    def __init__(self, message: str, *, hint: str = "", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.hint:
            payload["hint"] = self.hint
        if self.details:
            payload["details"] = self.details
        return {"error": payload}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


class SchemaViolation(VeradimError):
    """A metric or dimension YAML file does not satisfy the schema."""

    code = "SCHEMA_VIOLATION"


class IncompatibleComponents(VeradimError):
    """A derived or ratio metric combines components that cannot be combined."""

    code = "INCOMPATIBLE_COMPONENTS"


class MetricNotFound(VeradimError):
    code = "METRIC_NOT_FOUND"


class MetricNotApproved(VeradimError):
    """The metric exists only as a proposal and is therefore not queryable."""

    code = "METRIC_NOT_APPROVED"


class InvalidDimension(VeradimError):
    code = "INVALID_DIMENSION"


class InvalidFilter(VeradimError):
    code = "INVALID_FILTER"


class InvalidTimeRange(VeradimError):
    code = "INVALID_TIME_RANGE"


class FanoutJoinRejected(VeradimError):
    """A dimension is reachable only through a one_to_many join without an allocation rule."""

    code = "FANOUT_JOIN_REJECTED"


class UnsafeSQL(VeradimError):
    """Generated SQL failed validation (not a SELECT, unknown table/column, missing LIMIT...)."""

    code = "UNSAFE_SQL"


class QueryFailed(VeradimError):
    code = "QUERY_FAILED"


class DatabaseUnavailable(VeradimError):
    code = "DATABASE_UNAVAILABLE"


class ApprovalError(VeradimError):
    code = "APPROVAL_ERROR"
