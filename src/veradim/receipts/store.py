"""Receipts: every number Veradim returns can be traced and re-verified.

A receipt pins *what* was computed (metric + git blob hash of its definition
and of each component), *how* (the exact SQL and parameters), *on which data*
(a hash of table row counts and max timestamps) and *what came out* (a hash of
the result rows). The ``receipt_id`` is the sha256 of all of those fields, so
editing any stored field is detectable.

Receipts live in their own DuckDB file (``data/veradim_receipts.duckdb``): the
analytical database is only ever opened read-only.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb
from pydantic import BaseModel, Field

from veradim.errors import DatabaseUnavailable

FLOAT_DECIMALS = 6

_DDL = """
CREATE TABLE IF NOT EXISTS veradim_receipts (
    receipt_id          VARCHAR PRIMARY KEY,
    metric              VARCHAR NOT NULL,
    metric_version      VARCHAR NOT NULL,
    component_versions  VARCHAR NOT NULL,
    sql                 VARCHAR NOT NULL,
    parameters          VARCHAR NOT NULL,
    row_count           BIGINT NOT NULL,
    executed_at         VARCHAR NOT NULL,
    data_snapshot_hash  VARCHAR NOT NULL,
    result_hash         VARCHAR NOT NULL,
    tool                VARCHAR NOT NULL,
    result              VARCHAR NOT NULL
)
"""


class Receipt(BaseModel):
    receipt_id: str
    metric: str
    metric_version: str = Field(description="git blob hash of the metric YAML")
    component_versions: dict[str, str] = Field(default_factory=dict)
    sql: str
    parameters: dict[str, Any]
    row_count: int
    executed_at: str = Field(description="UTC ISO-8601 timestamp")
    data_snapshot_hash: str
    result_hash: str

    def expected_id(self) -> str:
        return compute_receipt_id(self.model_dump(exclude={"receipt_id"}))


def to_jsonable(value: Any) -> Any:
    """Normalize DuckDB values into canonical JSON-friendly values."""
    if isinstance(value, Decimal):
        value = float(value)
    if isinstance(value, float):
        return round(value, FLOAT_DECIMALS)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compute_result_hash(rows: list[dict[str, Any]]) -> str:
    return sha256(canonical_json(rows))


def compute_receipt_id(fields: dict[str, Any]) -> str:
    return sha256(canonical_json(fields))


def build_receipt(
    *,
    metric: str,
    metric_version: str,
    component_versions: dict[str, str],
    sql: str,
    parameters: dict[str, Any],
    rows: list[dict[str, Any]],
    data_snapshot_hash: str,
    executed_at: datetime | None = None,
) -> Receipt:
    fields = {
        "metric": metric,
        "metric_version": metric_version,
        "component_versions": dict(sorted(component_versions.items())),
        "sql": sql,
        "parameters": parameters,
        "row_count": len(rows),
        "executed_at": (executed_at or datetime.now(UTC)).isoformat(timespec="microseconds"),
        "data_snapshot_hash": data_snapshot_hash,
        "result_hash": compute_result_hash(rows),
    }
    return Receipt(receipt_id=compute_receipt_id(fields), **fields)


class StoredReceipt(BaseModel):
    receipt: Receipt
    tool: str
    rows: list[dict[str, Any]]

    @property
    def intact(self) -> bool:
        """True if neither the receipt fields nor the stored rows were modified."""
        return (
            self.receipt.expected_id() == self.receipt.receipt_id
            and compute_result_hash(self.rows) == self.receipt.result_hash
        )


class ReceiptStore:
    """Append-only store. Opens a short-lived connection per operation."""

    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            con = duckdb.connect(str(self.path))
        except duckdb.Error as exc:
            raise DatabaseUnavailable(f"cannot open receipt store {self.path}: {exc}") from exc
        con.execute(_DDL)
        return con

    def save(self, receipt: Receipt, *, tool: str, rows: list[dict[str, Any]]) -> None:
        con = self._connect()
        try:
            con.execute(
                "INSERT INTO veradim_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING",
                [
                    receipt.receipt_id,
                    receipt.metric,
                    receipt.metric_version,
                    canonical_json(receipt.component_versions),
                    receipt.sql,
                    canonical_json(receipt.parameters),
                    receipt.row_count,
                    receipt.executed_at,
                    receipt.data_snapshot_hash,
                    receipt.result_hash,
                    tool,
                    canonical_json(rows),
                ],
            )
        finally:
            con.close()

    def get(self, receipt_id: str) -> StoredReceipt | None:
        rows = self._select("WHERE receipt_id = ?", [receipt_id])
        return rows[0] if rows else None

    def recent(self, limit: int = 200) -> list[StoredReceipt]:
        return self._select("ORDER BY executed_at DESC LIMIT ?", [limit])

    def _select(self, clause: str, params: list[Any]) -> list[StoredReceipt]:
        con = self._connect()
        try:
            cur = con.execute(
                "SELECT receipt_id, metric, metric_version, component_versions, sql, parameters, row_count, "
                f"executed_at, data_snapshot_hash, result_hash, tool, result FROM veradim_receipts {clause}",
                params,
            )
            out = []
            for r in cur.fetchall():
                receipt = Receipt(
                    receipt_id=r[0],
                    metric=r[1],
                    metric_version=r[2],
                    component_versions=json.loads(r[3]),
                    sql=r[4],
                    parameters=json.loads(r[5]),
                    row_count=r[6],
                    executed_at=r[7],
                    data_snapshot_hash=r[8],
                    result_hash=r[9],
                )
                out.append(StoredReceipt(receipt=receipt, tool=r[10], rows=json.loads(r[11])))
            return out
        finally:
            con.close()
