"""Read-only access to the analytical database.

The connection is opened with ``read_only=True`` and external access disabled
(no reading arbitrary files, no extensions, no network), and the configuration
is locked so a query cannot turn those settings back on. Ingestion is a
separate script and is the only writer.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import duckdb

from veradim.errors import DatabaseUnavailable, QueryFailed


class ReadOnlyDatabase:
    def __init__(self, path: Path):
        self.path = path
        self._con: duckdb.DuckDBPyConnection | None = None
        self._lock = threading.Lock()
        self._schema: dict[str, dict[str, str]] | None = None
        self._snapshot: tuple[tuple[float, int], str] | None = None

    def _connection(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            if not self.path.exists():
                raise DatabaseUnavailable(
                    f"analytical database not found at {self.path}",
                    hint="Run `uv run python scripts/ingest.py` to build it from data/raw/.",
                )
            try:
                self._con = duckdb.connect(
                    str(self.path),
                    read_only=True,
                    config={"enable_external_access": False, "lock_configuration": True},
                )
            except duckdb.Error as exc:
                raise DatabaseUnavailable(f"cannot open {self.path} read-only: {exc}") from exc
        return self._con

    def close(self) -> None:
        with self._lock:
            if self._con is not None:
                self._con.close()
                self._con = None

    def execute(self, sql: str, params: dict[str, Any] | None = None) -> tuple[list[str], list[tuple]]:
        with self._lock:
            con = self._connection()
            try:
                cur = con.execute(sql, params or {})
                columns = [d[0] for d in cur.description]
                return columns, cur.fetchall()
            except duckdb.Error as exc:
                raise QueryFailed(
                    f"query failed: {exc}", hint="This is likely a bug in the metric definition."
                ) from exc

    def schema(self) -> dict[str, dict[str, str]]:
        """table -> {column: type} for every base table in the main schema."""
        if self._schema is None:
            _, rows = self.execute(
                "SELECT table_name, column_name, data_type FROM information_schema.columns "
                "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position"
            )
            schema: dict[str, dict[str, str]] = {}
            for table, column, dtype in rows:
                schema.setdefault(table, {})[column] = dtype
            self._schema = schema
        return self._schema

    def column_sets(self) -> dict[str, set[str]]:
        return {t: set(cols) for t, cols in self.schema().items()}

    def snapshot_hash(self) -> str:
        """Hash of every table's row count and max timestamp values.

        Changes whenever data is re-ingested with different content. Cached by
        the database file's mtime and size.
        """
        stat = self.path.stat()
        key = (stat.st_mtime, stat.st_size)
        if self._snapshot and self._snapshot[0] == key:
            return self._snapshot[1]
        state: dict[str, dict[str, Any]] = {}
        for table, cols in sorted(self.schema().items()):
            ts_cols = [c for c, t in cols.items() if t.startswith("TIMESTAMP") or t == "DATE"]
            selects = ["COUNT(*)"] + [f'MAX("{c}")' for c in ts_cols]
            _, rows = self.execute(f'SELECT {", ".join(selects)} FROM "{table}"')
            values = rows[0]
            state[table] = {"rows": values[0], **{c: str(v) for c, v in zip(ts_cols, values[1:], strict=True)}}
        digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        self._snapshot = (key, digest)
        return digest
