"""Runtime configuration.

Every path can be overridden with an environment variable so tests (and other
deployments) can point Veradim at their own database and metric catalog.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LIMIT = 1000
MAX_LIMIT = 10_000


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser().resolve() if value else default


@dataclass(frozen=True)
class Settings:
    db_path: Path = field(
        default_factory=lambda: _path_from_env("VERADIM_DB", PROJECT_ROOT / "data" / "veradim.duckdb")
    )
    receipts_db_path: Path = field(
        default_factory=lambda: _path_from_env("VERADIM_RECEIPTS_DB", PROJECT_ROOT / "data" / "veradim_receipts.duckdb")
    )
    metrics_dir: Path = field(default_factory=lambda: _path_from_env("VERADIM_METRICS_DIR", PROJECT_ROOT / "metrics"))

    @property
    def approved_dir(self) -> Path:
        return self.metrics_dir / "approved"

    @property
    def proposed_dir(self) -> Path:
        return self.metrics_dir / "proposed"

    @property
    def rejected_dir(self) -> Path:
        return self.metrics_dir / "rejected"

    @property
    def dimensions_file(self) -> Path:
        return self.metrics_dir / "dimensions.yaml"
