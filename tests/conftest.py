"""Shared fixtures.

Most tests run against a tiny synthetic database with the exact Olist schema
and hand-checkable numbers (see MINI_ROWS). Tests marked ``olist`` run against
the real ingested database and are skipped when it is missing.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
from pathlib import Path

import duckdb
import pytest

from veradim.cli import approve
from veradim.config import Settings
from veradim.tools.core import VeradimApp

ROOT = Path(__file__).resolve().parents[1]
OLIST_DB = ROOT / "data" / "veradim.duckdb"
APPROVAL_ORDER = ["revenue", "freight", "order_count", "margin", "avg_ticket"]


def _ingest_module():
    spec = importlib.util.spec_from_file_location("ingest", ROOT / "scripts" / "ingest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Hand-checkable data. Delivered item revenue = 490.00.
#   o1 delivered 2018-01-10 SP  items 100 (health_beauty) + 50 (pc_gamer, untranslated)
#                               payments credit_card 120 + voucher 45 (two methods)
#   o2 delivered 2018-02-05 RJ  item 200 (no category)          payment boleto 220
#   o3 delivered 2018-02-20 SP  item 80 (health_beauty)         payments voucher 0 + credit_card 0
#   o4 canceled  2018-02-21 RJ  item 999                        payment credit_card 999
#   o5 delivered 2018-03-01 RJ  item 30 (pc_gamer)              no payments at all
#   o6 delivered 2018-03-15 SP  item 30 (no category)           credit_card 20 + credit_card 13
MINI_ROWS = {
    "customers": [
        ("c1", "u1", "01000", "sao paulo", "SP"),
        ("c2", "u2", "20000", "rio de janeiro", "RJ"),
    ],
    "sellers": [("s1", "01001", "sao paulo", "SP"), ("s2", "20001", "rio de janeiro", "RJ")],
    "products": [
        ("p1", "beleza_saude", 10, 100, 1, 500, 10, 10, 10),
        ("p2", "pc_gamer", 10, 100, 1, 500, 10, 10, 10),
        ("p3", None, 10, 100, 1, 500, 10, 10, 10),
    ],
    "product_category_name_translation": [("beleza_saude", "health_beauty")],
    "orders": [
        ("o1", "c1", "delivered", "2018-01-10 10:00:00"),
        ("o2", "c2", "delivered", "2018-02-05 10:00:00"),
        ("o3", "c1", "delivered", "2018-02-20 10:00:00"),
        ("o4", "c2", "canceled", "2018-02-21 10:00:00"),
        ("o5", "c2", "delivered", "2018-03-01 10:00:00"),
        ("o6", "c1", "delivered", "2018-03-15 10:00:00"),
    ],
    "order_items": [
        ("o1", 1, "p1", "s1", "2018-01-11 10:00:00", 100.00, 10.00),
        ("o1", 2, "p2", "s2", "2018-01-11 10:00:00", 50.00, 5.00),
        ("o2", 1, "p3", "s1", "2018-02-06 10:00:00", 200.00, 20.00),
        ("o3", 1, "p1", "s1", "2018-02-21 10:00:00", 80.00, 8.00),
        ("o4", 1, "p1", "s2", "2018-02-22 10:00:00", 999.00, 1.00),
        ("o5", 1, "p2", "s2", "2018-03-02 10:00:00", 30.00, 3.00),
        ("o6", 1, "p3", "s1", "2018-03-16 10:00:00", 30.00, 3.00),
    ],
    "order_payments": [
        ("o1", 1, "credit_card", 1, 120.00),
        ("o1", 2, "voucher", 1, 45.00),
        ("o2", 1, "boleto", 1, 220.00),
        ("o3", 1, "voucher", 1, 0.00),
        ("o3", 2, "credit_card", 1, 0.00),
        ("o4", 1, "credit_card", 1, 999.00),
        ("o6", 1, "credit_card", 1, 20.00),
        ("o6", 2, "credit_card", 1, 13.00),
    ],
    "order_reviews": [("r1", "o1", 5, None, None, "2018-01-20 00:00:00", "2018-01-21 00:00:00")],
    "geolocation": [("01000", -23.5, -46.6, "sao paulo", "SP")],
}


def build_mini_db(path: Path) -> Path:
    ingest = _ingest_module()
    con = duckdb.connect(str(path))
    try:
        for table, (_, columns, _) in ingest.TABLES.items():
            renames = ingest.RENAMES.get(table, {})
            cols = ", ".join(f"{renames.get(c, c)} {t}" for c, t in columns.items())
            con.execute(f"CREATE TABLE {table} ({cols})")
            rows = MINI_ROWS.get(table, [])
            if rows:
                width = len(columns)
                padded = [tuple(r) + (None,) * (width - len(r)) for r in rows]
                con.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * width)})", padded)
    finally:
        con.close()
    return path


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def make_workspace(tmp_path: Path, *, approve_all: bool) -> Settings:
    """A git repo with a copy of metrics/ (proposals included) and a mini DB."""
    repo = tmp_path / "repo"
    metrics = repo / "metrics"
    (metrics / "approved").mkdir(parents=True)
    (metrics / "rejected").mkdir()
    shutil.copy(ROOT / "metrics" / "dimensions.yaml", metrics / "dimensions.yaml")
    shutil.copytree(ROOT / "metrics" / "proposed", metrics / "proposed")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Test Approver")
    _git(repo, "config", "user.email", "approver@example.com")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    settings = Settings(
        db_path=build_mini_db(tmp_path / "mini.duckdb"),
        receipts_db_path=tmp_path / "receipts.duckdb",
        metrics_dir=metrics,
    )
    if approve_all:
        for name in APPROVAL_ORDER:
            approve(settings, name)
    return settings


def _copy_workspace(template: Settings, tmp_path: Path) -> Settings:
    """Copy a prebuilt workspace (repo with .git + mini DB) so each test gets its own."""
    src_root = template.db_path.parent
    dst_root = tmp_path / "ws"
    shutil.copytree(src_root, dst_root)
    return Settings(
        db_path=dst_root / template.db_path.name,
        receipts_db_path=dst_root / "receipts.duckdb",
        metrics_dir=dst_root / template.metrics_dir.relative_to(src_root),
    )


@pytest.fixture(scope="session")
def _approved_template(tmp_path_factory) -> Settings:
    return make_workspace(tmp_path_factory.mktemp("approved"), approve_all=True)


@pytest.fixture(scope="session")
def _unapproved_template(tmp_path_factory) -> Settings:
    return make_workspace(tmp_path_factory.mktemp("unapproved"), approve_all=False)


@pytest.fixture
def workspace(_approved_template: Settings, tmp_path: Path) -> Settings:
    return _copy_workspace(_approved_template, tmp_path)


@pytest.fixture
def unapproved_workspace(_unapproved_template: Settings, tmp_path: Path) -> Settings:
    return _copy_workspace(_unapproved_template, tmp_path)


@pytest.fixture
def app(workspace: Settings):
    application = VeradimApp(workspace)
    yield application
    application.db.close()


@pytest.fixture(scope="session")
def olist_workspace(tmp_path_factory) -> Settings:
    if not OLIST_DB.exists():
        pytest.skip("Olist database not ingested (run scripts/ingest.py)")
    tmp = tmp_path_factory.mktemp("olist")
    settings = make_workspace(tmp, approve_all=False)
    settings = Settings(db_path=OLIST_DB, receipts_db_path=tmp / "receipts.duckdb", metrics_dir=settings.metrics_dir)
    for name in APPROVAL_ORDER:
        approve(settings, name)
    return settings


@pytest.fixture(scope="session")
def olist_app(olist_workspace: Settings):
    application = VeradimApp(olist_workspace)
    yield application
    application.db.close()
