"""Load the Olist Brazilian E-Commerce CSVs into DuckDB.

This is the only code path that opens the analytical database for writing.
Everything the MCP server does uses a read-only connection.

Usage:
    uv run python scripts/ingest.py [--raw data/raw] [--db data/veradim.duckdb]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]

# table name -> (csv file, {column: duckdb type}, primary key or None)
# Explicit types: zip prefixes stay VARCHAR (leading zeros), money is DECIMAL so
# sums are exact and receipts are reproducible, dates are TIMESTAMP.
TABLES: dict[str, tuple[str, dict[str, str], list[str] | None]] = {
    "customers": (
        "olist_customers_dataset.csv",
        {
            "customer_id": "VARCHAR",
            "customer_unique_id": "VARCHAR",
            "customer_zip_code_prefix": "VARCHAR",
            "customer_city": "VARCHAR",
            "customer_state": "VARCHAR",
        },
        ["customer_id"],
    ),
    "geolocation": (
        "olist_geolocation_dataset.csv",
        {
            "geolocation_zip_code_prefix": "VARCHAR",
            "geolocation_lat": "DOUBLE",
            "geolocation_lng": "DOUBLE",
            "geolocation_city": "VARCHAR",
            "geolocation_state": "VARCHAR",
        },
        None,
    ),
    "orders": (
        "olist_orders_dataset.csv",
        {
            "order_id": "VARCHAR",
            "customer_id": "VARCHAR",
            "order_status": "VARCHAR",
            "order_purchase_timestamp": "TIMESTAMP",
            "order_approved_at": "TIMESTAMP",
            "order_delivered_carrier_date": "TIMESTAMP",
            "order_delivered_customer_date": "TIMESTAMP",
            "order_estimated_delivery_date": "TIMESTAMP",
        },
        ["order_id"],
    ),
    "order_items": (
        "olist_order_items_dataset.csv",
        {
            "order_id": "VARCHAR",
            "order_item_id": "INTEGER",
            "product_id": "VARCHAR",
            "seller_id": "VARCHAR",
            "shipping_limit_date": "TIMESTAMP",
            "price": "DECIMAL(12,2)",
            "freight_value": "DECIMAL(12,2)",
        },
        ["order_id", "order_item_id"],
    ),
    "order_payments": (
        "olist_order_payments_dataset.csv",
        {
            "order_id": "VARCHAR",
            "payment_sequential": "INTEGER",
            "payment_type": "VARCHAR",
            "payment_installments": "INTEGER",
            "payment_value": "DECIMAL(12,2)",
        },
        ["order_id", "payment_sequential"],
    ),
    "order_reviews": (
        "olist_order_reviews_dataset.csv",
        {
            "review_id": "VARCHAR",
            "order_id": "VARCHAR",
            "review_score": "INTEGER",
            "review_comment_title": "VARCHAR",
            "review_comment_message": "VARCHAR",
            "review_creation_date": "TIMESTAMP",
            "review_answer_timestamp": "TIMESTAMP",
        },
        None,  # review_id is not unique in the source data
    ),
    "products": (
        "olist_products_dataset.csv",
        {
            "product_id": "VARCHAR",
            "product_category_name": "VARCHAR",
            # The source misspells "length"; we fix the column names on load.
            "product_name_lenght": "INTEGER",
            "product_description_lenght": "INTEGER",
            "product_photos_qty": "INTEGER",
            "product_weight_g": "INTEGER",
            "product_length_cm": "INTEGER",
            "product_height_cm": "INTEGER",
            "product_width_cm": "INTEGER",
        },
        ["product_id"],
    ),
    "sellers": (
        "olist_sellers_dataset.csv",
        {
            "seller_id": "VARCHAR",
            "seller_zip_code_prefix": "VARCHAR",
            "seller_city": "VARCHAR",
            "seller_state": "VARCHAR",
        },
        ["seller_id"],
    ),
    "product_category_name_translation": (
        "product_category_name_translation.csv",
        {
            "product_category_name": "VARCHAR",
            "product_category_name_english": "VARCHAR",
        },
        ["product_category_name"],
    ),
}

RENAMES = {
    "products": {
        "product_name_lenght": "product_name_length",
        "product_description_lenght": "product_description_length",
    }
}


def _sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def ingest(raw_dir: Path, db_path: Path) -> dict[str, int]:
    missing = [csv for csv, _, _ in TABLES.values() if not (raw_dir / csv).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing CSV files in {raw_dir}: {', '.join(missing)}. "
            "Download the Olist dataset (kaggle: olistbr/brazilian-ecommerce) and unzip it there."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp.duckdb")
    tmp_path.unlink(missing_ok=True)

    # Build into a temp file and swap atomically, so a failed ingest never
    # leaves a half-loaded database behind.
    con = duckdb.connect(str(tmp_path))
    counts: dict[str, int] = {}
    try:
        for table, (csv, columns, pk) in TABLES.items():
            col_spec = "{" + ", ".join(f"{_sql_str(c)}: {_sql_str(t)}" for c, t in columns.items()) + "}"
            con.execute(
                f"""
                CREATE TABLE {table} AS
                SELECT * FROM read_csv(
                    {_sql_str(str(raw_dir / csv))},
                    header = true,
                    columns = {col_spec},
                    timestampformat = '%Y-%m-%d %H:%M:%S',
                    quote = '"',
                    escape = '"'
                )
                """
            )
            for old, new in RENAMES.get(table, {}).items():
                con.execute(f"ALTER TABLE {table} RENAME COLUMN {old} TO {new}")
            if pk:
                keys = ", ".join(pk)
                dupes = con.execute(
                    f"SELECT COUNT(*) FROM (SELECT {keys} FROM {table} GROUP BY {keys} HAVING COUNT(*) > 1)"
                ).fetchone()[0]
                if dupes:
                    raise ValueError(f"{table}: {dupes} duplicate keys on ({keys})")
            counts[table] = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        con.execute("CHECKPOINT")
    finally:
        con.close()

    tmp_path.replace(db_path)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--db", type=Path, default=ROOT / "data" / "veradim.duckdb")
    args = parser.parse_args(argv)
    try:
        counts = ingest(args.raw, args.db)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    width = max(map(len, counts))
    for table, n in counts.items():
        print(f"{table:<{width}}  {n:>10,}")
    print(f"\nLoaded {len(counts)} tables into {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
