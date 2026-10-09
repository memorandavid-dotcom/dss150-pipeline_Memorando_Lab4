"""
src/storage.py
==============
Materialize pipeline data in multiple formats and load into PostgreSQL.

Formats produced:
  - CSV  : human-readable, interoperability
  - JSON/JSONL : API-friendly, document stores
  - Parquet : analytical workloads (already in curated layer)
  - PostgreSQL : query-ready, supports SQL transformations

Also handles partitioned storage (by content_type).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File-based materialization
# ---------------------------------------------------------------------------


def export_csv(df: pd.DataFrame, name: str, out_dir: Optional[Path] = None) -> Path:
    """Write DataFrame to CSV. Returns the output path."""
    out_dir = out_dir or config.Paths.curated
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.csv"
    df.to_csv(out_path, index=False)
    logger.info("CSV written → %s  (%d rows)", out_path.name, len(df))
    return out_path


def export_jsonl(df: pd.DataFrame, name: str, out_dir: Optional[Path] = None) -> Path:
    """Write DataFrame to JSON Lines format. Returns the output path."""
    out_dir = out_dir or config.Paths.curated
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"
    df.to_json(out_path, orient="records", lines=True, default_handler=str)
    logger.info("JSONL written → %s  (%d rows)", out_path.name, len(df))
    return out_path


def export_parquet(df: pd.DataFrame, name: str, out_dir: Optional[Path] = None) -> Path:
    """Write DataFrame to Parquet. Returns the output path."""
    out_dir = out_dir or config.Paths.curated
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    logger.info("Parquet written → %s  (%d rows)", out_path.name, len(df))
    return out_path


def export_all_formats(source_parquet: Optional[Path] = None) -> dict[str, Path]:
    """
    Export curated data in all required formats (Parquet, CSV, JSONL).
    The benchmark report documents format comparison.
    """
    path = source_parquet or (config.Paths.curated / "full.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Curated file not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")
    # Drop pipeline-internal audit columns for published outputs
    pub_df = df[[c for c in df.columns if not c.startswith("_")]].copy()

    return {
        "parquet": export_parquet(pub_df, "curated_published"),
        "csv": export_csv(pub_df, "curated_published"),
        "jsonl": export_jsonl(pub_df, "curated_published"),
    }


# ---------------------------------------------------------------------------
# Partitioned storage
# ---------------------------------------------------------------------------


def export_partitioned(source_parquet: Optional[Path] = None) -> Path:
    """
    Write curated data partitioned by content_type.
    Creates: data/partitioned/content_type=<value>/part-0.parquet
    """
    path = source_parquet or (config.Paths.curated / "full.parquet")
    if not path.exists():
        raise FileNotFoundError(f"Curated file not found: {path}")

    df = pd.read_parquet(path, engine="pyarrow")
    part_dir = config.Paths.partitioned
    part_dir.mkdir(parents=True, exist_ok=True)

    if "content_type" not in df.columns:
        logger.warning("'content_type' column not in curated data; skipping partition.")
        return part_dir

    # Write pyarrow partitioned parquet
    import pyarrow as pa
    import pyarrow.parquet as pq

    pub_df = df[[c for c in df.columns if not c.startswith("_")]].copy()
    table = pa.Table.from_pandas(pub_df)
    pq.write_to_dataset(
        table,
        root_path=str(part_dir),
        partition_cols=["content_type"],
        existing_data_behavior="overwrite_or_ignore",
    )
    logger.info("Partitioned Parquet written → %s", part_dir)
    return part_dir


# ---------------------------------------------------------------------------
# PostgreSQL storage
# ---------------------------------------------------------------------------


def get_engine():
    """Create and return a SQLAlchemy engine using config/env credentials."""
    url = config.DB.url()
    return create_engine(url, pool_pre_ping=True)


def load_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    schema: str = "public",
    if_exists: str = "replace",
) -> None:
    """
    Load a DataFrame into PostgreSQL.

    Args:
        df: DataFrame to load.
        table_name: Target table name.
        schema: PostgreSQL schema (default: public).
        if_exists: 'replace' | 'append' | 'fail'
    """
    engine = get_engine()
    # Convert pandas nullable types to Python native for SQLAlchemy
    df_export = df.copy()
    for col in df_export.select_dtypes(include=["Int64", "Int32"]).columns:
        df_export[col] = df_export[col].astype("int64", errors="ignore")

    df_export.to_sql(
        table_name,
        engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=500,
    )
    logger.info(
        "Loaded %d rows → PostgreSQL %s.%s  (if_exists=%s)",
        len(df),
        schema,
        table_name,
        if_exists,
    )


def load_curated_to_postgres() -> None:
    """Load the full curated dataset and train/test splits into PostgreSQL."""
    curated_dir = config.Paths.curated

    for fname, table in [
        ("full.parquet", "curated_full"),
        ("train.parquet", "curated_train"),
        ("test.parquet", "curated_test"),
    ]:
        path = curated_dir / fname
        if not path.exists():
            logger.warning("Curated file not found: %s — skipping.", fname)
            continue
        df = pd.read_parquet(path, engine="pyarrow")
        load_to_postgres(df, table_name=table)


def apply_sql_transformations() -> None:
    """Run sql/transformations.sql against the PostgreSQL database."""
    sql_path = config.Paths.root / "sql" / "transformations.sql"
    if not sql_path.exists():
        logger.warning("transformations.sql not found; skipping.")
        return
    with open(sql_path, "r", encoding="utf-8") as fh:
        sql_text = fh.read()
    engine = get_engine()
    with engine.begin() as conn:
        for stmt in sql_text.split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    logger.info("SQL transformations applied from %s", sql_path.name)
