"""
src/cli.py
==========
Command-line entry point for the pipeline.

Usage examples:
  python -m src.cli ingest --mode full
  python -m src.cli ingest --mode incremental
  python -m src.cli transform
  python -m src.cli storage --skip-postgres
  python -m src.cli validate
  python -m src.cli train [--force]
  python -m src.cli evaluate
  python -m src.cli run-all [--force]

All commands configure structured logging before executing.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """DSS150P Lab 4 - AI vs Human Text Detection Pipeline."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


# --------------------------------------------------------------------------
# ingest
# --------------------------------------------------------------------------


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["full", "incremental"], case_sensitive=False),
    default="full",
    show_default=True,
    help="Ingestion mode.",
)
@click.option("--force", is_flag=True, default=False, help="Force re-ingestion.")
def ingest(mode: str, force: bool) -> None:
    """Ingest source dataset into the raw layer."""
    from src.ingest import ingest_full, ingest_incremental

    if mode == "full":
        path = ingest_full(force=force)
        click.echo(f"[OK] Full ingestion -> {path}")
    else:
        paths = ingest_incremental(force=force)
        if paths:
            click.echo(f"[OK] Incremental ingestion: {len(paths)} new batch(es)")
        else:
            click.echo("[INFO] No new batches to ingest (all already committed).")


# --------------------------------------------------------------------------
# transform
# --------------------------------------------------------------------------


@cli.command()
@click.option("--force", is_flag=True, default=False, help="Force re-transformation.")
def transform(force: bool) -> None:
    """Transform raw -> staging -> curated."""
    from src.transform import transform_raw_to_staging, transform_staging_to_curated

    staging = transform_raw_to_staging(force=force)
    click.echo(f"[OK] Staging -> {staging}")
    paths = transform_staging_to_curated(force=force)
    click.echo(f"[OK] Curated -> {paths['train']} / {paths['test']}")


# --------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------


@cli.command()
def validate() -> None:
    """Run all data quality checks on the staging dataset."""
    import pandas as pd
    from src.validate import run_all_checks
    from src import config

    staging_path = config.Paths.staging / "staging.parquet"
    if not staging_path.exists():
        click.echo("[ERROR] Staging file not found. Run 'transform' first.")
        sys.exit(1)

    df = pd.read_parquet(staging_path, engine="pyarrow")
    results = run_all_checks(df, write_report=True)

    fail_count = sum(1 for r in results if r["status"] == "FAIL")
    warn_count = sum(1 for r in results if r["status"] == "WARN")
    pass_count = sum(1 for r in results if r["status"] == "PASS")

    click.echo(f"\n[OK] PASS: {pass_count}  [WARN] WARN: {warn_count}  [FAIL] FAIL: {fail_count}")
    if fail_count > 0:
        sys.exit(1)


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------


@cli.command()
@click.option("--skip-postgres", is_flag=True, default=False, help="Skip PostgreSQL loading.")
def storage(skip_postgres: bool) -> None:
    """Export curated data to all formats and optionally load into PostgreSQL."""
    from src.storage import export_all_formats, export_partitioned, load_curated_to_postgres

    paths = export_all_formats()
    for fmt, p in paths.items():
        click.echo(f"[OK] {fmt.upper()} -> {p}")

    export_partitioned()
    click.echo("[OK] Partitioned Parquet written.")

    if not skip_postgres:
        try:
            load_curated_to_postgres()
            click.echo("[OK] Loaded to PostgreSQL.")
        except Exception as exc:
            click.echo(f"[WARN] PostgreSQL load skipped: {exc}")


# --------------------------------------------------------------------------
# train
# --------------------------------------------------------------------------


@cli.command()
@click.option("--force", is_flag=True, default=False, help="Force re-training.")
def train(force: bool) -> None:
    """Train all classifiers defined in config/settings.yml."""
    from src.train import train_all

    model_paths = train_all(force=force)
    for name, p in model_paths.items():
        click.echo(f"[OK] {name} -> {p}")


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


@cli.command()
def evaluate() -> None:
    """Evaluate trained models on the test split."""
    from src.evaluate import evaluate_all

    metrics = evaluate_all()
    click.echo(f"\n[OK] Evaluation complete. {len(metrics)} classifier(s) evaluated.")


# --------------------------------------------------------------------------
# run-all
# --------------------------------------------------------------------------


@cli.command("run-all")
@click.option("--force", is_flag=True, default=False, help="Force re-run of all stages.")
@click.option("--skip-postgres", is_flag=True, default=False)
def run_all(force: bool, skip_postgres: bool) -> None:
    """Execute the full pipeline end-to-end."""
    from src import config
    from src.ingest import ingest_full, ingest_incremental
    from src.transform import transform_raw_to_staging, transform_staging_to_curated
    from src.storage import export_all_formats, export_partitioned, load_curated_to_postgres
    from src.validate import run_all_checks
    from src.train import train_all
    from src.evaluate import evaluate_all
    import pandas as pd

    config.ensure_dirs()

    click.echo("=== [1/7] Ingest (full) ===")
    ingest_full(force=force)

    click.echo("=== [2/7] Ingest (incremental) ===")
    ingest_incremental(force=force)

    click.echo("=== [3/7] Transform ===")
    transform_raw_to_staging(force=force)
    transform_staging_to_curated(force=force)

    click.echo("=== [4/7] Validate ===")
    staging_df = pd.read_parquet(config.Paths.staging / "staging.parquet")
    run_all_checks(staging_df, write_report=True)

    click.echo("=== [5/7] Storage ===")
    export_all_formats()
    export_partitioned()
    if not skip_postgres:
        try:
            load_curated_to_postgres()
        except Exception as exc:
            click.echo(f"[WARN] PostgreSQL skipped: {exc}")

    click.echo("=== [6/7] Train ===")
    train_all(force=force)

    click.echo("=== [7/7] Evaluate ===")
    metrics = evaluate_all()

    click.echo("\n[OK] Pipeline complete.")


# --------------------------------------------------------------------------
# check-source
# --------------------------------------------------------------------------


@cli.command("check-source")
def check_source() -> None:
    """Verify the source dataset checksum."""
    from src.ingest import verify_source

    is_valid, checksum = verify_source()
    if is_valid:
        click.echo(f"[OK] Checksum valid: {checksum}")
    else:
        click.echo(f"[FAIL] Checksum MISMATCH: {checksum}")
        sys.exit(1)


if __name__ == "__main__":
    cli()
