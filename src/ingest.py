"""
src/ingest.py
=============
Data ingestion module.

Responsibilities:
  - Full ingestion: copy immutable source to timestamped raw snapshot
  - Incremental simulation: split source into deterministic batches,
    track a checkpoint/watermark, skip already-committed batches on rerun
  - Audit metadata: pipeline_run_id, ingested_at_utc, source_file, checksum

Rerun-safe: ingesting the same batch twice is a no-op unless --force is used.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65_536), b""):
            h.update(chunk)
    return h.hexdigest()


def _new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def _load_checkpoint() -> dict:
    cp = config.Paths.checkpoint
    if cp.exists():
        with open(cp, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {"committed_batches": [], "last_full_run": None}


def _save_checkpoint(state: dict) -> None:
    cp = config.Paths.checkpoint
    cp.parent.mkdir(parents=True, exist_ok=True)
    with open(cp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    logger.debug("Checkpoint saved: %s", state)


# ---------------------------------------------------------------------------
# Source verification
# ---------------------------------------------------------------------------


def verify_source(source_path: Optional[Path] = None) -> tuple[bool, str]:
    """
    Verify that the source file matches the expected SHA-256 checksum.

    Returns (is_valid, actual_checksum).
    """
    path = source_path or config.Paths.source_primary
    if not path.exists():
        raise FileNotFoundError(
            f"Source dataset not found at: {path}\n"
            "Place the file according to the README and try again."
        )
    actual = _sha256(path)
    expected = config.get("source.primary_sha256")
    is_valid = actual == expected
    if not is_valid:
        logger.warning(
            "Checksum mismatch! Expected %s, got %s for %s",
            expected,
            actual,
            path.name,
        )
    else:
        logger.info("Source checksum verified ✓  [%s]", actual[:16] + "…")
    return is_valid, actual


# ---------------------------------------------------------------------------
# Full ingestion
# ---------------------------------------------------------------------------


def ingest_full(
    source_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    force: bool = False,
) -> Path:
    """
    Copy the immutable source dataset to data/raw/ with audit metadata.

    The raw snapshot is never modified. A new snapshot is only written if
    no snapshot for this run_id exists (or if force=True).

    Returns the path to the raw snapshot file.
    """
    config.ensure_dirs()
    source = source_path or config.Paths.source_primary
    run_id = run_id or _new_run_id()

    # Verify source identity
    is_valid, checksum = verify_source(source)
    if not is_valid:
        logger.error("Source checksum mismatch. Aborting ingestion.")
        raise ValueError("Source dataset checksum does not match expected value.")

    # Output path (include run_id so reruns don't overwrite)
    raw_dir = config.Paths.raw
    snapshot_name = f"full__{run_id}__{source.stem}.parquet"
    snapshot_path = raw_dir / snapshot_name

    if snapshot_path.exists() and not force:
        logger.info("Raw snapshot already exists for run_id=%s. Skipping.", run_id)
        return snapshot_path

    # Read source
    logger.info("Reading source: %s", source)
    df = pd.read_csv(source, dtype_backend="numpy_nullable")

    # Attach audit metadata
    df["_pipeline_run_id"] = run_id
    df["_ingested_at_utc"] = _now_utc()
    df["_source_file"] = source.name
    df["_source_checksum"] = checksum
    df["_batch_id"] = "full"
    df["_row_hash"] = df.apply(
        lambda r: hashlib.md5(  # noqa: S324
            "|".join(str(v) for v in r[: len(df.columns) - 5]).encode()
        ).hexdigest(),
        axis=1,
    )

    # Save as Parquet (columnar, efficient)
    df.to_parquet(snapshot_path, index=False, engine="pyarrow")
    logger.info("Full ingestion complete → %s  (%d rows)", snapshot_path.name, len(df))

    # Update checkpoint
    state = _load_checkpoint()
    state["last_full_run"] = {
        "run_id": run_id,
        "snapshot": snapshot_path.name,
        "rows": len(df),
        "checksum": checksum,
        "ingested_at": _now_utc(),
    }
    _save_checkpoint(state)

    return snapshot_path


# ---------------------------------------------------------------------------
# Incremental ingestion (simulation)
# ---------------------------------------------------------------------------


def _make_batches(df: pd.DataFrame, n_batches: int) -> list[pd.DataFrame]:
    """
    Split a DataFrame into n deterministic, non-overlapping batches.
    The split is based on row index (deterministic — same data → same batches).
    """
    size = len(df)
    batch_size = size // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size if i < n_batches - 1 else size
        batches.append(df.iloc[start:end].copy().reset_index(drop=True))
    return batches


def ingest_incremental(
    source_path: Optional[Path] = None,
    run_id: Optional[str] = None,
    force: bool = False,
) -> list[Path]:
    """
    Simulate incremental ingestion from the same source dataset.

    The source is split into config.n_batches deterministic chunks.
    Already-committed batch IDs are skipped unless force=True.

    Returns list of paths to newly written batch Parquet files.
    """
    config.ensure_dirs()
    source = source_path or config.Paths.source_primary
    run_id = run_id or _new_run_id()
    n_batches: int = config.get("source.n_batches", 3)

    is_valid, checksum = verify_source(source)
    if not is_valid:
        raise ValueError("Source checksum mismatch. Cannot proceed.")

    df_source = pd.read_csv(source, dtype_backend="numpy_nullable")
    batches = _make_batches(df_source, n_batches)

    state = _load_checkpoint()
    committed: list[str] = state.get("committed_batches", [])
    written_paths: list[Path] = []

    for idx, batch_df in enumerate(batches):
        batch_id = f"batch_{idx + 1:02d}_of_{n_batches:02d}"

        if batch_id in committed and not force:
            logger.info("Batch %s already committed. Skipping (watermark hit).", batch_id)
            continue

        # Audit columns
        batch_df["_pipeline_run_id"] = run_id
        batch_df["_ingested_at_utc"] = _now_utc()
        batch_df["_source_file"] = source.name
        batch_df["_source_checksum"] = checksum
        batch_df["_batch_id"] = batch_id

        # Row-level dedup hash (deterministic from data values)
        feature_cols = [c for c in batch_df.columns if not c.startswith("_")]
        batch_df["_row_hash"] = batch_df[feature_cols].apply(
            lambda r: hashlib.md5(  # noqa: S324
                "|".join(str(v) for v in r).encode()
            ).hexdigest(),
            axis=1,
        )

        # Write
        raw_dir = config.Paths.raw
        out_path = raw_dir / f"incremental__{batch_id}__{run_id}.parquet"
        batch_df.to_parquet(out_path, index=False, engine="pyarrow")
        logger.info(
            "Batch %s ingested → %s  (%d rows)", batch_id, out_path.name, len(batch_df)
        )

        # Commit watermark
        if batch_id not in committed:
            committed.append(batch_id)
        written_paths.append(out_path)

    state["committed_batches"] = committed
    state["last_incremental_run"] = {
        "run_id": run_id,
        "new_batches": [p.name for p in written_paths],
        "ingested_at": _now_utc(),
    }
    _save_checkpoint(state)

    if not written_paths:
        logger.info("No new batches to ingest. All %d batches already committed.", n_batches)

    return written_paths
