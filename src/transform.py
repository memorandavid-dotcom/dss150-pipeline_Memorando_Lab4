"""
src/transform.py
================
Deterministic, idempotent transformations across three layers:

  raw      → staging  : type casting, null handling, dedup, quarantine
  staging  → curated  : feature engineering, encoding, scaling, split manifest

All functions are rerun-safe: they check for existing output and skip unless
force=True. Transformation logic is kept outside the Airflow DAG.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latest_raw_file(pattern: str = "full__") -> Optional[Path]:
    """Find the most recently written raw Parquet file matching a pattern."""
    raw_dir = config.Paths.raw
    candidates = sorted(raw_dir.glob(f"{pattern}*.parquet"), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


# ---------------------------------------------------------------------------
# RAW → STAGING
# ---------------------------------------------------------------------------

EXPECTED_TYPES: dict[str, str] = {
    "word_count": "int",
    "character_count": "int",
    "sentence_count": "int",
    "lexical_diversity": "float",
    "avg_sentence_length": "float",
    "avg_word_length": "float",
    "punctuation_ratio": "float",
    "flesch_reading_ease": "float",
    "gunning_fog_index": "float",
    "grammar_errors": "int",
    "passive_voice_ratio": "float",
    "predictability_score": "float",
    "burstiness": "float",
    "sentiment_score": "float",
    "label": "int",
}


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to declared types; coerce bad values to NaN."""
    for col, dtype in EXPECTED_TYPES.items():
        if col not in df.columns:
            continue
        if dtype == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        elif dtype == "float":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def _impute_nulls(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute numeric nulls with column median.
    Rows that are null on the target column are quarantined.
    Returns (clean_df, quarantine_df).
    """
    target = config.Model.target_column
    quarantine_mask = df[target].isna()
    quarantine_df = df[quarantine_mask].copy()
    clean_df = df[~quarantine_mask].copy()

    numeric_cols = config.Model.numeric_features
    for col in numeric_cols:
        if col in clean_df.columns and clean_df[col].isna().any():
            median_val = clean_df[col].median()
            n_imputed = clean_df[col].isna().sum()
            clean_df[col] = clean_df[col].fillna(median_val)
            logger.info("Imputed %d nulls in '%s' with median=%.4f", n_imputed, col, median_val)

    return clean_df, quarantine_df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on _row_hash if present, otherwise full row."""
    before = len(df)
    if "_row_hash" in df.columns:
        df = df.drop_duplicates(subset=["_row_hash"])
    else:
        df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


def transform_raw_to_staging(
    raw_path: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """
    Transform the raw ingestion snapshot into the staging layer.

    Steps:
      1. Load latest full raw snapshot (or provided path)
      2. Cast column types per schema
      3. Remove duplicates
      4. Impute numeric nulls; quarantine rows missing target
      5. Add staging audit columns
      6. Write to data/staging/staging.parquet

    Returns path to staging file.
    """
    config.ensure_dirs()

    raw_path = raw_path or _latest_raw_file("full__")
    if raw_path is None:
        raise FileNotFoundError("No raw snapshot found. Run ingest_full() first.")

    staging_path = config.Paths.staging / "staging.parquet"
    if staging_path.exists() and not force:
        logger.info("Staging file already exists. Skipping (use force=True to rerun).")
        return staging_path

    logger.info("Loading raw snapshot: %s", raw_path.name)
    df = pd.read_parquet(raw_path, engine="pyarrow")

    # --- Transformations ---
    df = _cast_types(df)
    df = _deduplicate(df)
    df, quarantine_df = _impute_nulls(df)

    # --- Quarantine bad rows ---
    if len(quarantine_df):
        q_path = config.Paths.quarantine / f"quarantine_{_now_utc()[:10]}.parquet"
        q_path.parent.mkdir(parents=True, exist_ok=True)
        quarantine_df["_quarantine_reason"] = "null_target"
        quarantine_df.to_parquet(q_path, index=False)
        logger.warning("Quarantined %d rows → %s", len(quarantine_df), q_path.name)

    # --- Staging audit ---
    df["_staged_at_utc"] = _now_utc()
    df["_staging_version"] = "1.0"

    # Write
    df.to_parquet(staging_path, index=False, engine="pyarrow")
    logger.info("Staging complete → %s  (%d rows)", staging_path.name, len(df))
    return staging_path


# ---------------------------------------------------------------------------
# STAGING → CURATED
# ---------------------------------------------------------------------------


def transform_staging_to_curated(
    staging_path: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Path]:
    """
    Transform staging data into the curated (model-ready) layer.

    Steps:
      1. Load staging Parquet
      2. Encode categorical features (LabelEncoder for content_type)
      3. Scale numeric features (StandardScaler — fit on train, apply to test)
      4. Perform deterministic train/test split (fixed seed + split manifest)
      5. Write curated/train.parquet, curated/test.parquet, curated/full.parquet
      6. Save split_manifest.json

    Returns dict of output paths.
    """
    config.ensure_dirs()
    staging_path = staging_path or (config.Paths.staging / "staging.parquet")
    if not staging_path.exists():
        raise FileNotFoundError("Staging file not found. Run transform_raw_to_staging() first.")

    train_path = config.Paths.curated / "train.parquet"
    test_path = config.Paths.curated / "test.parquet"
    full_path = config.Paths.curated / "full.parquet"
    manifest_path = config.Paths.metadata / "split_manifest.json"

    if train_path.exists() and test_path.exists() and not force:
        logger.info("Curated splits already exist. Skipping (use force=True to rerun).")
        return {"train": train_path, "test": test_path, "full": full_path}

    logger.info("Loading staging: %s", staging_path.name)
    df = pd.read_parquet(staging_path, engine="pyarrow")

    # --- Feature selection ---
    numeric_features = config.Model.numeric_features
    categorical_features = config.Model.categorical_features
    target_col = config.Model.target_column

    available_features = [
        f for f in numeric_features + categorical_features if f in df.columns
    ]
    feature_df = df[available_features + [target_col]].copy()

    # --- Encode categoricals ---
    label_encoders: dict[str, LabelEncoder] = {}
    for cat_col in categorical_features:
        if cat_col not in feature_df.columns:
            continue
        le = LabelEncoder()
        feature_df[cat_col] = le.fit_transform(feature_df[cat_col].astype(str))
        label_encoders[cat_col] = le
        logger.info("Encoded '%s': classes=%s", cat_col, list(le.classes_))

    # --- Deterministic train/test split ---
    seed = config.Model.random_seed
    test_size = config.Model.test_size

    X = feature_df.drop(columns=[target_col])
    y = feature_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    logger.info(
        "Split: %d train / %d test  (seed=%d, test_size=%.2f, stratified)",
        len(X_train),
        len(X_test),
        seed,
        test_size,
    )

    # --- Standard scaling (fit on train only) ---
    scaler = StandardScaler()
    numeric_available = [f for f in numeric_features if f in X_train.columns]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[numeric_available] = scaler.fit_transform(X_train[numeric_available])
    X_test_scaled[numeric_available] = scaler.transform(X_test[numeric_available])

    # Reassemble
    train_df = X_train_scaled.copy()
    train_df[target_col] = y_train.values
    train_df["_split"] = "train"
    train_df["_original_index"] = X_train.index

    test_df = X_test_scaled.copy()
    test_df[target_col] = y_test.values
    test_df["_split"] = "test"
    test_df["_original_index"] = X_test.index

    curated_at = _now_utc()
    for d in [train_df, test_df]:
        d["_curated_at_utc"] = curated_at

    full_curated_df = pd.concat([train_df, test_df], ignore_index=True)

    # --- Write ---
    train_df.to_parquet(train_path, index=False, engine="pyarrow")
    test_df.to_parquet(test_path, index=False, engine="pyarrow")
    full_curated_df.to_parquet(full_path, index=False, engine="pyarrow")
    logger.info("Curated splits written.")

    # --- Split manifest ---
    manifest = {
        "created_at": curated_at,
        "random_seed": seed,
        "test_size": test_size,
        "stratified": True,
        "total_rows": len(df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": available_features,
        "target": target_col,
        "train_indices": X_train.index.tolist(),
        "test_indices": X_test.index.tolist(),
        "scaler_mean": dict(zip(numeric_available, scaler.mean_.tolist())),
        "scaler_std": dict(zip(numeric_available, scaler.scale_.tolist())),
        "label_encoders": {
            col: list(le.classes_) for col, le in label_encoders.items()
        },
    }
    config.Paths.metadata.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Split manifest saved → %s", manifest_path.name)

    return {"train": train_path, "test": test_path, "full": full_path}
