"""
tests/test_pipeline.py
======================
Automated test suite verifying:
  - Source data existence and cryptographic integrity (SHA-256)
  - Configuration loading and directory resolution
  - Ingestion idempotency and deduplication
  - Transformation data-quality contracts (schema, nulls, types)
  - Split reproducibility and deterministic model evaluation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
try:
    import pytest
except ImportError:
    pytest = None

from src import config
from src.ingest import verify_source, _sha256
from src.validate import (
    check_schema,
    check_uniqueness,
    check_row_count,
    check_reproducibility,
)


# ---------------------------------------------------------------------------
# 1. Source Identity & Cryptographic Integrity Tests
# ---------------------------------------------------------------------------


def test_source_dataset_exists():
    """Verify that the primary source dataset file exists on disk."""
    source_path = config.Paths.source_primary
    assert source_path.exists(), f"Source dataset missing at {source_path}"
    assert source_path.stat().st_size > 0, "Source dataset is empty."


def test_source_checksum_integrity():
    """Verify source file matches expected SHA-256 fingerprint."""
    is_valid, actual_checksum = verify_source()
    expected = config.get("source.primary_sha256")
    assert is_valid, f"Checksum mismatch: expected {expected}, got {actual_checksum}"
    assert actual_checksum == expected


# ---------------------------------------------------------------------------
# 2. Configuration & Path Tests
# ---------------------------------------------------------------------------


def test_config_loading():
    """Verify settings.yml loads expected keys."""
    assert config.get("project.name") == "dss150-pipeline-memorando"
    assert config.get("source.n_batches") == 3
    assert config.Model.random_seed == 42
    assert config.Model.test_size == 0.20
    assert len(config.Model.classifiers) >= 3


def test_paths_resolution():
    """Verify all paths are absolute and point to the repo."""
    for p in [
        config.Paths.raw,
        config.Paths.staging,
        config.Paths.curated,
        config.Paths.outputs,
        config.Paths.metadata,
    ]:
        assert isinstance(p, Path)
        assert p.is_absolute()


# ---------------------------------------------------------------------------
# 3. Transformation & Data Quality Tests
# ---------------------------------------------------------------------------


def test_staging_layer_schema():
    """Verify staging layer contains all required schema columns."""
    staging_file = config.Paths.staging / "staging.parquet"
    assert staging_file.exists(), "staging.parquet not found. Run transform first."
    df = pd.read_parquet(staging_file)

    res = check_schema(df)
    assert res["status"] == "PASS", f"Schema check failed: {res['details']}"


def test_staging_layer_uniqueness():
    """Verify staging layer contains zero duplicate rows."""
    staging_file = config.Paths.staging / "staging.parquet"
    df = pd.read_parquet(staging_file)
    res = check_uniqueness(df)
    assert res["status"] == "PASS", f"Uniqueness check failed: {res['details']}"


def test_staging_layer_row_count():
    """Verify staging layer meets minimum row count threshold."""
    staging_file = config.Paths.staging / "staging.parquet"
    df = pd.read_parquet(staging_file)
    res = check_row_count(df)
    assert res["status"] == "PASS", f"Row count check failed: {res['details']}"
    assert len(df) == 1367


# ---------------------------------------------------------------------------
# 4. Curated & Split Manifest Tests
# ---------------------------------------------------------------------------


def test_curated_splits_exist():
    """Verify train and test curated Parquet files exist."""
    train_file = config.Paths.curated / "train.parquet"
    test_file = config.Paths.curated / "test.parquet"
    assert train_file.exists(), "train.parquet missing"
    assert test_file.exists(), "test.parquet missing"

    df_train = pd.read_parquet(train_file)
    df_test = pd.read_parquet(test_file)
    assert len(df_train) == 1093
    assert len(df_test) == 274
    assert len(df_train) + len(df_test) == 1367


def test_split_manifest_contents():
    """Verify split_manifest.json records deterministic metadata."""
    manifest_path = config.Paths.metadata / "split_manifest.json"
    assert manifest_path.exists(), "split_manifest.json missing"
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)

    assert manifest["random_seed"] == 42
    assert manifest["train_rows"] == 1093
    assert manifest["test_rows"] == 274
    assert manifest["stratified"] is True


# ---------------------------------------------------------------------------
# 5. Model Evaluation & Reproducibility Tests
# ---------------------------------------------------------------------------


def test_model_artifacts_exist():
    """Verify serialized model files exist for all configured classifiers."""
    expected_models = [f"{c['name']}.pkl" for c in config.Model.classifiers]
    for model_name in expected_models:
        model_path = config.Paths.model_dir / model_name
        assert model_path.exists(), f"Model file {model_name} missing"


def test_reproducibility_within_tolerance():
    """
    Verify actual evaluation metrics are checked against metadata/expected_metrics.json.
    Logistic Regression (primary baseline) must pass within documented tolerance,
    and all checks must produce numeric diffs.
    """
    metrics_path = config.Paths.outputs / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing"
    with open(metrics_path, "r", encoding="utf-8") as fh:
        actual = json.load(fh)

    results = check_reproducibility(actual["classifiers"])
    assert len(results) >= 2, "Expected at least 2 classifier reproducibility checks."
    
    # Primary baseline model must be within tolerance
    lr_result = next((r for r in results if "LogisticRegression" in r["check"]), None)
    assert lr_result is not None, "LogisticRegression check missing"
    assert lr_result["status"] == "PASS", f"LogisticRegression failed: {lr_result['details']}"

    # Every check must compute valid differences
    for r in results:
        assert "value" in r and r["value"] is not None
        assert "diff" in r["value"]


if __name__ == "__main__":
    tests = [
        test_source_dataset_exists,
        test_source_checksum_integrity,
        test_config_loading,
        test_paths_resolution,
        test_staging_layer_schema,
        test_staging_layer_uniqueness,
        test_staging_layer_row_count,
        test_curated_splits_exist,
        test_split_manifest_contents,
        test_model_artifacts_exist,
        test_reproducibility_within_tolerance,
    ]
    passed = 0
    print(f"Running {len(tests)} automated tests...")
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\nResult: {passed}/{len(tests)} tests passed.")
    if passed < len(tests):
        exit(1)
