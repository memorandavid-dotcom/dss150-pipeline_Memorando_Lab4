"""
src/validate.py
===============
Data quality, schema, and reproducibility checks.

Check categories:
  1. Schema validation   – required columns, declared types
  2. Completeness        – null rate thresholds per column
  3. Uniqueness          – row hash deduplication verification
  4. Validity            – range/domain rules for numeric features and target
  5. Freshness           – ingestion timestamp recency
  6. Business rules      – target distribution, class balance
  7. Reproducibility     – compare run metrics against baseline expected_metrics.json

Each check returns a dict with: check_name, status (PASS/WARN/FAIL), details.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

CheckResult = dict[str, Any]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result(name: str, status: str, details: str, value: Any = None) -> CheckResult:
    return {"check": name, "status": status, "details": details, "value": value}


def _log_result(r: CheckResult) -> None:
    level = {"PASS": logging.INFO, "WARN": logging.WARNING, "FAIL": logging.ERROR}.get(
        r["status"], logging.INFO
    )
    logger.log(level, "[%s] %s — %s", r["status"], r["check"], r["details"])


# ---------------------------------------------------------------------------
# 1. Schema Validation
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = (
    config.Model.numeric_features
    + config.Model.categorical_features
    + [config.Model.target_column]
)


def check_schema(df: pd.DataFrame) -> CheckResult:
    """Verify all required columns are present."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return _result("schema", "FAIL", f"Missing columns: {missing}", missing)
    return _result("schema", "PASS", f"All {len(REQUIRED_COLUMNS)} required columns present.")


# ---------------------------------------------------------------------------
# 2. Completeness
# ---------------------------------------------------------------------------


def check_completeness(df: pd.DataFrame) -> list[CheckResult]:
    """Check null rate per column against max_null_rate threshold."""
    threshold = config.get("quality.max_null_rate", 0.10)
    results: list[CheckResult] = []
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        null_rate = df[col].isna().mean()
        status = "PASS" if null_rate <= threshold else "WARN"
        results.append(
            _result(
                f"completeness.{col}",
                status,
                f"null_rate={null_rate:.2%} (threshold={threshold:.0%})",
                null_rate,
            )
        )
    return results


# ---------------------------------------------------------------------------
# 3. Uniqueness / Deduplication
# ---------------------------------------------------------------------------


def check_uniqueness(df: pd.DataFrame) -> CheckResult:
    """Verify no duplicate rows (by _row_hash if available, else full row)."""
    if "_row_hash" in df.columns:
        n_dups = df["_row_hash"].duplicated().sum()
        col_used = "_row_hash"
    else:
        n_dups = df.duplicated().sum()
        col_used = "full_row"

    status = "PASS" if n_dups == 0 else "FAIL"
    return _result(
        "uniqueness",
        status,
        f"{n_dups} duplicate rows found (keyed by {col_used})",
        n_dups,
    )


# ---------------------------------------------------------------------------
# 4. Validity (range / domain rules)
# ---------------------------------------------------------------------------

_RANGE_RULES: dict[str, tuple[float, float]] = {
    "lexical_diversity": (0.0, 1.0),
    "punctuation_ratio": (0.0, 1.0),
    "passive_voice_ratio": (0.0, 1.0),
    "sentiment_score": (-1.0, 1.0),
    "word_count": (1, 100_000),
    "character_count": (1, 1_000_000),
    "sentence_count": (0, 10_000),
}


def check_validity(df: pd.DataFrame) -> list[CheckResult]:
    """Check numeric columns against declared valid ranges."""
    results: list[CheckResult] = []
    for col, (lo, hi) in _RANGE_RULES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        n_violations = ((series < lo) | (series > hi)).sum()
        status = "PASS" if n_violations == 0 else "WARN"
        results.append(
            _result(
                f"validity.{col}",
                status,
                f"{n_violations} values outside [{lo}, {hi}]",
                n_violations,
            )
        )

    # Target column domain check
    target_col = config.Model.target_column
    allowed_classes = config.get("quality.target_classes", [0, 1])
    if target_col in df.columns:
        bad = ~df[target_col].isin(allowed_classes)
        n_bad = bad.sum()
        status = "PASS" if n_bad == 0 else "FAIL"
        results.append(
            _result(
                f"validity.{target_col}",
                status,
                f"{n_bad} values not in {allowed_classes}",
                n_bad,
            )
        )

    return results


# ---------------------------------------------------------------------------
# 5. Freshness
# ---------------------------------------------------------------------------


def check_freshness(df: pd.DataFrame, max_hours: float = 24.0) -> CheckResult:
    """Check that ingestion timestamp is within max_hours of now."""
    if "_ingested_at_utc" not in df.columns:
        return _result("freshness", "WARN", "No _ingested_at_utc column found.")

    try:
        latest_str = df["_ingested_at_utc"].dropna().max()
        latest = datetime.fromisoformat(latest_str)
        now = datetime.now(timezone.utc)
        if latest.tzinfo is None:
            latest = latest.replace(tzinfo=timezone.utc)
        age_hours = (now - latest).total_seconds() / 3600
        status = "PASS" if age_hours <= max_hours else "WARN"
        return _result(
            "freshness",
            status,
            f"Latest ingestion: {latest_str} ({age_hours:.1f}h ago)",
            age_hours,
        )
    except Exception as exc:
        return _result("freshness", "WARN", f"Could not parse timestamp: {exc}")


# ---------------------------------------------------------------------------
# 6. Business Rules (target distribution / class balance)
# ---------------------------------------------------------------------------


def check_class_balance(df: pd.DataFrame) -> CheckResult:
    """Check for severe class imbalance (ratio < 0.25 or > 4.0 is WARN)."""
    target_col = config.Model.target_column
    if target_col not in df.columns:
        return _result("class_balance", "WARN", "Target column not found.")

    counts = df[target_col].value_counts()
    if len(counts) < 2:
        return _result("class_balance", "WARN", f"Only one class found: {counts.to_dict()}")

    ratio = counts.min() / counts.max()
    status = "PASS" if ratio >= 0.25 else "WARN"
    return _result(
        "class_balance",
        status,
        f"Class distribution: {counts.to_dict()}  min/max ratio={ratio:.3f}",
        ratio,
    )


def check_row_count(df: pd.DataFrame) -> CheckResult:
    """Check that row count meets minimum threshold after staging."""
    minimum = config.get("quality.min_rows_after_staging", 1200)
    n = len(df)
    status = "PASS" if n >= minimum else "FAIL"
    return _result(
        "row_count",
        status,
        f"{n} rows (minimum required: {minimum})",
        n,
    )


# ---------------------------------------------------------------------------
# 7. Reproducibility Check
# ---------------------------------------------------------------------------


def check_reproducibility(
    actual_metrics: dict[str, Any],
    baseline_path: Optional[Path] = None,
) -> list[CheckResult]:
    """
    Compare actual run metrics against baseline expected_metrics.json.
    Returns PASS if within declared tolerance, FAIL otherwise.
    """
    baseline_path = baseline_path or (config.Paths.metadata / "expected_metrics.json")
    if not baseline_path.exists():
        return [_result("reproducibility", "WARN", "No baseline metrics file found.")]

    with open(baseline_path, "r", encoding="utf-8") as fh:
        baseline = json.load(fh)

    tolerance = config.get("reproducibility.tolerance", 0.02)
    metric_key = config.get("reproducibility.metric", "f1_weighted")
    results: list[CheckResult] = []

    for classifier_name, expected in baseline.get("classifiers", {}).items():
        actual = actual_metrics.get(classifier_name, {})
        if not actual:
            results.append(
                _result(
                    f"reproducibility.{classifier_name}",
                    "WARN",
                    "No actual metrics found for this classifier.",
                )
            )
            continue

        exp_val = expected.get(metric_key)
        act_val = actual.get(metric_key)
        if exp_val is None or act_val is None:
            results.append(
                _result(
                    f"reproducibility.{classifier_name}",
                    "WARN",
                    f"Missing '{metric_key}' in baseline or actual.",
                )
            )
            continue

        diff = abs(act_val - exp_val)
        status = "PASS" if diff <= tolerance else "FAIL"
        results.append(
            _result(
                f"reproducibility.{classifier_name}",
                status,
                f"{metric_key}: expected={exp_val:.4f}, actual={act_val:.4f}, diff={diff:.4f} (tol={tolerance})",
                {"expected": exp_val, "actual": act_val, "diff": diff},
            )
        )

    return results


# ---------------------------------------------------------------------------
# Master validation runner
# ---------------------------------------------------------------------------


def run_all_checks(
    df: pd.DataFrame,
    actual_metrics: Optional[dict] = None,
    write_report: bool = True,
) -> list[CheckResult]:
    """
    Run all quality checks and optionally write a JSON report.
    Returns a flat list of CheckResult dicts.
    """
    all_results: list[CheckResult] = []

    all_results.append(check_schema(df))
    all_results.extend(check_completeness(df))
    all_results.append(check_uniqueness(df))
    all_results.extend(check_validity(df))
    all_results.append(check_freshness(df))
    all_results.append(check_class_balance(df))
    all_results.append(check_row_count(df))

    if actual_metrics:
        all_results.extend(check_reproducibility(actual_metrics))

    for r in all_results:
        _log_result(r)

    # Summary
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
    for r in all_results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    logger.info("Quality summary: %s", counts)

    if write_report:
        from datetime import datetime, timezone
        report = {
            "run_at": datetime.now(timezone.utc).isoformat(),
            "summary": counts,
            "checks": all_results,
        }
        report_path = config.Paths.outputs / "quality_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("Quality report written → %s", report_path)

    return all_results
