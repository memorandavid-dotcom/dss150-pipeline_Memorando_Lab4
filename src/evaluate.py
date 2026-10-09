"""
src/evaluate.py
===============
Model evaluation module.

Computes and persists:
  - Classification report per classifier (precision, recall, F1)
  - Confusion matrix
  - ROC-AUC
  - Comparison table across all classifiers
  - Predictions CSV (outputs/predictions.csv)
  - Consolidated metrics JSON (outputs/metrics.json)

Also runs the reproducibility check against metadata/expected_metrics.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from src import config
from src.validate import check_reproducibility

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_test() -> tuple[pd.DataFrame, pd.Series]:
    """Load curated test split, return (X_test, y_test)."""
    path = config.Paths.curated / "test.parquet"
    if not path.exists():
        raise FileNotFoundError("Test split not found. Run transform first.")
    df = pd.read_parquet(path, engine="pyarrow")
    target = config.Model.target_column
    feature_cols = [
        c
        for c in df.columns
        if c not in (target, "_split", "_original_index", "_curated_at_utc")
        and not c.startswith("_")
    ]
    X = df[feature_cols].fillna(0)
    y = df[target].astype(int)
    return X, y


def _evaluate_one(clf, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    """Evaluate a single fitted classifier."""
    y_pred = clf.predict(X_test)
    y_proba = None
    if hasattr(clf, "predict_proba"):
        y_proba = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_proba = clf.decision_function(X_test)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_binary": float(f1_score(y_test, y_pred, average="binary")),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if y_proba is not None else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "predictions": y_pred.tolist(),
    }
    return metrics


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_all(force: bool = False) -> dict[str, Any]:
    """
    Evaluate all trained models on the test split.

    Returns a dict: {classifier_name: metrics_dict}
    Writes:
      - outputs/metrics.json          (all metrics, machine-readable)
      - outputs/predictions.csv       (per-row predictions from best model)
      - outputs/quality_report.json   (via validate.check_reproducibility)
    """
    config.ensure_dirs()
    X_test, y_test = _load_test()
    logger.info("Test data: %d rows x %d features", len(X_test), X_test.shape[1])

    all_metrics: dict[str, dict] = {}
    best_clf_name: Optional[str] = None
    best_f1: float = -1.0
    best_clf = None

    model_dir = config.Paths.model_dir
    model_files = list(model_dir.glob("*.pkl"))
    if not model_files:
        raise FileNotFoundError("No trained models found. Run train_all() first.")

    for model_path in sorted(model_files):
        name = model_path.stem
        logger.info("Evaluating %s ...", name)
        clf = joblib.load(model_path)

        metrics = _evaluate_one(clf, X_test, y_test)
        all_metrics[name] = {k: v for k, v in metrics.items() if k != "predictions"}

        logger.info(
            "%s -> acc=%.4f  f1_weighted=%.4f  roc_auc=%s",
            name,
            metrics["accuracy"],
            metrics["f1_weighted"],
            f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "N/A",
        )

        if metrics["f1_weighted"] > best_f1:
            best_f1 = metrics["f1_weighted"]
            best_clf_name = name
            best_predictions = metrics["predictions"]
            best_clf = clf

    # --- Persist metrics.json ---
    metrics_out = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": config.Model.random_seed,
        "test_size": config.Model.test_size,
        "best_classifier": best_clf_name,
        "best_f1_weighted": best_f1,
        "classifiers": all_metrics,
    }
    metrics_path = config.Paths.outputs / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics_out, fh, indent=2, default=str)
    logger.info("Metrics saved -> %s", metrics_path.name)

    # --- Persist predictions.csv (from best classifier) ---
    if best_clf_name:
        test_df = pd.read_parquet(config.Paths.curated / "test.parquet", engine="pyarrow")
        test_df["_predicted_label"] = best_predictions
        test_df["_classifier"] = best_clf_name
        preds_path = config.Paths.outputs / "predictions.csv"
        test_df.to_csv(preds_path, index=False)
        logger.info(
            "Predictions (from %s) saved -> %s", best_clf_name, preds_path.name
        )

    # --- Reproducibility check ---
    repro_results = check_reproducibility(all_metrics)
    repro_path = config.Paths.outputs / "reproducibility_check.json"
    with open(repro_path, "w", encoding="utf-8") as fh:
        json.dump(repro_results, fh, indent=2, default=str)
    logger.info("Reproducibility check saved -> %s", repro_path.name)

    # --- Comparison table ---
    comparison_rows = []
    for clf_name, m in all_metrics.items():
        comparison_rows.append(
            {
                "classifier": clf_name,
                "accuracy": round(m["accuracy"], 4),
                "f1_weighted": round(m["f1_weighted"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "roc_auc": round(m["roc_auc"], 4) if m.get("roc_auc") else None,
            }
        )
    comp_df = pd.DataFrame(comparison_rows).sort_values("f1_weighted", ascending=False)
    comp_path = config.Paths.outputs / "classifier_comparison.csv"
    comp_df.to_csv(comp_path, index=False)
    logger.info("Comparison table saved -> %s", comp_path.name)

    try:
        logger.info("\n%s", comp_df.to_string(index=False))
    except Exception:
        pass

    return all_metrics
