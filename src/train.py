"""
src/train.py
============
Model training module.

Trains all classifiers declared in config/settings.yml against the curated
training split. Models are serialized with joblib. Supports cross-validation.

Design principles:
  - Deterministic: all classifiers use fixed random_state from config
  - Modular: each classifier is instantiated from its config dict
  - Rerun-safe: skips training if model artifacts already exist (unless force)
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from src import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Classifier factory
# ---------------------------------------------------------------------------


def _build_classifier(name: str, params: dict[str, Any]):
    """Instantiate a classifier by name with given params."""
    # Lazy imports to avoid loading all libraries upfront
    if name == "LogisticRegression":
        return LogisticRegression(**params)
    elif name == "RandomForest":
        return RandomForestClassifier(**params)
    elif name == "SVC":
        return SVC(**params)
    elif name == "GradientBoosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(**params)
    elif name == "DecisionTree":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params)
    elif name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            params = {k: v for k, v in params.items() if k != "use_label_encoder"}
            return XGBClassifier(**params)
        except ImportError:
            logger.warning("XGBoost not installed; falling back to GradientBoostingClassifier")
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(random_state=params.get("random_state", 42))
    elif name == "LightGBM":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(**params)
        except ImportError:
            logger.warning("LightGBM not installed; falling back to RandomForestClassifier")
            return RandomForestClassifier(random_state=params.get("random_state", 42))
    else:
        raise ValueError(f"Unknown classifier: {name}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _load_split(split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a curated split (train/test) and return (X, y)."""
    path = config.Paths.curated / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Curated {split} split not found: {path}")
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


def train_all(
    force: bool = False,
) -> dict[str, Path]:
    """
    Train all classifiers defined in config/settings.yml.

    For each classifier:
      1. Run StratifiedKFold CV on training set
      2. Fit final model on full training set
      3. Serialize to outputs/model/<name>.pkl

    Returns dict mapping classifier name -> model path.
    """
    config.ensure_dirs()
    config.Paths.model_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train = _load_split("train")
    logger.info("Training data: %d rows x %d features", len(X_train), X_train.shape[1])

    classifier_configs = config.Model.classifiers
    model_paths: dict[str, Path] = {}
    cv_results_all: dict[str, dict] = {}

    cv = StratifiedKFold(
        n_splits=config.Model.cv_folds,
        shuffle=True,
        random_state=config.Model.random_seed,
    )

    for clf_cfg in classifier_configs:
        name = clf_cfg["name"]
        params = clf_cfg.get("params", {})
        model_path = config.Paths.model_dir / f"{name}.pkl"

        if model_path.exists() and not force:
            logger.info("Model already exists for %s. Skipping (use force=True).", name)
            model_paths[name] = model_path
            continue

        logger.info("Training %s ...", name)
        clf = _build_classifier(name, params)

        # Cross-validation
        cv_res = cross_validate(
            clf,
            X_train,
            y_train,
            cv=cv,
            scoring=["accuracy", "f1_weighted", "roc_auc"],
            return_train_score=False,
            n_jobs=1,  # Deterministic
        )
        cv_summary = {
            k.replace("test_", ""): {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
            }
            for k, v in cv_res.items()
            if k.startswith("test_")
        }
        cv_results_all[name] = cv_summary
        logger.info(
            "%s CV: acc=%.4f+/-%.4f  f1=%.4f+/-%.4f",
            name,
            cv_summary["accuracy"]["mean"],
            cv_summary["accuracy"]["std"],
            cv_summary["f1_weighted"]["mean"],
            cv_summary["f1_weighted"]["std"],
        )

        # Fit final model on full training set
        clf.fit(X_train, y_train)
        joblib.dump(clf, model_path)
        logger.info("Model saved -> %s", model_path.name)
        model_paths[name] = model_path

    # Persist CV results
    cv_path = config.Paths.outputs / "cv_results.json"
    with open(cv_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"trained_at": datetime.now(timezone.utc).isoformat(), "cv": cv_results_all},
            fh,
            indent=2,
        )
    logger.info("CV results saved -> %s", cv_path.name)

    return model_paths
