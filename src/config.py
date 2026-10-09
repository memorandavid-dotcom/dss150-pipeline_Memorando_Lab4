"""
src/config.py
=============
Central configuration loader. Merges config/settings.yml with
environment variables from .env. All other modules import from here.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Resolve project root regardless of where the script is called from
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

# Load .env if it exists (silently skip in Docker where env is injected)
load_dotenv(ENV_FILE, override=False)


# ---------------------------------------------------------------------------
# Load settings.yml
# ---------------------------------------------------------------------------
_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


_RAW: dict[str, Any] = _load_yaml(_SETTINGS_PATH)


# ---------------------------------------------------------------------------
# Public accessors - all paths returned as absolute Path objects
# ---------------------------------------------------------------------------


def get(key_path: str, default: Any = None) -> Any:
    """Dot-separated key access into the settings dict.

    Example:
        get('model.random_seed')  -> 42
    """
    parts = key_path.split(".")
    node: Any = _RAW
    for part in parts:
        if not isinstance(node, dict):
            return default
        node = node.get(part, None)
        if node is None:
            return default
    return node


def path(key_path: str) -> Path:
    """Return a config path value as an absolute Path."""
    rel = get(key_path)
    if rel is None:
        raise KeyError(f"Config key '{key_path}' not found.")
    return PROJECT_ROOT / rel


# --- Convenience shortcuts ------------------------------------------------


class Paths:
    root = PROJECT_ROOT
    raw = PROJECT_ROOT / get("paths.raw_dir", "data/raw")
    staging = PROJECT_ROOT / get("paths.staging_dir", "data/staging")
    curated = PROJECT_ROOT / get("paths.curated_dir", "data/curated")
    partitioned = PROJECT_ROOT / get("paths.partitioned_dir", "data/partitioned")
    quarantine = PROJECT_ROOT / get("paths.quarantine_dir", "data/quarantine")
    outputs = PROJECT_ROOT / get("paths.outputs_dir", "outputs")
    model_dir = PROJECT_ROOT / get("paths.model_dir", "outputs/model")
    logs = PROJECT_ROOT / get("paths.logs_dir", "logs")
    metadata = PROJECT_ROOT / get("paths.metadata_dir", "metadata")
    source_primary = PROJECT_ROOT / get(
        "source.primary_file",
        "data/source/O_Files/ProjectData2(2025v)/ai_human_content_detection_dataset.csv",
    )
    checkpoint = PROJECT_ROOT / get("source.checkpoint_file", "data/raw/.checkpoint.json")


class DB:
    host: str = os.getenv("POSTGRES_HOST", "localhost")
    port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    db: str = os.getenv("POSTGRES_DB", "dss150_pipeline")
    user: str = os.getenv("POSTGRES_USER", "dss150_user")
    password: str = os.getenv("POSTGRES_PASSWORD", "changeme_local_only")

    @classmethod
    def url(cls) -> str:
        return (
            f"postgresql+psycopg2://{cls.user}:{cls.password}"
            f"@{cls.host}:{cls.port}/{cls.db}"
        )


class Model:
    random_seed: int = int(os.getenv("RANDOM_SEED", str(get("model.random_seed", 42))))
    test_size: float = float(os.getenv("TEST_SIZE", str(get("model.test_size", 0.2))))
    cv_folds: int = get("model.cv_folds", 5)
    classifiers: list = get("model.classifiers", [])
    target_column: str = get("schema.target_column", "label")
    numeric_features: list = get("schema.numeric_features", [])
    categorical_features: list = get("schema.categorical_features", [])


# ---------------------------------------------------------------------------
# Ensure output directories exist (idempotent)
# ---------------------------------------------------------------------------


def ensure_dirs() -> None:
    """Create all required pipeline directories if they don't exist."""
    for attr in vars(Paths):
        if attr.startswith("_"):
            continue
        val = getattr(Paths, attr)
        if isinstance(val, Path) and val.suffix == "":
            val.mkdir(parents=True, exist_ok=True)
