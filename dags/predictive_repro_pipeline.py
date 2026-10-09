"""
dags/predictive_repro_pipeline.py
==================================
Apache Airflow DAG orchestrating the end-to-end reproducible pipeline.

Key Design Principles:
  - Keeps all business logic outside the DAG file (delegates to src.cli)
  - Configures retry logic, failure behavior, execution timeouts, and audit logging
  - Explicit task dependencies:
      verify_source -> ingest_data -> transform_data -> validate_quality
                    -> storage_materialize -> train_models -> evaluate_reproduce
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

# ---------------------------------------------------------------------------
# Default arguments & retry policies
# ---------------------------------------------------------------------------
default_args = {
    "owner": "david_memorando",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=15),
}

# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
with DAG(
    dag_id="predictive_repro_pipeline",
    default_args=default_args,
    description="DSS150P Lab 4 - Reproducible AI vs Human Text Detection Data Pipeline",
    schedule_interval=None,  # Triggered manually or by upstream event
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["dss150p", "data-engineering", "reproducibility", "ai-detection"],
) as dag:

    # 1. Source verification (Cryptographic SHA-256 fingerprint test)
    verify_source_task = BashOperator(
        task_id="verify_source_checksum",
        bash_command="python -m src.cli check-source",
    )

    # 2. Ingestion (Full + Incremental simulation)
    ingest_task = BashOperator(
        task_id="ingest_full_and_incremental",
        bash_command="python -m src.cli ingest --mode full && python -m src.cli ingest --mode incremental",
    )

    # 3. Transformation (Raw -> Staging -> Curated)
    transform_task = BashOperator(
        task_id="transform_layers",
        bash_command="python -m src.cli transform",
    )

    # 4. Data Quality & Contract Validation (7-layer quality checks)
    validate_task = BashOperator(
        task_id="validate_data_quality",
        bash_command="python -m src.cli validate",
    )

    # 5. Multi-format & Partitioned Storage Materialization
    storage_task = BashOperator(
        task_id="materialize_storage",
        bash_command="python -m src.cli storage --skip-postgres",
    )

    # 6. Model Training (Deterministic Multi-Classifier Training)
    train_task = BashOperator(
        task_id="train_classifiers",
        bash_command="python -m src.cli train",
    )

    # 7. Model Evaluation & Reproducibility Check
    evaluate_task = BashOperator(
        task_id="evaluate_and_check_reproducibility",
        bash_command="python -m src.cli evaluate",
    )

    # -----------------------------------------------------------------------
    # Pipeline Dependencies (Linear, deterministic workflow)
    # -----------------------------------------------------------------------
    (
        verify_source_task
        >> ingest_task
        >> transform_task
        >> validate_task
        >> storage_task
        >> train_task
        >> evaluate_task
    )
