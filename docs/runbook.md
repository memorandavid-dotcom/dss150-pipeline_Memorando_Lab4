# Operational Runbook: AI vs. Human Text Detection Pipeline

**Author:** David Hyzxent L. Memorando (2024162312)  
**Section:** DSS150P_CM17_1Q2627  
**System:** DSS150P Production Data Engineering Pipeline  
**Version:** 1.0.0  

---

## 1. System Overview & Architecture

The pipeline processes immutable text classification data across five storage tiers:
1. **Source Tier:** Read-only immutable source file (`ai_human_content_detection_dataset.csv`) verified via SHA-256.
2. **Raw Tier:** Timestamped snapshots (`data/raw/full__*.parquet`) and batch chunks (`data/raw/incremental__*.parquet`) with audit metadata.
3. **Staging Tier:** Deduplicated, type-cast, and median-imputed dataset (`data/staging/staging.parquet`).
4. **Curated Tier:** Standardized train/test splits (`data/curated/train.parquet`, `data/curated/test.parquet`) and split manifest (`metadata/split_manifest.json`).
5. **Serving / Storage Tier:** Multi-format exports (Parquet, CSV, JSONL), partitioned Parquet by `content_type`, and PostgreSQL tables.

---

## 2. Standard Operating Procedures (SOP)

### SOP-01: Clean-Clone Pipeline Execution
**Trigger:** Deploying on a fresh machine or reviewer environment.
**Procedure:**
```bash
# 1. Clone repository
git clone https://github.com/memorandavid-dotcom/dss150-pipeline_Memorando_Lab4.git
cd dss150-pipeline_Memorando_Lab4

# 2. Setup environment variables
cp .env.example .env

# 3. Create virtual environment and install dependencies
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt

# 4. Verify source checksum
python -m src.cli check-source

# 5. Execute full pipeline
python -m src.cli run-all --skip-postgres

# 6. Run automated test suite
python tests/test_pipeline.py
```

### SOP-02: Executing via Docker & Apache Airflow
**Trigger:** Scheduled or containerized production deployment.
**Procedure:**
```bash
# Start all containers in background
docker compose up -d

# Check service health
docker compose ps

# Access Airflow UI
# URL: http://localhost:8080
# Username: admin
# Password: admin
# Trigger DAG: predictive_repro_pipeline
```

---

## 3. Incident Management & Recovery Procedures

### Incident 01: Source Dataset Checksum Failure
- **Symptom:** `python -m src.cli check-source` returns `[FAIL] Checksum MISMATCH`.
- **Root Cause:** Source file in `data/source/` was modified, corrupted, or replaced with an incorrect version.
- **Remediation:**
  1. Inspect `metadata/source_manifest.yml` for expected SHA-256 (`b63cbf71bf8921071fb4513a0d0616494040463e1d41c7c04ac46ccbb530be1f`).
  2. Restore the original `ai_human_content_detection_dataset.csv` from source backup.
  3. Re-run `python -m src.cli check-source`.

### Incident 02: Corrupted or Stale Incremental Watermark
- **Symptom:** New batches are skipped or duplicate records are suspected in incremental runs.
- **Root Cause:** Stale watermark state in `data/raw/.checkpoint.json`.
- **Remediation:**
  1. Run ingestion with `--force` flag:
     ```bash
     python -m src.cli ingest --mode incremental --force
     ```
  2. To completely reset watermarks, remove `data/raw/.checkpoint.json` and re-ingest.

### Incident 03: Data Quality or Contract Validation Failure
- **Symptom:** `python -m src.cli validate` returns exit code 1 with one or more `[FAIL]` checks.
- **Root Cause:** Staging data violated schema rules, row count limits (< 1,200), or unexpected target classes.
- **Remediation:**
  1. Inspect `outputs/quality_report.json` to identify the failing column or check.
  2. Check `data/quarantine/` for records excluded due to missing labels.
  3. Re-run `python -m src.cli transform --force` to regenerate staging tables.

### Incident 04: Model Drift or Reproducibility Failure
- **Symptom:** `reproducibility.<classifier>` fails tolerance gate in `outputs/reproducibility_check.json`.
- **Root Cause:** Random seed altered in `.env` or dependencies changed.
- **Remediation:**
  1. Check `.env` and verify `RANDOM_SEED=42`.
  2. Inspect `metadata/split_manifest.json` to verify train indices match baseline.
  3. Re-run training and evaluation:
     ```bash
     python -m src.cli train --force
     python -m src.cli evaluate
     ```

---

## 4. Maintenance & Housekeeping

To clean generated caches and temporary artifacts:
```bash
make clean
```
To purge and recreate Docker containers and PostgreSQL volumes:
```bash
docker compose down -v
docker compose up -d
```
