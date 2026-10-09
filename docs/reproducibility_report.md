# Reproducibility & Tolerance Verification Report

**Author:** David Hyzxent L. Memorando  
**Student Number:** 2024162312  
**Section:** DSS150P_CM17_1Q2627  
**Course:** DSS150P — Fundamentals of Data Engineering  
**Activity:** Integrated Individual Laboratory Activity #4  
**Project Title:** *A Comparative Analysis of Machine Learning Classifiers for Distinguishing Human-Written and AI-Generated Text Using Stylometric and Lexical Features*  
**Date:** October 2026  

---

## 1. Reproducibility Contract & Execution Guarantee

The pipeline was engineered to strictly satisfy the **DSS150P Reproducibility Contract**:
1. **Zero Hardcoded Paths:** Every directory and filepath is resolved relative to `PROJECT_ROOT` in `src/config.py`.
2. **Deterministic Preprocessing:** Standard scaling is fit exclusively on the training split, and categorical variables are mapped deterministically.
3. **Deterministic Splitting:** Seed `42` with stratification across target `label` is locked in `metadata/split_manifest.json`.
4. **Frozen Random Seeds:** All 5 classifiers enforce `random_state=42` during both cross-validation and final model fitting.
5. **Rerun Idempotency:** Subsequent executions do not duplicate rows in staging or curated datasets due to row-level MD5 hashing (`_row_hash`) and checkpoint watermarks (`data/raw/.checkpoint.json`).

---

## 2. Baseline Reproduction vs. Actual Run Evidence

The pipeline was executed against the primary dataset (`ai_human_content_detection_dataset.csv`, SHA-256: `b63cbf71bf8921071fb4513a0d0616494040463e1d41c7c04ac46ccbb530be1f`).

Below is the comparison between the baseline expectations recorded in `metadata/expected_metrics.json` and the empirical results computed in `outputs/metrics.json` under an automated tolerance threshold of **$\pm 0.02$ ($\pm 2.0\%$)**:

| Classifier | Baseline F1 (Weighted) | Actual Run F1 (Weighted) | Absolute Difference ($\Delta$) | Tolerance Limit | Verification Status | Accuracy | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.5584** | **0.5584** | **0.0000** | $\pm 0.0200$ | **PASS** | 0.5584 | 0.5521 |
| **Support Vector Classifier (SVC)** | **0.5547** | **0.5547** | **0.0000** | $\pm 0.0200$ | **PASS** | 0.5547 | 0.4383 |
| **Random Forest** | **0.4800** | **0.4800** | **0.0000** | $\pm 0.0200$ | **PASS** | 0.4818 | 0.4982 |
| **Gradient Boosting** | **0.4731** | **0.4731** | **0.0000** | $\pm 0.0200$ | **PASS** | 0.4745 | 0.4809 |
| **Decision Tree** | **0.4653** | **0.4653** | **0.0000** | $\pm 0.0200$ | **PASS** | 0.4818 | 0.5100 |

*Result:* **100% of classifiers passed the reproducibility gate**, reproducing metrics with near zero delta ($\Delta < 0.0001$).

---

## 3. Rerun Safety & Idempotency Audit

To prove rerun safety, the ingestion and transformation processes were re-executed sequentially:

1. **Initial Full Ingestion:**
   - 1,367 rows ingested into `data/raw/full__*.parquet`.
   - Audit columns generated: `_pipeline_run_id`, `_ingested_at_utc`, `_source_file`, `_source_checksum`, `_row_hash`.
2. **Incremental Batch Ingestion:**
   - Source split into 3 deterministic chunks (455, 455, 457 rows).
   - Checkpoint `.checkpoint.json` recorded batch IDs `batch_01_of_03`, `batch_02_of_03`, and `batch_03_of_03`.
3. **Repeated Incremental Ingestion without `--force`:**
   - Output log: `No new batches to ingest. All 3 batches already committed.`
   - Verification: Zero duplicate chunks were appended.
4. **Staging Transformation Deduplication:**
   - Deduplication assertion verified `0 duplicate rows found (keyed by _row_hash)`.
   - Staging table row count remained stable at exactly **1,367 rows**.

---

## 4. Engineering Improvements Over Historical Group Project

| Phase | Previous Group Project State | Production Data Engineering Treatment (DSS150P) |
| :--- | :--- | :--- |
| **Source Provenance** | Ad-hoc CSV download on local machine | Cryptographic SHA-256 fingerprint verified in automated CLI check |
| **Execution State** | Jupyter Notebook executed cell-by-cell | Modular Python package (`src/`) with CLI and Airflow DAG |
| **Data Cleaning** | Interactive Pandas cells with hidden state | Idempotent staging transformation with median imputation and quarantine |
| **Splitting Logic** | Uncontrolled random seed | Fixed seed `42` with stratified indices persisted in `split_manifest.json` |
| **Storage Architecture**| Single flat CSV file | Tiered Parquet storage, Hive-style partitioning, and PostgreSQL relational views |
| **Validation** | Informal manual inspection of head() rows | 29 automated data quality assertions in `outputs/quality_report.json` |
| **Verification Gate** | Subjective accuracy reporting | Machine-readable comparison against `expected_metrics.json` |

---

## 5. Conclusion

The predictive analytics project has been successfully productionized into an enterprise-grade data engineering system. Any engineer can clone the repository, run `make run-all`, and reproduce the exact model metrics and curated outputs with zero manual source code modification.
