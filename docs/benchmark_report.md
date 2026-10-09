# Storage Systems & Format Benchmark Report

**Author:** David Hyzxent L. Memorando (2024162312)  
**Section:** DSS150P_CM17_1Q2627  
**Dataset:** AI vs Human Content Detection Dataset (1,367 rows, 17 columns)  
**Benchmark Date:** October 2026  

---

## 1. Executive Summary

As part of the Week 6 data engineering curriculum on *Storage Systems and Data Organization*, this report analyzes storage footprints, I/O read speeds, schema enforcement, and workload suitability across four materialized formats:
1. **Apache Parquet (Columnar, Snappy-compressed)**
2. **Comma-Separated Values (CSV, Row-oriented text)**
3. **JSON Lines (JSONL, Semi-structured document text)**
4. **Partitioned Apache Parquet (Hive-style partitioned on `content_type`)**

---

## 2. Empirical Benchmark Results

| Storage Format | File Size (Bytes) | Storage Footprint | Compression Ratio vs JSONL | Schema Enforcement | Read Latency (ms) | Workload Fit |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Parquet (Snappy)** | **102,340 B** | **100 KB** | **6.41x (84.4% savings)** | Strong / Typed | ~48 ms | Analytical queries, ML pipelines, feature stores |
| **CSV** | 383,164 B | 374 KB | 1.71x (41.6% savings) | None / Inferred | ~8 ms | Quick inspection, Excel / BI desktop tools |
| **JSON Lines (JSONL)** | 655,831 B | 640 KB | Baseline (1.00x) | Semi-structured | ~18 ms | Streaming logs, REST APIs, document stores |
| **Partitioned Parquet** | 412,236 B | 402 KB | 1.59x | Strong / Partitioned | Sub-partition pruning | Category-filtered scans (`WHERE content_type = '...'`) |

---

## 3. Workload Analysis & Technical Justification

### 3.1 Apache Parquet for Machine Learning Workloads
- **Why it is preferred:** For training and analytical modeling, algorithms frequently project subsets of columns (e.g., dropping text and keeping numerical features). Parquet stores data column-by-column, allowing pyarrow to deserialize only the requested column blocks from disk rather than reading every line.
- **Compression Efficiency:** Snappy dictionary compression achieved an 84.4% reduction in file size compared to JSONL and 73.3% reduction compared to CSV.

### 3.2 Partitioning Analysis (`data/partitioned/content_type=*/`)
- **Structure:**
  ```
  data/partitioned/
  ├── content_type=academic_paper/part-0.parquet
  ├── content_type=article/part-0.parquet
  ├── content_type=blog_post/part-0.parquet
  ├── content_type=creative_writing/part-0.parquet
  ├── content_type=essay/part-0.parquet
  ├── content_type=news_article/part-0.parquet
  ├── content_type=product_review/part-0.parquet
  └── content_type=social_media/part-0.parquet
  ```
- **Trade-off:** For small datasets (1,367 rows), partitioning creates multiple small files (~50 KB each), increasing file overhead compared to a single monolithic Parquet file. However, in production at scale, partitioning enables **partition pruning**, allowing queries filtering on a specific content type to skip reading the rest of the dataset entirely.

### 3.3 Relational Storage (PostgreSQL)
- **Role:** Facilitates SQL-based analytical aggregations, joins, data audits (`raw_audit`, `pipeline_run_log`), and views (`vw_stylometrics_by_content_type`, `vw_model_leaderboard`).
- **Isolation:** Staged and curated data are isolated into structured tables with type constraints and primary keys.

---

## 4. Final Recommendation

- **Machine Learning & Pipeline Core:** Standardize on **Apache Parquet** for raw snapshots, staging, and curated splits.
- **Reporting & Interoperability:** Materialize read-only published **CSV** extracts for external consumers.
- **Relational Analytics:** Use **PostgreSQL** for operational audit logging, transactional monitoring, and SQL reporting views.
