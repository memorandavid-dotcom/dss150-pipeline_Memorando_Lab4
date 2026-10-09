# Reproducibility & Tolerance Verification Report

**Author:** David Hyzxent L. Memorando  
**Student Number:** 2024162312  
**Section:** DSS150P_CM17_1Q2627  
**Course:** DSS150P — Fundamentals of Data Engineering  
**Activity:** Integrated Individual Laboratory Activity #4  
**Date:** October 2026  

---

## 1. Prior Study Citation & Ground Truth Baseline

This data product reproduces the predictive analytics research originally conducted and reported in:

> **Memorando, D. H. L., & Dela Llarte, J. M. D. (2025).** *A Comparative Analysis of Machine Learning Classifiers for Distinguishing Human-Written and AI-Generated Text Using Stylometric and Lexical Features.* School of Information and Technology, Mapúa University, Makati, Philippines.

### Original Experimental Setup:
- **Project Pipeline:** *Rosetta* — an interpretable supervised classification framework.
- **Source Dataset:** `rosetta_dataset_tabular.xlsx` (1,611 raw observations, reduced to **1,499 complete rows** via `dropna` across the 9 primary stylometric features; `edit_level` excluded due to 84.9% missingness).
- **Class Distribution:** AI-Generated (Class 1) = 833 (55.6%), Human-Written (Class 0) = 666 (44.4%).
- **Data Split:** 80/20 train/test split using `scikit-learn` `train_test_split(..., test_size=0.20, random_state=42, stratify=y)`.
- **Sample Sizes:** Training = 1,199 samples; Held-out Test Set = **300 samples** (167 AI, 133 Human).
- **Feature Scaling:** `StandardScaler` fit exclusively on `X_train` and applied to `X_test`.
- **Primary Feature Set (Tabular Baseline):** 9 stylometric indicators (`word_count`, `character_count`, `sentence_count`, `lexical_diversity`, `avg_sentence_length`, `avg_word_length`, `punctuation_ratio`, `flesch_reading_ease`, `gunning_fog_index`).

---

## 2. Real Baseline vs. Current Pipeline Execution Evidence

Below is the honest comparative analysis between the **actual reported numbers from the original paper** (stored in `metadata/expected_metrics.json`) and the empirical evaluation produced by this pipeline run (stored in `outputs/metrics.json` and checked via `outputs/reproducibility_check.json` under tolerance threshold $\pm 0.05$):

| Classifier | Baseline Metric (Paper) | Actual Run Metric (Pipeline) | Absolute Delta ($\Delta$) | Tolerance Limit | Verification Status | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | **Weighted F1: 0.5800**<br>(Accuracy: 0.5767) | **Weighted F1: 0.5584**<br>(Accuracy: 0.5584) | **0.0216** | $\pm 0.0500$ | **PASS** | Within tolerance; exhibits high bias consistent with paper. |
| **Decision Tree** | **Weighted F1: 0.4800**<br>(Accuracy: 0.5267) | **Weighted F1: 0.5497**<br>(Accuracy: 0.5511) | **0.0697** | $\pm 0.0500$ | **FAIL (Misses Tolerance)** | Overfitting characteristics vary due to feature space differences. |
| **XGBoost / Gradient Boost** | **Weighted F1: 0.5900**<br>(Accuracy: 0.5933) | **Weighted F1: 0.5145**<br>(Accuracy: 0.5146) | **0.0755** | $\pm 0.0500$ | **FAIL (Misses Tolerance)** | Default threshold 0.50 vs paper's balanced threshold ($\tau=0.5245$). |
| **Random Forest** | *Not evaluated in prior study* | Weighted F1: 0.4800<br>(Accuracy: 0.4818) | N/A | N/A | **INFO** | Added during exploratory benchmarking. |
| **SVC** | *Not evaluated in prior study* | Weighted F1: 0.5547<br>(Accuracy: 0.5547) | N/A | N/A | **INFO** | Added during exploratory benchmarking. |

---

## 3. Engineering Diagnosis: Why Do The Metrics Differ?

In accordance with the core laboratory goal of **transparent and honest reproducibility**, we identify five concrete engineering factors that account for the metric differences:

1. **Source Dataset Snapshot Discrepancy (1,367 vs. 1,499 rows):**
   - The authoritative CSV provided in this repository (`ai_human_content_detection_dataset.csv`) contains **1,367 rows**, whereas the original paper analyzed a **1,499-row** dataset derived from an earlier 1,611-row Excel workbook.
   - Because the source row population differs by 132 records, the resulting 80/20 test set contains **274 samples** in this pipeline rather than the **300 samples** evaluated in the publication.

2. **Missing Value Imputation vs. Listwise Deletion (`dropna`):**
   - In the paper, rows containing missing values in readability metrics were discarded via `dropna()`.
   - In this production data pipeline, to prevent data loss in streaming/incremental batches, numerical nulls (`flesch_reading_ease`: 79 nulls, `gunning_fog_index`: 35 nulls, etc.) were imputed using the column median. This alters the feature variance and boundary distributions slightly.

3. **Classification Threshold Tuning ($\tau = 0.5245$):**
   - The paper's strongest XGBoost results relied on a post-hoc probability threshold calibration ($\tau = 0.5245$) selected on an internal validation set to balance class recall.
   - The automated CLI pipeline evaluates models at the standard decision boundary ($\tau = 0.5000$). At $\tau = 0.50$, the original paper also reported higher class asymmetry (AI recall 0.65 vs Human recall 0.53).

4. **Expanded Feature Space (15 features vs. 9 features):**
   - The paper's tabular baseline isolated strictly 9 stylometric features.
   - The current dataset contains additional engineered columns (`grammar_errors`, `passive_voice_ratio`, `predictability_score`, `burstiness`, `sentiment_score`), altering tree splits and regularization penalties.

5. **Runtime Environment & Package Implementation:**
   - In Python 3.14 on Windows, XGBoost binary wheels fall back gracefully to scikit-learn's `GradientBoostingClassifier` with default boosting parameters, which has minor implementation differences from the native C++ DMatrix boosting engine of `xgboost 2.0.3`.

---

## 4. Rerun Safety & Determinism Verification

Despite dataset snapshot differences, the engineered data pipeline guarantees **deterministic rerun execution**:
- Re-running the pipeline on the identical source snapshot produces the exact same train/test splits, scaler parameters (`metadata/split_manifest.json`), and model weights.
- All evaluation metrics are deterministic across repeated runs when fixed to seed `42`.
- Zero duplicate records are introduced into staging or curated layers on repeated execution.

---

## 5. Summary Conclusion

By replacing self-referential targets with the **true published results of Memorando & Dela Llarte (2025)**, this laboratory provides authentic, audit-ready scientific reproducibility evidence. The pipeline successfully flags where and why results diverge due to dataset versioning and thresholding differences, fulfilling the data engineering lifecycle requirements of DSS150P.
