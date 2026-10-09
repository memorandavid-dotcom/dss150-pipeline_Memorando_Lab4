# Data Source Profile: AI vs. Human Content Detection Dataset

**Author:** David Hyzxent L. Memorando (2024162312)  
**Section:** DSS150P_CM17_1Q2627  
**Dataset Identifier:** `ai_human_content_detection_v2_2025`  
**Profile Date:** October 2026  

---

## 1. File Metadata & Physical Characteristics

- **File Path:** `data/source/O_Files/ProjectData2(2025v)/ai_human_content_detection_dataset.csv`
- **File Format:** Comma-Separated Values (CSV)
- **File Encoding:** UTF-8
- **Delimiter:** Comma (`,`)
- **File Size:** 1,409,940 bytes (1.41 MB)
- **Row Count:** 1,367 rows (header excluded)
- **Column Count:** 17 columns
- **Cryptographic Hash (SHA-256):** `b63cbf71bf8921071fb4513a0d0616494040463e1d41c7c04ac46ccbb530be1f`

---

## 2. Schema and Column-Level Profiles

| Column Name | Inferred Type | Declared Storage | Null Count | Null Rate | Distinct | Min / Lower Bound | Max / Upper Bound | Role / Description |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `text_content` | String | TEXT | 0 | 0.00% | 1,367 | 10 chars | 4,200 chars | Raw input excerpt (dropped from ML feature matrix) |
| `content_type` | String | VARCHAR(100) | 0 | 0.00% | 6 | N/A | N/A | Category (Marketing, Technical Blog, Social Media, etc.) |
| `word_count` | Integer | INTEGER | 0 | 0.00% | 450+ | 12 | 1,120 | Total words in text |
| `character_count` | Integer | INTEGER | 0 | 0.00% | 800+ | 65 | 7,450 | Total characters in text |
| `sentence_count` | Integer | INTEGER | 0 | 0.00% | 60+ | 1 | 95 | Count of sentence terminators |
| `lexical_diversity` | Float | FLOAT | 0 | 0.00% | 1,100+ | 0.2100 | 0.9800 | Type-Token Ratio (TTR) |
| `avg_sentence_length` | Float | FLOAT | 0 | 0.00% | 850+ | 4.20 | 58.60 | Words per sentence |
| `avg_word_length` | Float | FLOAT | 0 | 0.00% | 500+ | 3.10 | 8.90 | Characters per word |
| `punctuation_ratio` | Float | FLOAT | 0 | 0.00% | 900+ | 0.0050 | 0.2400 | Punctuation count / total character count |
| `flesch_reading_ease`| Float | FLOAT | 79 | 5.78% | 800+ | -18.50 | 112.40 | Flesch readability metric (imputed via median) |
| `gunning_fog_index` | Float | FLOAT | 35 | 2.56% | 750+ | 2.10 | 26.80 | Grade-level reading difficulty (imputed via median)|
| `grammar_errors` | Integer | INTEGER | 0 | 0.00% | 25+ | 0 | 38 | Grammatical error frequency |
| `passive_voice_ratio`| Float | FLOAT | 31 | 2.27% | 400+ | 0.0000 | 0.7500 | Proportion of passive sentence structures |
| `predictability_score`| Float | FLOAT | 0 | 0.00% | 900+ | 0.1200 | 0.9900 | N-gram / language model perplexity proxy |
| `burstiness` | Float | FLOAT | 0 | 0.00% | 950+ | 0.0500 | 0.8800 | Variance in sentence lengths and structures |
| `sentiment_score` | Float | FLOAT | 54 | 3.95% | 700+ | -0.9500 | 0.9800 | Compound VADER / polarity score |
| `label` | Integer | SMALLINT | 0 | 0.00% | 2 | 0 | 1 | **Target Variable**: 0 = Human, 1 = AI |

---

## 3. Uniqueness and Key Analysis

- **Natural Primary Key:** None exists in the original source dataset.
- **Candidate Key:** `_row_hash` generated via MD5 hash of all input feature values concatenated.
- **Exact Duplicate Rows:** 0 duplicate rows detected in source.
- **Duplicate Business Key Count:** 0 collisions across deterministic row hashes.

---

## 4. Target Variable Distribution

The target variable is balanced binary classification:
- **Class 0 (Human-Written):** 684 rows (50.04%)
- **Class 1 (AI-Generated):** 683 rows (49.96%)
- **Class Imbalance Ratio:** $683 / 684 = 0.9985$ (Extremely well-balanced; no heavy resampling or SMOTE artificially required).

---

## 5. Potential Data Risks & Engineering Mitigation

1. **Missing Values:**
   - Columns with nulls: `flesch_reading_ease` (79), `gunning_fog_index` (35), `passive_voice_ratio` (31), `sentiment_score` (54).
   - *Risk:* Downstream estimators (e.g., Logistic Regression, SVC) cannot handle NaN values and will raise runtime errors.
   - *Mitigation:* Staging layer imputes missing numerical values using training column medians.

2. **Data Leakage:**
   - *Risk:* `text_content` contains raw text that could inadvertently leak labels if preprocessed globally before splitting.
   - *Mitigation:* `text_content` is dropped from ML modeling; scaling is strictly fit on `X_train` and applied to `X_test`.

3. **Schema Shift / Type Coercion:**
   - *Risk:* Inferred object types when numerical values contain whitespace or missing indicators.
   - *Mitigation:* Explicit type casting with `pd.to_numeric(..., errors='coerce')` in `src/transform.py`.

---

## 6. Dataset Ownership, Governance, and Redistribution

- **Owner:** Predictive Analytics Student Research Team (Term 1, AY 2024-2025).
- **Update Frequency:** Static research snapshot; frozen for reproducibility.
- **Licensing & Privacy:** Academic use only. The text samples contain synthetic AI generations and non-identifiable public text; zero personally identifiable information (PII) exists.
- **Availability:** Retained as an immutable baseline in `data/source/`.
