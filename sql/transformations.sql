-- ============================================================
-- sql/transformations.sql
-- Analytical SQL transformations and reporting views
-- ============================================================

-- 1. View: Stylometric profiles by content type and target class
CREATE OR REPLACE VIEW vw_stylometrics_by_content_type AS
SELECT
    content_type,
    label,
    CASE WHEN label = 1 THEN 'AI-Generated' ELSE 'Human-Written' END AS label_desc,
    COUNT(*) AS total_samples,
    ROUND(AVG(word_count)::NUMERIC, 2) AS avg_words,
    ROUND(AVG(lexical_diversity)::NUMERIC, 4) AS avg_lexical_diversity,
    ROUND(AVG(avg_sentence_length)::NUMERIC, 2) AS avg_sentence_len,
    ROUND(AVG(flesch_reading_ease)::NUMERIC, 2) AS avg_flesch_reading_ease,
    ROUND(AVG(gunning_fog_index)::NUMERIC, 2) AS avg_gunning_fog,
    ROUND(AVG(predictability_score)::NUMERIC, 4) AS avg_predictability,
    ROUND(AVG(burstiness)::NUMERIC, 4) AS avg_burstiness,
    ROUND(AVG(sentiment_score)::NUMERIC, 4) AS avg_sentiment
FROM staging
GROUP BY content_type, label
ORDER BY content_type, label;

-- 2. View: Readability contrast between AI and Human texts
CREATE OR REPLACE VIEW vw_readability_contrast AS
SELECT
    CASE WHEN label = 1 THEN 'AI-Generated' ELSE 'Human-Written' END AS author_type,
    COUNT(*) AS n_records,
    ROUND(AVG(flesch_reading_ease)::NUMERIC, 2) AS mean_reading_ease,
    ROUND(STDDEV(flesch_reading_ease)::NUMERIC, 2) AS std_reading_ease,
    ROUND(AVG(gunning_fog_index)::NUMERIC, 2) AS mean_fog_index,
    ROUND(AVG(passive_voice_ratio)::NUMERIC, 4) AS mean_passive_voice_ratio,
    ROUND(AVG(predictability_score)::NUMERIC, 4) AS mean_predictability
FROM staging
GROUP BY label;

-- 3. View: Model Evaluation Leaderboard
CREATE OR REPLACE VIEW vw_model_leaderboard AS
SELECT
    classifier,
    ROUND(accuracy::NUMERIC, 4) AS test_accuracy,
    ROUND(f1_weighted::NUMERIC, 4) AS f1_weighted,
    ROUND(f1_macro::NUMERIC, 4) AS f1_macro,
    ROUND(roc_auc::NUMERIC, 4) AS roc_auc,
    evaluated_at,
    RANK() OVER (ORDER BY f1_weighted DESC) as rank
FROM model_metrics
ORDER BY rank;

-- 4. Analytical Transformation Table: Enriched stylometric ratios
CREATE TABLE IF NOT EXISTS staging_enriched AS
SELECT
    id,
    content_type,
    label,
    word_count,
    character_count,
    lexical_diversity,
    avg_sentence_length,
    -- Synthesized complexity ratio
    ROUND((character_count::NUMERIC / NULLIF(word_count, 0)), 2) AS chars_per_word,
    -- Standardized readability metric
    ROUND((flesch_reading_ease / 100.0)::NUMERIC, 4) AS flesch_normalized,
    predictability_score,
    burstiness,
    sentiment_score,
    staged_at_utc
FROM staging;
