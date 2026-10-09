-- ============================================================
-- sql/schema.sql
-- PostgreSQL schema for the DSS150P pipeline
-- All tables are created with IF NOT EXISTS (rerun-safe)
-- ============================================================

-- Raw ingestion audit table
CREATE TABLE IF NOT EXISTS raw_audit (
    id               SERIAL PRIMARY KEY,
    pipeline_run_id  VARCHAR(64)  NOT NULL,
    batch_id         VARCHAR(64)  NOT NULL,
    source_file      VARCHAR(255) NOT NULL,
    source_checksum  CHAR(64)     NOT NULL,
    row_count        INTEGER      NOT NULL,
    ingested_at_utc  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (pipeline_run_id, batch_id)
);

-- Staging layer: cleaned and typed records
CREATE TABLE IF NOT EXISTS staging (
    id                    SERIAL PRIMARY KEY,
    text_content          TEXT,
    content_type          VARCHAR(100),
    word_count            INTEGER,
    character_count       INTEGER,
    sentence_count        INTEGER,
    lexical_diversity     FLOAT,
    avg_sentence_length   FLOAT,
    avg_word_length       FLOAT,
    punctuation_ratio     FLOAT,
    flesch_reading_ease   FLOAT,
    gunning_fog_index     FLOAT,
    grammar_errors        INTEGER,
    passive_voice_ratio   FLOAT,
    predictability_score  FLOAT,
    burstiness            FLOAT,
    sentiment_score       FLOAT,
    label                 SMALLINT     NOT NULL CHECK (label IN (0, 1)),
    row_hash              CHAR(32),
    pipeline_run_id       VARCHAR(64),
    ingested_at_utc       TIMESTAMPTZ,
    staged_at_utc         TIMESTAMPTZ
);

-- Curated / model-ready table (scaled features)
CREATE TABLE IF NOT EXISTS curated_full (
    id                    SERIAL PRIMARY KEY,
    content_type          FLOAT,
    word_count            FLOAT,
    character_count       FLOAT,
    sentence_count        FLOAT,
    lexical_diversity     FLOAT,
    avg_sentence_length   FLOAT,
    avg_word_length       FLOAT,
    punctuation_ratio     FLOAT,
    flesch_reading_ease   FLOAT,
    gunning_fog_index     FLOAT,
    grammar_errors        FLOAT,
    passive_voice_ratio   FLOAT,
    predictability_score  FLOAT,
    burstiness            FLOAT,
    sentiment_score       FLOAT,
    label                 SMALLINT NOT NULL,
    split                 VARCHAR(10)  -- 'train' or 'test'
);

-- Model evaluation metrics
CREATE TABLE IF NOT EXISTS model_metrics (
    id               SERIAL PRIMARY KEY,
    run_id           VARCHAR(64),
    evaluated_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    classifier       VARCHAR(100) NOT NULL,
    accuracy         FLOAT,
    f1_weighted      FLOAT,
    f1_macro         FLOAT,
    roc_auc          FLOAT,
    random_seed      INTEGER,
    test_size        FLOAT
);

-- Quarantine table for rows that failed validation
CREATE TABLE IF NOT EXISTS quarantine (
    id               SERIAL PRIMARY KEY,
    quarantine_date  DATE         NOT NULL DEFAULT CURRENT_DATE,
    reason           VARCHAR(255),
    source_file      VARCHAR(255),
    raw_data         JSONB
);

-- Pipeline run log
CREATE TABLE IF NOT EXISTS pipeline_run_log (
    id               SERIAL PRIMARY KEY,
    run_id           VARCHAR(64)  NOT NULL UNIQUE,
    started_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ,
    status           VARCHAR(20)  DEFAULT 'RUNNING',
    stage            VARCHAR(50),
    notes            TEXT
);
