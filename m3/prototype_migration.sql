-- =============================================================================
-- AgroVision Prototype Migration: M2 Rice Pipeline
-- =============================================================================

BEGIN;

-- 1. FACT TABLE RESET (Irreversible)
TRUNCATE features.market_features,
         features.environmental_features,
         serving.serving_predictions;

-- 2. PIPELINE AUDIT TABLE
CREATE TABLE IF NOT EXISTS serving.pipeline_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ran_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    crop_id VARCHAR(32) REFERENCES features.crop_dim(crop_id),
    status VARCHAR(20) NOT NULL CHECK (status IN ('RUNNING', 'COMPLETED', 'FAILED')),
    records_ingested INT DEFAULT 0,
    predictions_generated INT DEFAULT 0,
    error_message TEXT
);

-- 3. LONG-FORMAT PRICE FORECASTS
CREATE TABLE IF NOT EXISTS serving.price_forecasts (
    forecast_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES serving.pipeline_runs(run_id) ON DELETE CASCADE,
    mandi_id VARCHAR(32) NOT NULL REFERENCES features.mandi_dim(mandi_id),
    crop_id VARCHAR(32) NOT NULL REFERENCES features.crop_dim(crop_id),
    horizon_label VARCHAR(10) NOT NULL CHECK (horizon_label IN ('short', '14d', '30d')),
    forecast_origin_date DATE NOT NULL,
    target_date DATE NOT NULL,
    current_modal_price DOUBLE PRECISION NOT NULL,
    predicted_modal_price DOUBLE PRECISION NOT NULL,
    predicted_price_change DOUBLE PRECISION NOT NULL,
    predicted_price_lower DOUBLE PRECISION NOT NULL,
    predicted_price_upper DOUBLE PRECISION NOT NULL,
    naive_baseline_price DOUBLE PRECISION,
    naive_baseline_mae DOUBLE PRECISION,
    volatility_class VARCHAR(20) CHECK (volatility_class IN ('LOW', 'MEDIUM', 'HIGH')),
    volatility_prob_low DOUBLE PRECISION,
    volatility_prob_medium DOUBLE PRECISION,
    volatility_prob_high DOUBLE PRECISION,
    model_version VARCHAR(50) NOT NULL,
    is_experimental BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_run_mandi_crop_horizon UNIQUE (run_id, mandi_id, crop_id, horizon_label)
);

-- 4. MISSING HOOGHLY MANDIS (From rice_group_mappings.json)
INSERT INTO features.mandi_dim (mandi_id, mandi_name, district_id) VALUES
  ('WB_MND_CHINSURAH',      'ChinSurah',             'WB_HG_01'),
  ('WB_MND_KALIPUR_CHAMP',  'Kalipur(Champadanga)',  'WB_HG_01')
ON CONFLICT (mandi_id) DO NOTHING;

-- 5. CANONICAL RICE VARIETIES IN CROP_DIM
INSERT INTO features.crop_dim (crop_id, canonical_name, season_type, base_temperature_gdd) VALUES
  ('RICE_COMMON',        'Rice (Common)',                     'Kharif', 10.0),
  ('RICE_RATNACHUDI',    'Rice (Ratnachudi 718 5-749)',       'Kharif', 10.0),
  ('RICE_SONA_MANSOORI', 'Rice (Sona Mansoori Non Basmati)',  'Kharif', 10.0),
  ('RICE_MASOORI',       'Rice (Masoori-Non Basmti)',         'Kharif', 10.0),
  ('RICE_FINE',          'Rice (Fine)',                       'Kharif', 10.0),
  ('RICE_OTHER',         'Rice (Other)',                      'Kharif', 10.0),
  ('RICE_JAYA',          'Rice (Jaya)',                       'Kharif', 10.0),
  ('RICE_MASURI',        'Rice (Masuri)',                     'Kharif', 10.0)
ON CONFLICT (crop_id) DO NOTHING;

-- 6. GRANT APPLICATION ROLE PERMISSIONS
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA serving TO agrovision;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA features TO agrovision;

COMMIT;