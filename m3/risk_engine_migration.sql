-- =============================================================================
-- AgroVision Stage 2 Migration: Weather Shock Signals & Composite Risk Serving
-- Non-destructive: preserves all existing market_features & price_forecasts
-- =============================================================================

BEGIN;

-- 1. Extend environmental_features hypertable with post-harvest shock signals
ALTER TABLE features.environmental_features
    ADD COLUMN IF NOT EXISTS rh_mean_pct DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS forecast_rain_48h_mm DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS trailing_rain_3d_mm DOUBLE PRECISION,
    ADD COLUMN IF NOT EXISTS weather_source VARCHAR(32) DEFAULT 'open-meteo';

-- 2. Multi-Source Composite Risk & Decision Serving Table
CREATE TABLE IF NOT EXISTS serving.risk_assessments (
    assessment_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID REFERENCES serving.pipeline_runs(run_id) ON DELETE CASCADE,
    mandi_id VARCHAR(32) NOT NULL REFERENCES features.mandi_dim(mandi_id),
    crop_id VARCHAR(32) NOT NULL REFERENCES features.crop_dim(crop_id),
    district_id VARCHAR(32) NOT NULL REFERENCES features.location_dim(district_id),
    assessed_origin_date DATE NOT NULL,
    horizon_label VARCHAR(10) NOT NULL CHECK (horizon_label IN ('short', '14d', '30d')),

    -- Decomposed Risk Components [0.0, 1.0]
    v_norm DOUBLE PRECISION NOT NULL,
    h_anomaly DOUBLE PRECISION NOT NULL,
    storage_vuln DOUBLE PRECISION NOT NULL,
    composite_risk_score DOUBLE PRECISION NOT NULL CHECK (composite_risk_score >= 0.0 AND composite_risk_score <= 1.0),
    risk_regime VARCHAR(16) NOT NULL CHECK (risk_regime IN ('LOW', 'MEDIUM', 'HIGH')),

    -- Exogenous Weather Inputs Used
    forecast_rain_48h_mm DOUBLE PRECISION NOT NULL,
    trailing_rain_3d_mm DOUBLE PRECISION NOT NULL,
    rh_mean_pct DOUBLE PRECISION,
    storage_avail_index DOUBLE PRECISION NOT NULL,

    -- Decision Override & Weather-Adjusted Downside
    advisory_action VARCHAR(32) NOT NULL,
    advisory_headline TEXT NOT NULL,
    advisory_reason TEXT NOT NULL,
    weather_adjusted_lower DOUBLE PRECISION NOT NULL,

    -- M1 Crop Yield Integration
    predicted_yield_kg_ha DOUBLE PRECISION,
    yield_method VARCHAR(64),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_risk_run_mandi_crop_horizon UNIQUE (run_id, mandi_id, crop_id, horizon_label)
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA serving TO agrovision;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA features TO agrovision;

COMMIT;
