-- AgroVision Physical Schema

-- 1. Root Dimension: Location
CREATE TABLE IF NOT EXISTS features.location_dim (
    district_id VARCHAR(32) PRIMARY KEY,
    district_name VARCHAR(100) NOT NULL,
    state_name VARCHAR(100) NOT NULL,
    storage_avail_index DOUBLE PRECISION NOT NULL CHECK (storage_avail_index >= 0.0 AND storage_avail_index <= 1.0)
);

-- 2. Root Dimension: Crop
CREATE TABLE IF NOT EXISTS features.crop_dim (
    crop_id VARCHAR(32) PRIMARY KEY,
    canonical_name VARCHAR(100) NOT NULL,
    season_type VARCHAR(20) NOT NULL CHECK (season_type IN ('Kharif', 'Rabi', 'Zaid')),
    base_temperature_gdd DOUBLE PRECISION NOT NULL CHECK (base_temperature_gdd >= 0.0)
);

-- 3. Dependent Dimension: Mandi
CREATE TABLE IF NOT EXISTS features.mandi_dim (
    mandi_id VARCHAR(32) PRIMARY KEY,
    mandi_name VARCHAR(100) NOT NULL,
    district_id VARCHAR(32) NOT NULL REFERENCES features.location_dim(district_id)
);

-- 4. Dependent Dimension: NDVI Baseline
CREATE TABLE IF NOT EXISTS features.ndvi_baseline (
    district_id VARCHAR(32) NOT NULL REFERENCES features.location_dim(district_id),
    day_of_year INT NOT NULL CHECK (day_of_year >= 1 AND day_of_year <= 366),
    multi_year_mean_ndvi DOUBLE PRECISION NOT NULL CHECK (multi_year_mean_ndvi >= -1.0 AND multi_year_mean_ndvi <= 1.0),
    PRIMARY KEY (district_id, day_of_year)
);

-- 5. Feature Store: Environmental Features Base Table
CREATE TABLE IF NOT EXISTS features.environmental_features (
    time TIMESTAMPTZ NOT NULL,
    district_id VARCHAR(32) NOT NULL REFERENCES features.location_dim(district_id),
    crop_id VARCHAR(32) NOT NULL REFERENCES features.crop_dim(crop_id),
    season_start_date DATE NOT NULL,
    precipitation_mm DOUBLE PRECISION,
    t_min_c DOUBLE PRECISION,
    t_max_c DOUBLE PRECISION,
    mean_ndvi DOUBLE PRECISION,
    ndvi_anomaly DOUBLE PRECISION,
    cumulative_gdd DOUBLE PRECISION,
    moisture_deficit DOUBLE PRECISION,
    PRIMARY KEY (time, district_id, crop_id)
);

-- Converting environmental_features base table into a TimescaleDB hypertable
SELECT create_hypertable(
    'features.environmental_features',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 6. Feature Store: Market Features Base Table
CREATE TABLE IF NOT EXISTS features.market_features (
    time TIMESTAMPTZ NOT NULL,
    mandi_id VARCHAR(32) NOT NULL REFERENCES features.mandi_dim(mandi_id),
    district_id VARCHAR(32) NOT NULL REFERENCES features.location_dim(district_id),
    crop_id VARCHAR(32) NOT NULL REFERENCES features.crop_dim(crop_id),
    modal_price_inr DOUBLE PRECISION,
    arrivals_mt DOUBLE PRECISION,
    rolling_7d_ewma_std DOUBLE PRECISION,
    rolling_14d_ewma_std DOUBLE PRECISION,
    rolling_30d_ewma_std DOUBLE PRECISION,
    arrival_to_volume_ratio DOUBLE PRECISION,
    price_entropy DOUBLE PRECISION,
    rolling_30d_momentum DOUBLE PRECISION,
    PRIMARY KEY (time, mandi_id, crop_id)
);

-- Converting market_features base table into a TimescaleDB hypertable
SELECT create_hypertable(
    'features.market_features',
    'time',
    chunk_time_interval => INTERVAL '7 days',
    if_not_exists => TRUE
);

-- 7. Ground Truth: Yield History
CREATE TABLE IF NOT EXISTS features.yield_history (
    district_id VARCHAR(32) NOT NULL REFERENCES features.location_dim(district_id),
    crop_id VARCHAR(32) NOT NULL REFERENCES features.crop_dim(crop_id),
    crop_year INT NOT NULL CHECK (crop_year >= 1900),
    season VARCHAR(20) NOT NULL,
    area_hectares DOUBLE PRECISION CHECK (area_hectares >= 0.0),
    production_tonnes DOUBLE PRECISION CHECK (production_tonnes >= 0.0),
    yield_tonnes_per_ha DOUBLE PRECISION CHECK (yield_tonnes_per_ha >= 0.0),
    PRIMARY KEY (district_id, crop_id, crop_year, season)
);

-- 8. Serving Layer: Model Predictions & Inference Store
CREATE TABLE IF NOT EXISTS serving.serving_predictions (
    prediction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    model_version VARCHAR(50) NOT NULL,
    forecast_target_date DATE NOT NULL,
    district_id VARCHAR(32) NOT NULL REFERENCES features.location_dim(district_id),
    crop_id VARCHAR(32) NOT NULL REFERENCES features.crop_dim(crop_id),
    
    -- Yield Prediction
    pred_yield_deviation_pct DOUBLE PRECISION,
    
    -- Price Forecasts with Confidence Intervals
    pred_price_7d DOUBLE PRECISION,
    pred_price_7d_lower DOUBLE PRECISION,
    pred_price_7d_upper DOUBLE PRECISION,
    pred_price_14d DOUBLE PRECISION,
    pred_price_14d_lower DOUBLE PRECISION,
    pred_price_14d_upper DOUBLE PRECISION,
    pred_price_30d DOUBLE PRECISION,
    pred_price_30d_lower DOUBLE PRECISION,
    pred_price_30d_upper DOUBLE PRECISION,
    
    -- Risk & Volatility Metrics
    risk_score_float DOUBLE PRECISION CHECK (risk_score_float >= 0.0 AND risk_score_float <= 1.0),
    composite_risk_label VARCHAR(20) CHECK (composite_risk_label IN ('Low', 'Medium', 'High')),
    volatility_class_probs JSONB,
    volatility_regime VARCHAR(50),
    
    -- Explainability
    shap_attributions JSONB
);

-- 9. Permissions: Table-level and default privileges for application role(This is for my system only)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA features TO agrovision;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA serving TO agrovision;

ALTER DEFAULT PRIVILEGES IN SCHEMA features GRANT ALL PRIVILEGES ON TABLES TO agrovision;
ALTER DEFAULT PRIVILEGES IN SCHEMA serving GRANT ALL PRIVILEGES ON TABLES TO agrovision;