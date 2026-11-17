-- =============================================================================
-- AgroVision — DEMO QUERIES (read-only except section 10, which always rolls back)
-- Run each block separately in DBeaver (Ctrl+Enter) or via psql.
-- =============================================================================

-- 1. INVENTORY: every table, populated -----------------------------------------
SELECT 'features.location_dim' AS table_name, COUNT(*) AS row_count FROM features.location_dim
UNION ALL SELECT 'features.crop_dim',               COUNT(*) FROM features.crop_dim
UNION ALL SELECT 'features.mandi_dim',              COUNT(*) FROM features.mandi_dim
UNION ALL SELECT 'features.ndvi_baseline',          COUNT(*) FROM features.ndvi_baseline
UNION ALL SELECT 'features.environmental_features', COUNT(*) FROM features.environmental_features
UNION ALL SELECT 'features.market_features',        COUNT(*) FROM features.market_features
UNION ALL SELECT 'features.yield_history',          COUNT(*) FROM features.yield_history
UNION ALL SELECT 'serving.serving_predictions',     COUNT(*) FROM serving.serving_predictions;

-- 2a. TIMESCALEDB PROOF: which tables are hypertables ---------------------------
SELECT hypertable_schema, hypertable_name, num_dimensions, num_chunks
FROM timescaledb_information.hypertables;

-- 2b. Partitioning geometry (7-day chunks) --------------------------------------
SELECT hypertable_name, column_name, dimension_type, time_interval
FROM timescaledb_information.dimensions;

-- 2c. The actual chunks the seed data landed in ---------------------------------
SELECT hypertable_name, chunk_name, range_start, range_end
FROM timescaledb_information.chunks
WHERE hypertable_schema = 'features'
ORDER BY hypertable_name, range_start;

-- 3. DIMENSIONS: district -> mandi relationship + storage index ----------------
SELECT l.district_id, l.district_name, l.storage_avail_index,
       COUNT(m.mandi_id) AS mandis,
       STRING_AGG(m.mandi_name, ', ' ORDER BY m.mandi_name) AS mandi_names
FROM features.location_dim l
LEFT JOIN features.mandi_dim m ON m.district_id = l.district_id
GROUP BY l.district_id, l.district_name, l.storage_avail_index
ORDER BY l.district_id;

-- 4. MARKET FEATURES: the synthetic glut story at Arambagh ---------------------
SELECT (time AT TIME ZONE 'Asia/Kolkata')::date AS trade_date,
       modal_price_inr, arrivals_mt, rolling_7d_ewma_std, arrival_to_volume_ratio,
       ROUND((100.0 * (modal_price_inr - FIRST_VALUE(modal_price_inr) OVER (ORDER BY time))
              / FIRST_VALUE(modal_price_inr) OVER (ORDER BY time))::numeric, 1) AS price_change_pct
FROM features.market_features
WHERE mandi_id = 'WB_MND_ARAMBAGH' AND crop_id = 'POTATO_JYOTI'
ORDER BY time;

-- 5. THE ASOF JOIN: each mandi price row + latest environmental observation <= trade time
--    (Postgres has no native ASOF JOIN; LEFT JOIN LATERAL + ORDER BY time DESC LIMIT 1 is the equivalent.)
SELECT (m.time AT TIME ZONE 'Asia/Kolkata')::date AS trade_date,
       md.mandi_name, m.crop_id, m.modal_price_inr, m.arrivals_mt,
       (e.env_time AT TIME ZONE 'Asia/Kolkata')::date AS env_obs_date,
       e.cumulative_gdd, e.ndvi_anomaly, e.moisture_deficit
FROM features.market_features m
JOIN features.mandi_dim md ON md.mandi_id = m.mandi_id
LEFT JOIN LATERAL (
    SELECT ef.time AS env_time, ef.cumulative_gdd, ef.ndvi_anomaly, ef.moisture_deficit
    FROM features.environmental_features ef
    WHERE ef.district_id = m.district_id
      AND ef.crop_id     = m.crop_id
      AND ef.time       <= m.time
    ORDER BY ef.time DESC
    LIMIT 1
) e ON TRUE
WHERE m.crop_id = 'POTATO_JYOTI'
ORDER BY md.mandi_name, m.time;

-- 6. NDVI DESIGN CHECK: stored anomaly vs recomputed from the calendar-day baseline
SELECT (e.time AT TIME ZONE 'Asia/Kolkata')::date AS obs_date, e.district_id,
       e.mean_ndvi, b.multi_year_mean_ndvi AS baseline,
       e.ndvi_anomaly AS stored_anomaly,
       ROUND((e.mean_ndvi - b.multi_year_mean_ndvi)::numeric, 3) AS recomputed_anomaly
FROM features.environmental_features e
JOIN features.ndvi_baseline b
  ON b.district_id = e.district_id
 AND b.day_of_year = EXTRACT(DOY FROM e.time AT TIME ZONE 'Asia/Kolkata')::int
WHERE e.crop_id = 'POTATO_JYOTI'
ORDER BY e.district_id, e.time;

-- 7. GRAIN TRADEOFF: same district-day weather repeated per crop; only GDD differs
SELECT (time AT TIME ZONE 'Asia/Kolkata')::date AS obs_date, district_id, crop_id,
       season_start_date, t_min_c, t_max_c, precipitation_mm, cumulative_gdd
FROM features.environmental_features
WHERE district_id = 'WB_HG_01' AND time = '2026-01-30 00:00:00+05:30'
ORDER BY crop_id;

-- 8. YIELD HISTORY: year-over-year deviation vs mean of prior years (the yield-model target idea)
WITH y AS (
  SELECT district_id, crop_id, crop_year, season, area_hectares, production_tonnes, yield_tonnes_per_ha,
         AVG(yield_tonnes_per_ha) OVER (
           PARTITION BY district_id, crop_id, season
           ORDER BY crop_year
           ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
         ) AS prior_mean
  FROM features.yield_history
)
SELECT district_id, crop_id, crop_year, yield_tonnes_per_ha,
       ROUND((production_tonnes / area_hectares)::numeric, 1) AS yield_check,
       ROUND(prior_mean::numeric, 2) AS prior_mean,
       ROUND((100 * (yield_tonnes_per_ha - prior_mean) / prior_mean)::numeric, 1) AS deviation_pct
FROM y
WHERE district_id = 'WB_HG_01' AND crop_id = 'POTATO_JYOTI'
ORDER BY crop_year;

-- 9a. SERVING LAYER: what the dashboard's choropleth/table would read (STUB data) ------
SELECT l.district_name, c.canonical_name,
       p.composite_risk_label AS risk_label,
       ROUND(p.risk_score_float::numeric, 2) AS risk_score,
       p.volatility_regime,
       (p.volatility_class_probs ->> 'Crash-Glut')::numeric AS p_crash_glut,
       p.pred_price_7d, p.pred_price_7d_lower, p.pred_price_7d_upper,
       p.pred_yield_deviation_pct, p.model_version
FROM serving.serving_predictions p
JOIN features.location_dim l ON l.district_id = p.district_id
JOIN features.crop_dim     c ON c.crop_id     = p.crop_id
ORDER BY p.risk_score_float DESC;

-- 9b. Explainability payload unpacked from JSONB (per-feature attributions) -----------
SELECT p.district_id, p.crop_id, s.key AS feature, s.value::numeric AS attribution
FROM serving.serving_predictions p,
     jsonb_each_text(p.shap_attributions) AS s
WHERE p.district_id = 'WB_HG_01' AND p.crop_id = 'POTATO_JYOTI'
ORDER BY ABS(s.value::numeric) DESC;

-- 10. INTEGRITY: constraints reject bad data (run in psql; each block ends in ROLLBACK) -----
-- 10a. CHECK constraint: storage index must be within 0..1
BEGIN;
INSERT INTO features.location_dim (district_id, district_name, state_name, storage_avail_index)
VALUES ('WB_XX_99', 'Test District', 'West Bengal', 1.7);
ROLLBACK;

-- 10b. FOREIGN KEY: a mandi must belong to an existing district
BEGIN;
INSERT INTO features.mandi_dim (mandi_id, mandi_name, district_id)
VALUES ('WB_MND_GHOST', 'Ghost Mandi', 'NO_SUCH_DISTRICT');
ROLLBACK;