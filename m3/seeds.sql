-- =============================================================================
-- AgroVision — DEMO SEED DATA  (synthetic / illustrative — NOT real Agmarknet/IMD/Bhuvan data)
-- Run as the app role over TCP:
--   psql -h 127.0.0.1 -U agrovision -d agrovision -f seeds.sql
--
-- Conventions used:
--   * Timestamps are IST midnight (+05:30). Day-of-year lookups in queries use
--     "AT TIME ZONE 'Asia/Kolkata'" so they don't shift with the session timezone.
--   * modal_price_inr = INR per quintal (Agmarknet convention); arrivals_mt = metric tonnes.
--   * crop_year = starting year of the season (Rabi 2025 = the 2025-26 Rabi season).
--   * serving_predictions rows are STUBS (model_version = 'stub-v0'); no trained model exists yet.
--   * Risk-label cut-offs used in the stubs (<0.33 Low, <0.66 Medium, else High) are placeholders.
--
-- Re-running: dimension/fact inserts are idempotent (ON CONFLICT DO NOTHING), but
-- serving_predictions has no natural key and WILL duplicate. For a clean reseed, run
-- the TRUNCATE block at the bottom first.
-- =============================================================================

BEGIN;

-- 1. LOCATION_DIM (8 rows) -----------------------------------------------------
INSERT INTO features.location_dim (district_id, district_name, state_name, storage_avail_index) VALUES
  ('WB_HG_01', 'Hooghly',            'West Bengal', 0.72),
  ('WB_BR_01', 'Purba Bardhaman',    'West Bengal', 0.81),
  ('WB_BK_01', 'Bankura',            'West Bengal', 0.55),
  ('WB_ND_01', 'Nadia',              'West Bengal', 0.38),
  ('WB_BI_01', 'Birbhum',            'West Bengal', 0.41),
  ('WB_PM_01', 'Paschim Medinipur',  'West Bengal', 0.47),
  ('WB_MU_01', 'Murshidabad',        'West Bengal', 0.29),
  ('WB_JP_01', 'Jalpaiguri',         'West Bengal', 0.33)
ON CONFLICT (district_id) DO NOTHING;

-- 2. CROP_DIM (6 rows) ---------------------------------------------------------
-- base_temperature_gdd values are PLACEHOLDERS — Member 1 owns the real T_base values.
INSERT INTO features.crop_dim (crop_id, canonical_name, season_type, base_temperature_gdd) VALUES
  ('POTATO_JYOTI',        'Potato (Kufri Jyoti)',        'Rabi',   7.0),
  ('POTATO_CHANDRAMUKHI', 'Potato (Kufri Chandramukhi)', 'Rabi',   7.0),
  ('ONION_RED',           'Onion (Red)',                 'Rabi',   5.0),
  ('TOMATO_HYBRID',       'Tomato (Hybrid)',             'Rabi',  10.0),
  ('PADDY_SWARNA',        'Paddy (Swarna)',              'Kharif',10.0),
  ('WHEAT_HD2967',        'Wheat (HD-2967)',             'Rabi',   5.0)
ON CONFLICT (crop_id) DO NOTHING;

-- 3. MANDI_DIM (9 rows) --------------------------------------------------------
-- Names are illustrative; verify against Agmarknet's actual market list once Member 2's data lands.
INSERT INTO features.mandi_dim (mandi_id, mandi_name, district_id) VALUES
  ('WB_MND_ARAMBAGH',     'Arambagh',     'WB_HG_01'),
  ('WB_MND_BARDHAMAN',    'Bardhaman',    'WB_BR_01'),
  ('WB_MND_KATWA',        'Katwa',        'WB_BR_01'),
  ('WB_MND_BISHNUPUR',    'Bishnupur',    'WB_BK_01'),
  ('WB_MND_KRISHNANAGAR', 'Krishnanagar', 'WB_ND_01'),
  ('WB_MND_BOLPUR',       'Bolpur',       'WB_BI_01'),
  ('WB_MND_MIDNAPORE',    'Midnapore',    'WB_PM_01'),
  ('WB_MND_BERHAMPORE',   'Berhampore',   'WB_MU_01'),
  ('WB_MND_JALPAIGURI',   'Jalpaiguri',   'WB_JP_01')
ON CONFLICT (mandi_id) DO NOTHING;

-- 4. NDVI_BASELINE (9 rows) — calendar-day keyed (2026: Jan 1=1, Jan 15=15, Jan 30=30,
--    Feb 14=45, Mar 1=60, Mar 16=75, Mar 31=90) --------------------------------
INSERT INTO features.ndvi_baseline (district_id, day_of_year, multi_year_mean_ndvi) VALUES
  ('WB_HG_01',  1, 0.60),
  ('WB_HG_01', 15, 0.66),
  ('WB_HG_01', 30, 0.72),
  ('WB_HG_01', 45, 0.76),
  ('WB_HG_01', 60, 0.74),
  ('WB_HG_01', 75, 0.66),
  ('WB_HG_01', 90, 0.52),
  ('WB_BR_01', 30, 0.71),
  ('WB_BR_01', 60, 0.73)
ON CONFLICT (district_id, day_of_year) DO NOTHING;

-- 5. ENVIRONMENTAL_FEATURES (10 rows, hypertable) ------------------------------
-- ndvi_anomaly = mean_ndvi - baseline for that district/day_of_year (consistent with table 4).
-- Row 8 deliberately repeats Hooghly's 30-Jan weather under a second crop: same weather,
-- different season_start_date/cumulative_gdd (the crop_id-in-partition-key tradeoff).
INSERT INTO features.environmental_features
  (time, district_id, crop_id, season_start_date, precipitation_mm, t_min_c, t_max_c,
   mean_ndvi, ndvi_anomaly, cumulative_gdd, moisture_deficit) VALUES
  ('2026-01-01 00:00:00+05:30', 'WB_HG_01', 'POTATO_JYOTI',        '2025-11-20',  0.0, 10.8, 24.6, 0.62,  0.02,  482.5,  22.0),
  ('2026-01-15 00:00:00+05:30', 'WB_HG_01', 'POTATO_JYOTI',        '2025-11-20',  0.0,  9.4, 23.1, 0.65, -0.01,  611.0,  34.5),
  ('2026-01-30 00:00:00+05:30', 'WB_HG_01', 'POTATO_JYOTI',        '2025-11-20',  2.4, 11.6, 26.2, 0.73,  0.01,  762.8,  47.0),
  ('2026-02-14 00:00:00+05:30', 'WB_HG_01', 'POTATO_JYOTI',        '2025-11-20',  0.0, 14.2, 28.9, 0.79,  0.03,  934.6,  58.5),
  ('2026-03-01 00:00:00+05:30', 'WB_HG_01', 'POTATO_JYOTI',        '2025-11-20',  0.8, 17.3, 32.1, 0.70, -0.04, 1146.9,  71.0),
  ('2026-03-16 00:00:00+05:30', 'WB_HG_01', 'POTATO_JYOTI',        '2025-11-20',  5.6, 20.4, 34.6, 0.61, -0.05, 1401.3,  88.0),
  ('2026-03-31 00:00:00+05:30', 'WB_HG_01', 'POTATO_JYOTI',        '2025-11-20', 12.2, 23.1, 36.4, 0.47, -0.05, 1688.2, 109.5),
  ('2026-01-30 00:00:00+05:30', 'WB_HG_01', 'POTATO_CHANDRAMUKHI', '2025-11-25',  2.4, 11.6, 26.2, 0.73,  0.01,  702.9,  47.0),
  ('2026-01-30 00:00:00+05:30', 'WB_BR_01', 'POTATO_JYOTI',        '2025-11-25',  0.0, 10.9, 26.0, 0.70, -0.01,  706.4,  49.3),
  ('2026-03-01 00:00:00+05:30', 'WB_BR_01', 'POTATO_JYOTI',        '2025-11-25',  0.0, 16.8, 32.8, 0.68, -0.05, 1085.9,  74.8)
ON CONFLICT (time, district_id, crop_id) DO NOTHING;

-- 6. MARKET_FEATURES (10 rows, hypertable) -------------------------------------
-- Rows 1-6: a synthetic "glut building" story at Arambagh — arrivals climb, price slides,
-- short-window volatility and arrival ratio rise. district_id matches the mandi's district.
INSERT INTO features.market_features
  (time, mandi_id, district_id, crop_id, modal_price_inr, arrivals_mt,
   rolling_7d_ewma_std, rolling_14d_ewma_std, rolling_30d_ewma_std,
   arrival_to_volume_ratio, price_entropy, rolling_30d_momentum) VALUES
  ('2026-03-01 00:00:00+05:30', 'WB_MND_ARAMBAGH',     'WB_HG_01', 'POTATO_JYOTI',  1720.0, 118.5, 34.2,  41.8,  55.6, 0.96, 1.42,  0.012),
  ('2026-03-02 00:00:00+05:30', 'WB_MND_ARAMBAGH',     'WB_HG_01', 'POTATO_JYOTI',  1705.0, 126.0, 35.1,  42.0,  55.9, 1.02, 1.45,  0.006),
  ('2026-03-03 00:00:00+05:30', 'WB_MND_ARAMBAGH',     'WB_HG_01', 'POTATO_JYOTI',  1690.0, 139.4, 36.8,  42.7,  56.1, 1.13, 1.51, -0.002),
  ('2026-03-04 00:00:00+05:30', 'WB_MND_ARAMBAGH',     'WB_HG_01', 'POTATO_JYOTI',  1662.0, 152.8, 39.5,  43.9,  56.8, 1.24, 1.60, -0.014),
  ('2026-03-05 00:00:00+05:30', 'WB_MND_ARAMBAGH',     'WB_HG_01', 'POTATO_JYOTI',  1601.0, 171.2, 47.3,  46.5,  58.0, 1.39, 1.78, -0.032),
  ('2026-03-06 00:00:00+05:30', 'WB_MND_ARAMBAGH',     'WB_HG_01', 'POTATO_JYOTI',  1540.0, 190.6, 58.9,  50.2,  59.7, 1.55, 1.97, -0.058),
  ('2026-03-01 00:00:00+05:30', 'WB_MND_BARDHAMAN',    'WB_BR_01', 'POTATO_JYOTI',  1745.0, 210.0, 31.0,  38.5,  52.1, 0.98, 1.38,  0.015),
  ('2026-03-06 00:00:00+05:30', 'WB_MND_BARDHAMAN',    'WB_BR_01', 'POTATO_JYOTI',  1580.0, 268.5, 52.4,  47.0,  56.3, 1.42, 1.85, -0.049),
  ('2026-03-01 00:00:00+05:30', 'WB_MND_KRISHNANAGAR', 'WB_ND_01', 'ONION_RED',     2350.0,  64.0, 41.5,  55.0,  78.3, 1.05, 1.51,  0.021),
  ('2026-03-01 00:00:00+05:30', 'WB_MND_BERHAMPORE',   'WB_MU_01', 'TOMATO_HYBRID', 1820.0,  22.5, 88.0, 102.4, 131.7, 0.88, 1.66, -0.037)
ON CONFLICT (time, mandi_id, crop_id) DO NOTHING;

-- 7. YIELD_HISTORY (10 rows) — production = area x yield (kept arithmetically consistent) ---
INSERT INTO features.yield_history
  (district_id, crop_id, crop_year, season, area_hectares, production_tonnes, yield_tonnes_per_ha) VALUES
  ('WB_HG_01', 'POTATO_JYOTI',  2021, 'Rabi',   98500.0, 2895900.0, 29.4),
  ('WB_HG_01', 'POTATO_JYOTI',  2022, 'Rabi',  101200.0, 3046120.0, 30.1),
  ('WB_HG_01', 'POTATO_JYOTI',  2023, 'Rabi',   99800.0, 2864260.0, 28.7),
  ('WB_HG_01', 'POTATO_JYOTI',  2024, 'Rabi',  103600.0, 3232320.0, 31.2),
  ('WB_HG_01', 'POTATO_JYOTI',  2025, 'Rabi',  102900.0, 2870910.0, 27.9),
  ('WB_BR_01', 'POTATO_JYOTI',  2024, 'Rabi',  118300.0, 3643640.0, 30.8),
  ('WB_BR_01', 'POTATO_JYOTI',  2025, 'Rabi',  116700.0, 3314280.0, 28.4),
  ('WB_HG_01', 'PADDY_SWARNA',  2024, 'Kharif', 62400.0,  243360.0,  3.9),
  ('WB_BR_01', 'PADDY_SWARNA',  2024, 'Kharif',248000.0, 1041600.0,  4.2),
  ('WB_ND_01', 'WHEAT_HD2967',  2024, 'Rabi',   21800.0,   63220.0,  2.9)
ON CONFLICT (district_id, crop_id, crop_year, season) DO NOTHING;

-- 8. SERVING_PREDICTIONS (8 rows) — STUB outputs, hand-set, not model-generated --------
-- forecast_target_date holds the as-of/issue date (2026-03-06) for all three price horizons.
INSERT INTO serving.serving_predictions
  (model_version, forecast_target_date, district_id, crop_id,
   pred_yield_deviation_pct,
   pred_price_7d,  pred_price_7d_lower,  pred_price_7d_upper,
   pred_price_14d, pred_price_14d_lower, pred_price_14d_upper,
   pred_price_30d, pred_price_30d_lower, pred_price_30d_upper,
   risk_score_float, composite_risk_label,
   volatility_class_probs, volatility_regime, shap_attributions) VALUES
  ('stub-v0', '2026-03-06', 'WB_HG_01', 'POTATO_JYOTI',
   -5.2, 1495, 1448, 1541, 1470, 1392, 1548, 1525, 1398, 1652,
   0.74, 'High',
   '{"Stable": 0.09, "Crash-Glut": 0.76, "Spike-Shortage": 0.15}'::jsonb, 'Crash-Glut',
   '{"arrival_to_volume_ratio": 0.31, "rolling_7d_ewma_std": 0.18, "ndvi_anomaly": -0.07, "moisture_deficit": -0.04}'::jsonb),
  ('stub-v0', '2026-03-06', 'WB_HG_01', 'POTATO_CHANDRAMUKHI',
   -3.9, 1560, 1512, 1608, 1540, 1462, 1618, 1585, 1455, 1715,
   0.68, 'High',
   '{"Stable": 0.14, "Crash-Glut": 0.69, "Spike-Shortage": 0.17}'::jsonb, 'Crash-Glut',
   '{"arrival_to_volume_ratio": 0.27, "rolling_7d_ewma_std": 0.15, "ndvi_anomaly": -0.05, "moisture_deficit": -0.03}'::jsonb),
  ('stub-v0', '2026-03-06', 'WB_BR_01', 'POTATO_JYOTI',
   -6.8, 1552, 1500, 1604, 1520, 1437, 1603, 1560, 1425, 1695,
   0.71, 'High',
   '{"Stable": 0.11, "Crash-Glut": 0.73, "Spike-Shortage": 0.16}'::jsonb, 'Crash-Glut',
   '{"arrival_to_volume_ratio": 0.34, "rolling_7d_ewma_std": 0.16, "ndvi_anomaly": -0.08, "moisture_deficit": -0.05}'::jsonb),
  ('stub-v0', '2026-03-06', 'WB_BK_01', 'POTATO_JYOTI',
    1.4, 1640, 1590, 1690, 1628, 1548, 1708, 1650, 1520, 1780,
   0.42, 'Medium',
   '{"Stable": 0.52, "Crash-Glut": 0.31, "Spike-Shortage": 0.17}'::jsonb, 'Stable',
   '{"arrival_to_volume_ratio": 0.09, "rolling_7d_ewma_std": 0.06, "ndvi_anomaly": 0.02, "moisture_deficit": -0.01}'::jsonb),
  ('stub-v0', '2026-03-06', 'WB_ND_01', 'ONION_RED',
    0.6, 2385, 2310, 2460, 2410, 2300, 2520, 2520, 2350, 2690,
   0.29, 'Low',
   '{"Stable": 0.68, "Crash-Glut": 0.08, "Spike-Shortage": 0.24}'::jsonb, 'Stable',
   '{"rolling_30d_momentum": 0.07, "arrival_to_volume_ratio": -0.04, "rolling_14d_ewma_std": 0.03, "ndvi_anomaly": 0.01}'::jsonb),
  ('stub-v0', '2026-03-06', 'WB_MU_01', 'TOMATO_HYBRID',
   -2.1, 1790, 1690, 1890, 1740, 1610, 1870, 1830, 1640, 2020,
   0.55, 'Medium',
   '{"Stable": 0.38, "Crash-Glut": 0.44, "Spike-Shortage": 0.18}'::jsonb, 'Crash-Glut',
   '{"rolling_7d_ewma_std": 0.21, "price_entropy": 0.12, "arrival_to_volume_ratio": 0.10, "t_max_c": 0.05}'::jsonb),
  ('stub-v0', '2026-03-06', 'WB_PM_01', 'WHEAT_HD2967',
    2.7, 2560, 2530, 2590, 2565, 2520, 2610, 2580, 2510, 2650,
   0.18, 'Low',
   '{"Stable": 0.81, "Crash-Glut": 0.07, "Spike-Shortage": 0.12}'::jsonb, 'Stable',
   '{"rolling_30d_momentum": 0.04, "precipitation_mm": 0.02, "cumulative_gdd": 0.02, "ndvi_anomaly": 0.01}'::jsonb),
  ('stub-v0', '2026-03-06', 'WB_BI_01', 'POTATO_JYOTI',
   -1.7, 1610, 1560, 1660, 1598, 1520, 1676, 1640, 1510, 1770,
   0.47, 'Medium',
   '{"Stable": 0.47, "Crash-Glut": 0.36, "Spike-Shortage": 0.17}'::jsonb, 'Stable',
   '{"arrival_to_volume_ratio": 0.13, "rolling_7d_ewma_std": 0.09, "ndvi_anomaly": -0.03, "moisture_deficit": -0.02}'::jsonb);

COMMIT;

-- Row counts (paste this output back so the seed can be logged as confirmed) ---
SELECT 'features.location_dim' AS table_name, COUNT(*) AS row_count FROM features.location_dim
UNION ALL SELECT 'features.crop_dim',             COUNT(*) FROM features.crop_dim
UNION ALL SELECT 'features.mandi_dim',            COUNT(*) FROM features.mandi_dim
UNION ALL SELECT 'features.ndvi_baseline',        COUNT(*) FROM features.ndvi_baseline
UNION ALL SELECT 'features.environmental_features', COUNT(*) FROM features.environmental_features
UNION ALL SELECT 'features.market_features',      COUNT(*) FROM features.market_features
UNION ALL SELECT 'features.yield_history',        COUNT(*) FROM features.yield_history
UNION ALL SELECT 'serving.serving_predictions',   COUNT(*) FROM serving.serving_predictions;

-- =============================================================================
-- CLEAN RESET (run manually before re-seeding, or before real ingestion starts).
-- One statement, all tables together, so FK ordering is not an issue.
-- =============================================================================
-- TRUNCATE features.location_dim, features.crop_dim, features.mandi_dim,
--          features.ndvi_baseline, features.environmental_features,
--          features.market_features, features.yield_history,
--          serving.serving_predictions;