# AgroVision-ML — Unified Multi-Commodity Model Registry
## Master Cryptographic & Architectural Model Inventory

**Document Version:** 2.1.0  
**Effective Date:** 2026-11-16  
**Status:** Unified Integration Master  
**Registry Scope:** All Production, V2, and Baseline Commodities (Rice, Wheat, Potato, Tea)  
**Maintained By:** Principal ML Systems Architect & Production Operations  
**Source Manifests:**
- Rice: `member2-price-volatility/artifacts/model_handoff_manifest.json`
- Wheat V2: `member2-price-volatility/artifacts/wheat/v2/manifest.json`
- Potato V2: `member2-price-volatility/artifacts/potato/v2/manifest.json`
- Baseline SHA-256 Digest: `member2-price-volatility/artifacts/baseline_sha256_checksums.json`

---

## 1. Executive Summary & Cross-Commodity Governance

The AgroVision platform enforces strict architectural isolation between distinct commodities while exposing a unified prediction runtime for Member 3 (FastAPI service):
- **Rice:** Protected Production Reference system (48 features, XGBoost Regressor, Price-Normalized XGBoost Volatility + Isotonic Calibration).
- **Wheat:** Production V2 System (81 candidate features, CatBoost MAE Regressor with deadband $\epsilon=3.0$, CatBoost Volatility + Isotonic Calibration).
- **Potato:** Production V2 System (81 candidate features, XGBoost Pseudo-Huber Regressor with deadband $\epsilon=5.0$, XGBoost Volatility + Platt Sigmoid Calibration).
- **Tea:** Active Production Model (23 features, XGBoost Regressor, XGBoost Volatility).

### Core Governance Principles:
1. **Bit-for-Bit Immutability:** Pre-existing production binaries (Rice and V1 baselines) are cryptographically locked and must never be modified or overwritten.
2. **Unified Calling Convention:** All models are accessed through the standardized bridge `shared.predict_bridge.AgroVisionPredictor(commodity=...)` or `shared.predict_bridge.predict(...)`.
3. **Multi-Horizon Categorization:** Models are categorized into explicit operational tiers:
   - `V2 PRODUCTION CANDIDATE`: Primary 1-to-7-day modal price regressor and calibrated volatility classifier.
   - `V2 EXPERIMENTAL PASS`: Direct 14-day and 30-day horizon regressors (advisory tier with empirical prediction intervals).
   - `RETAIN V1`: Baseline models retained exclusively for audit benchmarks, regression testing, and rollback safety.
4. **No Cross-Commodity Fallback:** If an artifact for a requested commodity or version is missing or damaged, an explicit exception (`FileNotFoundError` or `ValueError`) is raised immediately. Silent substitution between commodities is prohibited.

---

## 2. Multi-Commodity Master Inventory & Cryptographic Registry

Below is the verified master registry of all model artifacts across all supported commodities with exact file sizes, verified SHA-256 digests, and formal operational tiers.

| Commodity | Version | Artifact Relative Path | Size (Bytes) | SHA-256 Checksum | Release Tier | Horizon | Benchmark Performance | Architecture & Details |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model.pkl` | 1,843,701 | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | **PRODUCTION** | 1–7d | Beats Persistence (-₹0.45/q MAE) | Primary price regressor (48 features, XGBoost) |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model_14d.pkl` | 1,572,754 | `1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab` | **OPT-IN** | 14d | Validated (MAE ₹59.87 vs ₹60.16) | 14-day direct calendar horizon price regressor |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model_30d.pkl` | 1,492,514 | `ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343` | **EXPERIMENTAL** | 30d | Advisory Only (Parity on flat mkts) | 30-day direct calendar horizon price regressor |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_normalized_model.pkl` | 1,291,739 | `dfc83d89ae3e93d9abf225b669b82fd9e1788972797353529680f7937b614a9f` | **PRODUCTION** | 1–7d | Beats Persistence (+8.04 pp Acc) | Price-normalized volatility classifier (35 feats, Isotonic) |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_model.pkl` | 1,762,176 | `f1acfa394e18b59cb2b25a318bbef7fe941626e8867ecccd80fc719622979253` | **RETAIN V1** | 1–7d | Baseline Reference | Uncalibrated raw volatility baseline |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_model_v2.pkl` | 361,371 | `8a21c26bb632f8e5910d0961d0c0b3a63c549f980c8b2822ec576b789d75446d` | **RETAIN V1** | 1–7d | Baseline Reference | Calibrated V2 volatility model |
| **Wheat** | `1.0.0` | `artifacts/wheat/wheat_price_model.pkl` | 804,457 | `9db711b7560494315cfff7775933d656b9378b89ad08152a4c4b9549787d7126` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost regressor (20 features) |
| **Wheat** | `1.0.0` | `artifacts/wheat/wheat_volatility_model.pkl` | 1,754,750 | `2f1ddebb9a46dd137f175980464e3f85004362e50fcbe7dacb09bbabb32785d4` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost classifier (19 features) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_model.pkl` | 460,901 | `f7d388b3bae30e158de2e5513cbc756b2722026faea18fa5da9e650d56b1450c` | **PRODUCTION** | 1–7d | Beats Persistence (-₹0.19/q MAE, -₹14.61/q RMSE) | Short-term price regressor (CatBoost MAE, $\epsilon=3.0$, 81 feats) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_model_v2.pkl` | 460,901 | `f7d388b3bae30e158de2e5513cbc756b2722026faea18fa5da9e650d56b1450c` | **PRODUCTION** | 1–7d | Beats Persistence (Alias) | Standardized alias for Wheat V2 price model |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_14d_model.pkl` | 461,814 | `5a98a46eca598de1ff59aee6b14290dce76127b6449bc871e2e11f80d5fe555a` | **EXPERIMENTAL** | 14d | Fails Persistence (+₹8.15/q Deficit) | 14-day direct CatBoost regressor (purged, advisory only) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_30d_model.pkl` | 461,318 | `93e3dbdc6c64a8298d1102b0154d56a7746cd27d1fc8ded1ac843577dd8f3892` | **EXPERIMENTAL** | 30d | Fails Persistence (+₹20.03/q Deficit) | 30-day direct CatBoost regressor (purged, advisory only) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_volatility_model.pkl` | 355,504 | `b1ffea4abff19e17f8065e8711e78262e77643249c3131d4f92431f9f9b38348` | **PRODUCTION** | 1–7d | Beats Baseline (+25.09 pp Acc, 73.4% HIGH Prec) | Calibrated CatBoost classifier (Isotonic) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_volatility_model_v2.pkl` | 355,504 | `b1ffea4abff19e17f8065e8711e78262e77643249c3131d4f92431f9f9b38348` | **PRODUCTION** | 1–7d | Beats Baseline (Alias) | Standardized alias for Wheat V2 volatility |
| **Potato** | `1.0.0` | `artifacts/potato/potato_price_model.pkl` | 770,525 | `5ef4f95667d1231ce39c487d1156c65a9eb3e2bf482641273e8279b7c8412b3a` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost regressor (20 features) |
| **Potato** | `1.0.0` | `artifacts/potato/potato_volatility_model.pkl` | 1,754,696 | `23340d66dd7357d637e7f592edb373eb3f09efbf14bf3d576fc57364e2ea4fc9` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost classifier (19 features) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_model.pkl` | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` | **PRODUCTION** | 1–7d | Matches Persistence MAE, Beats on RMSE (-₹1.05/q) | Short-term price regressor (XGB Pseudo-Huber, $\epsilon=5.0$, 81 feats) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_model_v2.pkl` | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` | **PRODUCTION** | 1–7d | Matches Persistence (Alias) | Standardized alias for Potato V2 price model |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_14d_model.pkl` | 460,590 | `6088ac503bbfae7789c7166ca850c69c11113bf98aa9f572cefad7fd94114f17` | **EXPERIMENTAL** | 14d | Fails Persistence (+₹23.33/q Deficit) | 14-day direct CatBoost regressor (purged, advisory only) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_30d_model.pkl` | 460,694 | `de169b0211e03d036265ba4c73fe2df166532dd900ed5452ea339d13e25e992f` | **EXPERIMENTAL** | 30d | Fails Persistence (+₹16.70/q Deficit) | 30-day direct CatBoost regressor (purged, advisory only) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_volatility_model.pkl` | 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` | **PRODUCTION** | 1–7d | Beats Baseline (+17.56 pp Acc, Balanced Acc 56.57%) | Calibrated XGBoost classifier (Platt Sigmoid) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_volatility_model_v2.pkl` | 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` | **PRODUCTION** | 1–7d | Beats Baseline (Alias) | Standardized alias for Potato V2 volatility |
| **Tea** | `1.0.0` | `artifacts/tea/tea_price_model.pkl` | 772,999 | `e437c37a6b22eb01eb1fa2888cf36c05eb7614e7a6f2ea351b6dbb20eb3057e9` | **PRODUCTION** | 1–7d | Production Baseline | Primary tea price regressor (23 features, XGBoost) |
| **Tea** | `1.0.0` | `artifacts/tea/tea_volatility_model.pkl` | 1,754,752 | `10bc93ea651ef8a1db576e330be4511dbb4e05d0d82847d95b542035eb4cf21e` | **PRODUCTION** | 1–7d | Production Baseline | Primary tea volatility classifier (19 features, XGBoost) |

---

## 3. Feature Configurations & Architecture Breakdown

### 3.1 Feature Vector Dimensions
- **Rice (Production):** 48 features (Modal price lags, rolling statistics, calendar seasonality, market encoding, arrivals volume). Volatility model uses 35 price-normalized features.
- **Wheat V2 (Production & Horizons):** 81 candidate features configured in `artifacts/wheat/v2/wheat_feature_config.json`. Includes multi-window EWMA, price differences, lag ratios, arrival dynamics, and day-of-week/month cyclical encoders.
- **Potato V2 (Production & Horizons):** 81 candidate features configured in `artifacts/potato/v2/potato_feature_config.json`. Aligned structure with commodity-specific variance scales.
- **Tea (Production):** 23 price features configured in `artifacts/tea/tea_feature_config.json` (`price_model["features"]`), 19 volatility features.

### 3.2 Deadband Filter Configurations ($\epsilon$)
To prevent micro-fluctuation noise amplification during stable market regimes, production V2 regressors utilize deadband filters:
- **Wheat V2:** Deadband epsilon $\epsilon = 3.0$ ₹/quintal. Absolute changes $\le ₹3.0$ are clamped to zero.
- **Potato V2:** Deadband epsilon $\epsilon = 5.0$ ₹/quintal. Absolute changes $\le ₹5.0$ are clamped to zero.
- **Rice:** Epsilon clamping evaluated in horizon governance.

---

## 4. Empirical Residual Distributions & Uncertainty Calibration

For all commodities and horizons, empirical holdout error distributions are serialized to enable non-parametric prediction intervals:

### 4.1 Artifact File Paths
- **Rice:** `artifacts/rice/rice_holdout_residuals_7d.json`, `rice_holdout_residuals_14d.json`, `rice_holdout_residuals_30d.json`
- **Wheat:** `artifacts/wheat/v2/wheat_residual_distributions.json`
- **Potato:** `artifacts/potato/v2/potato_residual_distributions.json`

### 4.2 Residual Distribution Summary Table

| Commodity | Horizon | Mean Error (₹/q) | Std Dev (₹/q) | p10 (₹/q) | p25 (₹/q) | p50 (Median) | p75 (₹/q) | p90 (₹/q) | Sample Count ($N$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rice** | 7d | +1.24 | 42.15 | -35.20 | -12.40 | +0.50 | +14.80 | +38.60 | 19,412 |
| **Rice** | 14d | +2.18 | 58.74 | -52.10 | -20.30 | +1.10 | +23.40 | +56.90 | 18,921 |
| **Rice** | 30d | +4.65 | 98.22 | -95.40 | -42.80 | +2.30 | +48.10 | +102.50 | 17,860 |
| **Wheat** | Short (1d) | -0.12 | 18.45 | -15.80 | -5.20 | 0.00 | +5.10 | +16.20 | 22,140 |
| **Wheat** | 14d | +3.42 | 44.12 | -41.50 | -16.80 | +1.80 | +21.40 | +47.60 | 21,805 |
| **Wheat** | 30d | +8.75 | 78.60 | -76.20 | -34.50 | +4.20 | +42.10 | +85.30 | 20,950 |
| **Potato** | Short (1d) | +0.45 | 32.80 | -28.40 | -9.60 | +0.10 | +10.20 | +29.50 | 24,620 |
| **Potato** | 14d | +6.80 | 72.40 | -68.30 | -29.10 | +3.50 | +35.80 | +77.40 | 23,980 |
| **Potato** | 30d | +14.20 | 124.90 | -119.50 | -55.20 | +7.80 | +64.10 | +136.20 | 22,810 |

---

## 5. Model Loading, Caching & Isolation Mechanics

The model loading pipeline implements:
1. **Commodity-Scoped Cache:** In-memory caching keyed by `(commodity, artifacts_dir, version, horizon_days)` ensures fast inference while preventing cross-talk.
2. **Explicit Dependency Resolution:** Artifact configs are verified against the requested commodity (`configuration["commodity"] == commodity`).
3. **No Cross-Commodity Substitution:** Calling `predict(..., commodity="wheat")` will strictly load Wheat artifacts. If Wheat artifacts are corrupted or missing, it will raise `FileNotFoundError` rather than silently defaulting to Rice or Potato.
4. **Backward Compatibility Preservation:** The legacy Rice pipeline (`inference.RiceInferencePipeline`) remains completely intact, preserving existing methods, class names, and return schemas while delivering unified outputs to new consumers.
