# AgroVision-ML — Multi-Commodity Production Model Registry
## Master Cryptographic & Architectural Model Inventory

**Document Version:** 2.0.0
**Effective Date:** 2026-11-15
**Registry Scope:** All Production, V2, and Baseline Commodities (Rice, Wheat, Potato, Tea)
**Maintained By:** Principal ML Systems Architect & Production Operations
**Source Manifests:**
- Rice: `member2-price-volatility/artifacts/model_handoff_manifest.json`
- Wheat: `member2-price-volatility/artifacts/wheat/v2/manifest.json`
- Potato: `member2-price-volatility/artifacts/potato/v2/manifest.json`
- Baselines: `member2-price-volatility/artifacts/baseline_sha256_checksums.json`

---

## 1. Executive Summary & Cross-Commodity Governance

The AgroVision platform enforces strict architectural separation between distinct commodities while exposing a unified prediction runtime for Member 3 (FastAPI service):
- **Rice:** Protected Production Reference system (48 features, XGBoost Regressor, XGBoost Volatility + Isotonic Calibration).
- **Wheat:** Production V2 System (81 features, CatBoost MAE Regressor with deadband, CatBoost Volatility + Isotonic Calibration).
- **Potato:** Production V2 System (81 features, XGBoost Pseudo-Huber Regressor with deadband, XGBoost Volatility + Sigmoid Calibration).
- **Tea:** Active Production Model (20 features, XGBoost Regressor, XGBoost Volatility).

### Core Governance Principles:
1. **Bit-for-Bit Immutability:** Pre-existing production binaries (Rice and V1 baselines) are cryptographically locked and must never be modified or overwritten.
2. **Unified Calling Convention:** All models are accessed through the standardized bridge `shared.predict_bridge.AgroVisionPredictor(commodity=...)`.
3. **Multi-Horizon Categorization:** Models are categorized into explicit operational tiers:
   - `V2 PRODUCTION CANDIDATE`: Primary 1-to-7-day modal price regressor and calibrated volatility classifier.
   - `V2 EXPERIMENTAL PASS`: Direct 14-day and 30-day horizon regressors (advisory tier with empirical prediction intervals).
   - `RETAIN V1`: Baseline models retained exclusively for audit benchmarks, regression testing, and rollback safety.

---

## 2. Multi-Commodity Model Registry & Cryptographic Inventory

Below is the master table of all commodity $\times$ version model artifacts with verified SHA-256 digests and formal release verdicts:

| Commodity | Version | Artifact Relative Path | Size (Bytes) | SHA-256 Checksum | Final Release Verdict | Persistence Benchmark Status | Description & Architecture |
| :--- | :---: | :--- | :---: | :--- | :---: | :--- | :--- |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model.pkl` | 1,843,701 | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | **V2 PRODUCTION CANDIDATE** | Beats Persistence (-₹0.45/q MAE) | Primary production price regressor (48 features, XGBoost) |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model_14d.pkl` | 1,572,754 | `1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab` | **V2 EXPERIMENTAL PASS** | Advisory Only (Horizon Drift) | 14-day direct calendar horizon price regressor |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model_30d.pkl` | 1,492,514 | `ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343` | **V2 EXPERIMENTAL PASS** | Advisory Only (Horizon Drift) | 30-day direct calendar horizon price regressor |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_model.pkl` | 1,762,176 | `f1acfa394e18b59cb2b25a318bbef7fe941626e8867ecccd80fc719622979253` | **RETAIN V1** | Baseline Reference | Uncalibrated raw volatility baseline |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_model_v2.pkl` | 361,371 | `8a21c26bb632f8e5910d0961d0c0b3a63c549f980c8b2822ec576b789d75446d` | **RETAIN V1** | Baseline Reference | Calibrated V2 volatility model |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_normalized_model.pkl` | 1,291,739 | `dfc83d89ae3e93d9abf225b669b82fd9e1788972797353529680f7937b614a9f` | **V2 PRODUCTION CANDIDATE** | Beats Persistence (+8.04 pp Acc) | Price-normalized volatility classifier (35 feats) |
| **Wheat** | `1.0.0` | `artifacts/wheat/wheat_price_model.pkl` | 804,457 | `9db711b7560494315cfff7775933d656b9378b89ad08152a4c4b9549787d7126` | **RETAIN V1** | Baseline Reference | Historical V1 XGBoost regressor (20 features) |
| **Wheat** | `1.0.0` | `artifacts/wheat/wheat_volatility_model.pkl` | 1,754,750 | `2f1ddebb9a46dd137f175980464e3f85004362e50fcbe7dacb09bbabb32785d4` | **RETAIN V1** | Baseline Reference | Historical V1 XGBoost classifier (19 features) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_model.pkl` | 460,901 | `f7d388b3bae30e158de2e5513cbc756b2722026faea18fa5da9e650d56b1450c` | **V2 PRODUCTION CANDIDATE**| Beats Persistence (-₹0.19/q MAE, -₹14.61/q RMSE) | Short-term price regressor (CatBoost MAE, $\epsilon=3.0$) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_model_v2.pkl` | 460,901 | `f7d388b3bae30e158de2e5513cbc756b2722026faea18fa5da9e650d56b1450c` | **V2 PRODUCTION CANDIDATE**| Beats Persistence (Alias) | Standardized alias for Wheat V2 price model |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_14d_model.pkl` | 461,814 | `5a98a46eca598de1ff59aee6b14290dce76127b6449bc871e2e11f80d5fe555a` | **V2 EXPERIMENTAL PASS** | **FAILS Persistence on MAE** (+₹8.15/q Deficit) | 14-day direct CatBoost regressor (purged) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_30d_model.pkl` | 461,318 | `93e3dbdc6c64a8298d1102b0154d56a7746cd27d1fc8ded1ac843577dd8f3892` | **V2 EXPERIMENTAL PASS** | **FAILS Persistence across MAE/RMSE/R²** (+₹20.03/q Deficit) | 30-day direct CatBoost regressor (purged) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_volatility_model.pkl` | 355,504 | `b1ffea4abff19e17f8065e8711e78262e77643249c3131d4f92431f9f9b38348` | **V2 PRODUCTION CANDIDATE**| Beats Baseline (+25.09 pp Acc, 73.4% HIGH Prec) | Calibrated CatBoost classifier (Isotonic) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_volatility_model_v2.pkl` | 355,504 | `b1ffea4abff19e17f8065e8711e78262e77643249c3131d4f92431f9f9b38348` | **V2 PRODUCTION CANDIDATE**| Beats Baseline (Alias) | Standardized alias for Wheat V2 volatility |
| **Potato** | `1.0.0` | `artifacts/potato/potato_price_model.pkl` | 770,525 | `5ef4f95667d1231ce39c487d1156c65a9eb3e2bf482641273e8279b7c8412b3a` | **RETAIN V1** | Baseline Reference | Historical V1 XGBoost regressor (20 features) |
| **Potato** | `1.0.0` | `artifacts/potato/potato_volatility_model.pkl` | 1,754,696 | `23340d66dd7357d637e7f592edb373eb3f09efbf14bf3d576fc57364e2ea4fc9` | **RETAIN V1** | Baseline Reference | Historical V1 XGBoost classifier (19 features) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_model.pkl` | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` | **V2 PRODUCTION CANDIDATE**| Matches Persistence MAE, Beats on RMSE (-₹1.05/q, -₹4.05/q Unseen) | Short-term price regressor (XGB Pseudo-Huber, $\epsilon=5.0$)|
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_model_v2.pkl` | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` | **V2 PRODUCTION CANDIDATE**| Matches Persistence (Alias) | Standardized alias for Potato V2 price model |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_14d_model.pkl`| 460,590 | `6088ac503bbfae7789c7166ca850c69c11113bf98aa9f572cefad7fd94114f17` | **V2 EXPERIMENTAL PASS** | **FAILS Persistence on MAE/RMSE** (+₹23.33/q Deficit) | 14-day direct CatBoost regressor (purged) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_30d_model.pkl`| 460,694 | `de169b0211e03d036265ba4c73fe2df166532dd900ed5452ea339d13e25e992f` | **V2 EXPERIMENTAL PASS** | **FAILS Persistence across MAE/RMSE/R²** (+₹16.70/q Deficit) | 30-day direct CatBoost regressor (purged) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_volatility_model.pkl`| 2,031,230| `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` | **V2 PRODUCTION CANDIDATE**| Beats Baseline (+17.56 pp Acc, Balanced Acc 56.57%) | Calibrated XGBoost classifier (Platt Sigmoid) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_volatility_model_v2.pkl`| 2,031,230| `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` | **V2 PRODUCTION CANDIDATE**| Beats Baseline (Alias) | Standardized alias for Potato V2 volatility |
| **Tea** | `1.0.0` | `artifacts/tea/tea_price_model.pkl` | 594,157 | `2becae635c299352f0c5b772c663527c3904eba6e8e197eb061a364546b5aefe` | **RETAIN V1** | Production Baseline | Specialized tea auction modal price regressor |
| **Tea** | `1.0.0` | `artifacts/tea/tea_volatility_model.pkl` | 1,685,122 | `9cc8b93f405ccccb7caaf9159061e16be2d28bb49c8f8fa4f6cc77c5b3bab565` | **RETAIN V1** | Production Baseline | Specialized tea auction volatility classifier |

---

## 3. Rice vs. Wheat vs. Potato Executive Comparison

| Evaluation Dimension | Rice (*Production Reference*) | Wheat (*V2 Production Candidate*) | Potato (*V2 Production Candidate*) |
| :--- | :--- | :--- | :--- |
| **Crop Classification** | Kharif Cereal Grain | Rabi Cereal Grain | Perishable Horticultural Cash Tuber |
| **Market Scope** | 52 mandis, 28 varieties | 38 mandis, 16 varieties | 88 mandis, 13 varieties |
| **Clean Observation Rows** | 102,084 | 64,364 | 265,047 |
| **Holdout Sample Size ($N$)** | 20,465 | 12,281 | 51,243 |
| **Holdout MAE (₹/q)** | **₹18.98** (Persistence: ₹19.43) | **₹18.39** (Persistence: ₹18.58) | **₹27.17** (Persistence: ₹27.17) |
| **Holdout RMSE (₹/q)** | **₹85.45** (Persistence: ₹87.20) | **₹74.73** (Persistence: ₹89.34) | **₹80.79** (Persistence: ₹81.84) |
| **Holdout $R^2$ Variance** | **0.9805** | **0.9576** | **0.9746** |
| **Mean Abs Pct Error (MAPE)**| **0.54%** | **0.77%** | **2.06%** |
| **Movement Days MAE / RMSE** | ₹49.12 / ₹138.40 | **₹48.16 / ₹124.11** | **₹60.82 / ₹121.02** |
| **Volatility Accuracy** | **68.25%** | **64.43%** | **59.35%** |
| **Volatility Balanced Acc** | **63.20%** | **63.42%** | **56.57%** |
| **Volatility Macro F1** | **0.6247** | **0.6303** | **0.5708** |
| **Expected Calib Error (ECE)**| **0.0380** | **0.0571** | **0.0186** |
| **Beats Price Persistence?** | **YES** (-₹0.45/q MAE, -₹1.75/q RMSE) | **YES** (-₹0.19/q MAE, -₹14.61/q RMSE) | **YES** (-₹1.05/q RMSE, -₹4.05/q unseen) |
| **Primary Structural Challenge**| Moderate price stickiness (68.2%) | Severe price stickiness (74.8%) | Calendar overfitting (41.4% in V1) & non-stationary inflation |
| **Architectural Solution** | 48 features + Bias Correction | 81 features + CatBoost MAE + Deadband | 81 features + Pseudo-Huber + Circular Seasons |
| **Production Release Status** | **V2 PRODUCTION CANDIDATE** (Frozen Reference) | **V2 PRODUCTION CANDIDATE** | **V2 PRODUCTION CANDIDATE** |

---

## 4. Release Governance & Deployment Verification Checklist

- [x] All 81 point-in-time features verified leak-free.
- [x] All cryptographic SHA-256 hashes verified on disk.
- [x] Protected Rice production binaries (23 files) confirmed untouched and bit-for-bit identical.
- [x] Baseline V1 models preserved for rollback safety.
- [x] 186/186 pytest tests in `member2-price-volatility/tests/` passing cleanly.
- [x] Direct calendar horizon models (14d/30d) equipped with empirical residual quantiles (`p05` through `p95`).
- [x] Shared bridge (`AgroVisionPredictor`) provides uniform multi-commodity runtime interface.
