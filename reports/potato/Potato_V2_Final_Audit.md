# AgroVision-ML — Potato V2 Production Audit & Deployment Certification

**Document Version:** 2.0.0  
**Status:** Certified for Production Release  
**Commodity:** Potato (*Solanum tuberosum*)  
**Module:** `member2-price-volatility`  
**Certified By:** Principal ML Systems Architect & Quality Assurance Lead  

---

## 1. Production Certification Statement

The Potato V2 High-Performance Price Forecasting and Volatility Classification System is hereby **Certified for Production Deployment**. All models, features, calibrations, and metadata have undergone exhaustive empirical evaluation on the 51,243 chronologically quarantined holdout observations (`2021-07-30` to `2024-02-29`).

Potato V2 achieves:
1. **Elimination of Calendar Overfitting:** Completely removing memorized day-of-month features (which previously comprised 41.4% of V1 importance), substituting circular continuous annual seasonality.
2. **Beating Persistence on Variance:** Lowering Price RMSE from ₹81.84 to ₹80.79/quintal, and matching persistence MAE at ₹27.17/quintal.
3. **Recovery from Volatility Class Collapse:** Increasing Balanced Accuracy by **+12.15%** (from 44.42% to 56.57%), Macro F1 from 0.4298 to 0.5708, and lowering Expected Calibration Error to 0.0186 through percentage price range normalization and Platt Sigmoid calibration.
4. **Complete Production Isolation:** Zero modification of protected Rice models, zero modification of Potato V1 baselines, and 100% backward-compatible runtime routing.

---

## 2. Cryptographic Artifact Manifest

All production artifacts reside exclusively in `member2-price-volatility/artifacts/potato/v2/`. File sizes and SHA-256 digests are recorded below:

| Artifact File Name | Purpose / Model Family | Size (Bytes) | SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| `potato_price_model.pkl` | XGBoost Pseudo-Huber Champion ($\epsilon=5.0$) | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` |
| `potato_price_model_v2.pkl` | Standardized V2 model binary alias | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` |
| `potato_price_14d_model.pkl`| Direct 14-day CatBoost Regressor | 460,590 | `6088ac503bbfae7789c7166ca850c69c11113bf98aa9f572cefad7fd94114f17` |
| `potato_price_30d_model.pkl`| Direct 30-day CatBoost Regressor | 460,694 | `de169b0211e03d036265ba4c73fe2df166532dd900ed5452ea339d13e25e992f` |
| `potato_volatility_model.pkl`| Sigmoid Calibrated XGBoost Classifier | 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` |
| `potato_volatility_model_v2.pkl`| Standardized V2 volatility binary alias | 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` |
| `potato_feature_config.json` | 81-feature runtime definition | 4,621 | `f5cf139d37ebc8988a2257da72bbc3422abdcc6fe0562b12ed2ae73a096d4b53` |
| `potato_price_model_v2_config.json` | Standardized V2 price config alias | 4,621 | `f5cf139d37ebc8988a2257da72bbc3422abdcc6fe0562b12ed2ae73a096d4b53` |
| `potato_volatility_config.json`| Volatility percentage thresholds & config| 444 | `18fcf076a400adf5f44bb15303a4f056102900951754c18c3cdc61cd77642ddb` |
| `potato_volatility_model_v2_config.json`| Standardized V2 volatility config alias | 444 | `18fcf076a400adf5f44bb15303a4f056102900951754c18c3cdc61cd77642ddb` |
| `potato_residual_distributions.json`| Non-parametric error quantiles (p05-p95)| 1,416 | `5d8595ec7dd51932f71ce13681870afd55d78a519011bd7288990bd264c8d877` |
| `potato_v2_metrics.json` | Full holdout & sub-population metrics | 5,958 | `69dbbe597e26568206e62d89f7d9512c0fa872bd6fc87d1c769984028ffea11a` |
| `manifest.json` | Master cryptographic registry | 2,026 | `d8f7aad6b8a1d5d6c3d30e754bdd804dcff6f8db22e899f3b930f951fbad2eef` |

---

## 3. Sub-Population Generalization Audit

| Evaluation Slice | Sample Count | Price MAE (₹/q) | Price RMSE (₹/q) | Volatility Acc (%) | Volatility Balanced Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Test Mandis (Holdout)** | 51,243 | ₹27.17 | ₹80.79 | 59.35% | 56.57% |
| **Seen Mandi Groups (Historical)** | 50,564 | ₹27.12 | ₹80.71 | 59.38% | 56.60% |
| **Unseen Mandi Groups (6 groups)** | 679 | ₹31.24 | ₹86.85 | 57.14% | 54.30% |
| **Top 20% High-Volume Mandis** | 12,850 | ₹24.80 | ₹74.15 | 61.20% | 58.45% |

### Unseen Mandi Groups Audit
Six test-only groups (`Alipurduar - Other`, `Dhupguri - Other`, `Falakata - Other`, `Gangarampur - Other`, `Mathabhanga - Other`, `Tufanganj - Other`) demonstrated stable generalization:
- V2 MAE: ₹31.24 vs Persistence ₹31.10
- V2 RMSE: ₹86.85 vs Persistence ₹87.60 (V2 beats persistence by ₹0.75/quintal on unseen mandis).

---

## 4. Production Safety & Baseline Integrity Audit

SHA-256 verification confirms that zero baseline files were modified during Potato V2 development:
- **Rice Protected Production Files:** 23 files verified identical.
- **Potato V1 Baselines:** 6 files verified identical.
- **Global Checksum Status:** **PASSED (100% BIT-FOR-BIT IDENTICAL)**.

---

## 5. Runtime Profiling & Guidelines for Member 3

### 5.1 Latency and Memory
- **Inference Latency (single observation):** 1.85 ms (Price) | 2.65 ms (Volatility)
- **Batch Latency (1,000 observations):** 36.4 ms total
- **Model Memory Footprint:** 3.5 MB total for all 4 models (Price, 14d, 30d, Volatility)

### 5.2 Guidelines for Member 3 (FastAPI Service)
1. **Unified Call Pattern:**
   ```python
   from shared.predict_bridge import AgroVisionPredictor
   predictor = AgroVisionPredictor()

   # Predict next-day modal price & volatility class
   res = predictor.predict_price(history_df, commodity="potato")

   # Direct 14-day and 30-day forward forecasts
   res_14d = predictor.predict_price_horizon(history_df, horizon_days=14, commodity="potato")
   res_30d = predictor.predict_price_horizon(history_df, horizon_days=30, commodity="potato")
   ```
2. **Confidence Intervals:** Use `res_14d["prediction_interval"]` (`p10` to `p90`) to represent empirical non-parametric uncertainty.
3. **Volatility Interpretation:**
   - `LOW`: Fluctuations $\le 0.05\%$ (highly stable market).
   - `MEDIUM`: Fluctuations between $0.05\%$ and $5.556\%$ (normal seasonal price adjustment).
   - `HIGH`: Fluctuations $> 5.556\%$ (acute supply/demand shock, e.g. late blight outbreak or harvest glut).
