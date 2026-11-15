# AgroVision-ML — Potato V2 Production Audit & Deployment Certification

**Document Version:** 2.0.0
**Status:** Certified for Production Release (V2 PRODUCTION CANDIDATE)
**Commodity:** Potato (*Solanum tuberosum*)
**Module:** `member2-price-volatility`
**Certified By:** Senior ML Systems Architect & Production Readiness Auditor
**Effective Date:** 2026-11-15

---

## 1. Executive Summary & Production Certification Statement

The Potato V2 High-Performance Price Forecasting and Volatility Classification System is hereby **Certified for Production Release**.

All models, feature engineering pipelines, probability calibration layers, and metadata artifacts have undergone exhaustive empirical evaluation against the 51,243 chronologically quarantined holdout observations (`2021-07-30` to `2024-02-01`).

### Key Audit Achievements:
1. **Elimination of Calendar Overfitting:** Completely removes raw day-of-month memorization (which comprised 41.4% of V1 importance), substituting circular annual trigonometric seasonality ($\sin / \cos$).
2. **Beating Persistence on Variance & Tails:** Matches persistence MAE at ₹27.17 / quintal while lowering overall RMSE from ₹81.84 to **₹80.79 / quintal** and unseen mandi RMSE from ₹229.81 to **₹225.76 / quintal**.
3. **Recovery from Volatility Class Collapse:** Increases Volatility Balanced Accuracy by **+12.15 pp** (from 44.42% to **56.57%**), Macro F1 from 0.4298 to **0.5708**, and Matthews Correlation from 0.2138 to **0.3599** with an outstanding Platt Sigmoid Expected Calibration Error of **0.0186 (1.86%)**.
4. **Complete Production Isolation & Safety:** Zero modification of protected Rice models and Potato V1 baselines. All existing API contracts remain strictly backward-compatible.

---

## 2. Cryptographic Artifact Manifest

All Potato V2 production assets reside exclusively in `member2-price-volatility/artifacts/potato/v2/`. Below is the complete cryptographically verified inventory:

| Artifact Relative Path | Purpose / Model Family | Size (Bytes) | Verified SHA-256 Digest |
| :--- | :--- | :---: | :--- |
| `artifacts/potato/v2/potato_price_model.pkl` | XGBoost Pseudo-Huber Champion ($\epsilon=5.0$, 81 feats) | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` |
| `artifacts/potato/v2/potato_price_model_v2.pkl` | Standardized V2 model binary alias | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` |
| `artifacts/potato/v2/potato_price_14d_model.pkl`| Direct 14-day CatBoost Regressor | 460,590 | `6088ac503bbfae7789c7166ca850c69c11113bf98aa9f572cefad7fd94114f17` |
| `artifacts/potato/v2/potato_price_30d_model.pkl`| Direct 30-day CatBoost Regressor | 460,694 | `de169b0211e03d036265ba4c73fe2df166532dd900ed5452ea339d13e25e992f` |
| `artifacts/potato/v2/potato_volatility_model.pkl`| Sigmoid Calibrated XGBoost Classifier | 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` |
| `artifacts/potato/v2/potato_volatility_model_v2.pkl`| Standardized V2 volatility binary alias | 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` |
| `artifacts/potato/v2/potato_feature_config.json` | 81-feature runtime definition | 4,621 | `f5cf139d37ebc8988a2257da72bbc3422abdcc6fe0562b12ed2ae73a096d4b53` |
| `artifacts/potato/v2/potato_price_model_v2_config.json` | Standardized V2 price config alias | 4,621 | `f5cf139d37ebc8988a2257da72bbc3422abdcc6fe0562b12ed2ae73a096d4b53` |
| `artifacts/potato/v2/potato_volatility_config.json`| Volatility percentage thresholds & config | 444 | `18fcf076a400adf5f44bb15303a4f056102900951754c18c3cdc61cd77642ddb` |
| `artifacts/potato/v2/potato_volatility_model_v2_config.json`| Standardized V2 volatility config alias | 444 | `18fcf076a400adf5f44bb15303a4f056102900951754c18c3cdc61cd77642ddb` |
| `artifacts/potato/v2/potato_residual_distributions.json`| Non-parametric error quantiles (p05-p95) | 1,416 | `5d8595ec7dd51932f71ce13681870afd55d78a519011bd7288990bd264c8d877` |
| `artifacts/potato/v2/potato_v2_metrics.json` | Full holdout & sub-population metrics | 5,958 | `69dbbe597e26568206e62d89f7d9512c0fa872bd6fc87d1c769984028ffea11a` |
| `artifacts/potato/v2/manifest.json` | Cryptographic directory registry | 2,026 | `d8f7aad6b8a1d5d6c3d30e754bdd804dcff6f8db22e899f3b930f951fbad2eef` |

---

## 3. Production Safety & Baseline Integrity Audit

SHA-256 verification confirms that zero baseline files were modified during Potato V2 development:
- **Rice Protected Binaries & Configs:** 23 files verified bit-for-bit identical (`0` altered, `0` deleted).
- **Potato V1 Baselines:**
  - `potato_price_model.pkl`: `5ef4f95667d1231ce39c487d1156c65a9eb3e2bf482641273e8279b7c8412b3a` (**PASS — Untouched**)
  - `potato_volatility_model.pkl`: `23340d66dd7357d637e7f592edb373eb3f09efbf14bf3d576fc57364e2ea4fc9` (**PASS — Untouched**)
- **Baseline Preservation Status:** **100% BIT-FOR-BIT IDENTICAL**.

---

## 4. Section 17 Standard Audit Compliance Summary

| Section 17 Item | Verification Status | Evidentiary Findings |
| :--- | :---: | :--- |
| **1. Dataset Summary** | **VERIFIED** | 265,047 clean records, 88 mandis, 13 varieties, 2002–2024 time span. Split date: `2021-07-29`. |
| **2. Original V1 Baseline** | **VERIFIED** | V1 XGBoost: MAE ₹27.57, RMSE ₹81.84, R² 0.9740. Volatility Acc: 50.88%, F1: 0.4298, BalAcc: 44.42%. |
| **3. Feature Engineering** | **VERIFIED** | 81 candidate features. Day-of-month removed to eliminate 41.4% calendar bias. Added circular sin/cos. |
| **4. Model Experiments** | **VERIFIED** | Compared XGBoost (MSE, MAE, Pseudo-Huber), CatBoost (MAE), HistGB, Random Forest. |
| **5. Final Hyperparameters** | **VERIFIED** | XGBRegressor (n_est=300, max_d=5, lr=0.04, loss=reg:pseudohubererror, deadband=₹5.0). |
| **6. Walk-Forward Validation** | **VERIFIED** | 3 expanding folds (Fold 1: 81,840/40,922; Fold 2: 122,762/40,898; Fold 3: 163,660/40,943). XGBoost Pseudo-Huber achieves CV RMSE ₹87.92 vs P ₹88.76. |
| **7. Final Holdout Results** | **VERIFIED** | Exact metrics from `potato_v2_metrics.json`: MAE ₹27.17, RMSE ₹80.79, R² 0.9746, MAPE 2.06%. |
| **8. Regression Metrics Tables**| **VERIFIED** | Complete multi-subpopulation breakdowns for Overall, Seen, Unseen, High-Volume, and Movement days. |
| **9. Classification Tables** | **VERIFIED** | Volatility metrics: Acc 59.35%, BalAcc 56.57%, Macro F1 0.5708, MCC 0.3599, PR-AUC 0.6285. |
| **10. Confusion Matrices** | **VERIFIED** | 3x3 holdout confusion matrix verified ($N=51,228$): LOW 4,389/3,277/2,915, MED 1,839/13,256/6,315, HIGH 1,837/4,642/12,758. |
| **11. Persistence Comparison** | **VERIFIED** | Matches persistence on MAE (₹27.17), beats persistence on RMSE (-₹1.05/q overall, -₹4.05/q unseen). |
| **12. Market Generalization** | **VERIFIED** | Evaluated on 6 unseen mandi groups ($N=679$). V2 MAE ₹31.24 vs P ₹31.56, RMSE ₹225.76 vs P ₹229.81. |
| **13. Leakage Tests Summary** | **VERIFIED** | Automated tests confirm zero lookahead, group isolation, and horizon boundary purging. |
| **14. Model Selection Rationale**| **VERIFIED** | Short-Term Price & Volatility: **V2 PRODUCTION CANDIDATE**. 14d/30d: **V2 EXPERIMENTAL PASS**. |
| **15. Limitations** | **VERIFIED** | Horticultural perishability and storage release shocks; 14d/30d long horizon macro drift. |
| **16. Final Recommendations** | **VERIFIED** | Expose empirical prediction intervals [p10, p90]; deploy via `AgroVisionPredictor`. |

---

## 5. Sub-Population Performance Scorecard

Evaluated across four distinct market slices on the holdout test set:

| Evaluation Slice | Sample Count ($N$) | Price MAE (₹/q) | Persist MAE (₹/q) | Price RMSE (₹/q) | Persist RMSE (₹/q) | Volatility Acc (%) | Volatility Balanced Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall Holdout** | 51,243 / 51,228 | **₹27.17** | ₹27.17 | **₹80.79** | ₹81.84 | **59.35%** | **56.57%** |
| **Known (Seen) Mandis**| 50,564 / 50,569 | **₹27.12** | ₹27.11 | **₹77.00** | ₹77.96 | **59.35%** | **56.44%** |
| **Unseen Mandi Groups** | 679 / 659 | **₹31.24** | ₹31.56 | **₹225.76** | ₹229.81 | **59.33%** | **58.33%** |
| **High-Volume Mandis** | 13,202 / 13,169 | ₹24.42 | **₹24.41** | **₹58.50** | ₹58.59 | **58.09%** | **57.70%** |

---

## 6. Runtime Latency & Memory Footprint

- **Single Prediction Latency:**
  - Price Regressor (`predict_price`): **1.85 ms**
  - Volatility Classifier (`predict_volatility`): **2.65 ms**
- **Batch Processing Throughput (1,000 observations):** 36.4 ms total ($>27,000$ records/sec).
- **RAM Footprint:** Combined models occupy less than **3.5 MB** in process memory.

---

## 7. Final Deployment Verdict

### Certification Decision: **RELEASE TO PRODUCTION (V2 PRODUCTION CANDIDATE)**

Potato V2 is fully validated, bit-for-bit reproducible, mathematically sound, and ready for end-user serving in AgroVision. Downstream microservices may immediately bind to `AgroVisionPredictor(commodity="potato")`.
