# AgroVision-ML — Final Unified Model Release Registry
## Master Cryptographic & Architectural Model Inventory Across Member 1 and Member 2

**Document ID:** AV-M4-REGISTRY-006  
**Version:** 3.0.0  
**Effective Date:** 2026-11-17  
**Classification:** Production Engineering & Independent Systems Audit  
**Maintained By:** Principal ML Systems Architect & Independent Release Auditor  
**Registry Scope:** Complete Platform Inventory (Member 1 Yield Forecasting & Member 2 Market Price/Volatility Forecasting)  
**Target Environment:** Staging to Production Transition  

---

## 1. Executive Summary & Master Governance Framework

This document represents the definitive, unified cryptographic release registry for the **AgroVision-ML** predictive platform. It establishes the master inventory of all production, candidate, opt-in, experimental, and retained baseline models across:
1. **Member 1 (`member1-yield-model`):** District-level seasonal Rice crop yield forecasting across West Bengal (*Aus*, *Aman*, *Boro*).
2. **Member 2 (`member2-price-volatility`):** Multi-commodity wholesale mandi price and market volatility forecasting across 4 commodities (**Rice**, **Wheat**, **Potato**, and **Tea**).

### Master Architectural Governance Principles:
1. **Cryptographic Immutability:** All production binaries are protected by verified SHA-256 digests. Zero retraining, parameter modification, or silent artifact swapping is permitted without formal re-audit and registry increment.
2. **Strict Commodity Domain Boundaries:**
   - **Member 1 specializes exclusively in West Bengal Rice Yield.**
   - **Wheat, Potato, and Tea yield predictions are strictly UNSUPPORTED.** The unified bridge explicitly rejects non-rice yield prediction requests with `ValueError` or `NotImplementedError`. Zero data fabrication or cross-commodity substitution is permitted.
3. **Operational Release Tiers:**
   - `PRODUCTION`: Primary operational model powering public APIs and default workflows.
   - `SHADOW CANDIDATE`: Monitored candidate running in shadow mode; cannot serve live public decisions until promotion criteria are met.
   - `OPT-IN`: Validated advanced model available via explicit client opt-in flags.
   - `EXPERIMENTAL`: Research/advisory models with empirical prediction intervals, disabled by default in public endpoints.
   - `RETAIN V1`: Certified baseline models retained exclusively for audit benchmarks, non-regression verification, and disaster rollback.
4. **Universal Calling Bridge:** All inference across Member 1 and Member 2 is mediated by `shared.predict_bridge.AgroVisionPredictor` or top-level functions `predict()`, `predict_price()`, `predict_yield()`.

---

## 2. Member 1: Rice Crop Yield Model Inventory

### 2.1 Scope & Agronomic Specification
- **Commodity:** Rice (*Oryza sativa*)
- **Geographic Domain:** 23 administrative districts of West Bengal, India.
- **Seasonal Coverage:**
  - Autumn (*Aus*): Harvest Oct–Nov, weather cutoff July 31.
  - Winter (*Aman*): Harvest Nov–Jan, weather cutoff September 30.
  - Summer (*Boro*): Harvest Mar–May, weather cutoff February 28.
- **Target Formulation:** Residual-target prediction ($\Delta = \text{Yield}_t - \text{Yield}_{t-1}$), where $\hat{y}_t = \text{Yield}_{t-1} + \hat{\Delta}_t$.
- **Feature Vector:** 20 features (cutoff-enforced cumulative precipitation, mean/max/min temperatures, relative humidity, solar radiation, historical yield lags, district trend indicators).
- **Default Production Routing Policy:** **Policy 4** (Autumn and Winter routed to Delta Ridge; Summer routed to persistence baseline $\text{Yield}_{t-1}$).
- **Shadow Candidate Routing Policy:** **Candidate Pure Delta Ridge** (All three seasons routed to Delta Ridge).

### 2.2 Member 1 Artifact Inventory Table

| Artifact Path | File Size (Bytes) | Certified SHA-256 Checksum | Release Tier | Target / Horizon | Out-of-Time Performance ($N=192$) | Architecture & Details |
| :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| `models/m1_improved_delta_ridge.pkl` | 6,018 | `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56` | **PRODUCTION** | Seasonal Yield (Policy 4) | MAE: $261.84\text{ kg/ha}$ ($\mathbf{+22.33\text{ kg/ha}}$ vs Base), RMSE: $347.16$, $R^2: 0.6358$ | Regularized L2 Delta Ridge regressor ($\alpha=1.0$), 20 features. Frozen production artifact. |
| `models/m1_improved_delta_ridge.pkl` | 6,018 | `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56` | **SHADOW CANDIDATE** | Seasonal Yield (Pure Delta) | MAE: $255.05\text{ kg/ha}$ ($\mathbf{+29.12\text{ kg/ha}}$ vs Base), RMSE: $340.71$, $R^2: 0.6492$ | Pure Delta Ridge without Boro persistence routing. Outperformed Policy 4 on 2020–2022 holdout. Under prospective shadow evaluation. |
| `models/m1_best_ridge.pkl` | 5,082 | `00ff14ab29d91f2cc0c7e2ef3d68fbcfdd22c60e42d76ee179f8c85eb07f0bb1` | **RETAIN V1** | Direct Yield Baseline | MAE: $291.40\text{ kg/ha}$ | Pre-delta direct Ridge regression baseline. |
| `models/m1_improved_delta_xgboost.pkl`| 158,544 | `98442be7ffdf6eb7dc5fc1768c34f10bf8b5da02ff937072cb0fa8ce7ff857f6` | **EXPERIMENTAL** | Non-linear Delta | MAE: $274.50\text{ kg/ha}$ | Non-linear gradient boosted tree benchmark. |
| `models/m1_improved_delta_xgboost_conservative.pkl` | 103,012 | `606ee3cb2b74ddc42d3d0f0fa4b5a2bf405786a518e3dd567b4010a3014a4286` | **EXPERIMENTAL** | Regularized Boosted | MAE: $268.10\text{ kg/ha}$ | Depth-constrained XGBoost alternative. |

---

## 3. Member 2: Multi-Commodity Market Price & Volatility Inventory

### 3.1 Scope & Market Specification
- **Commodities:** Rice, Wheat, Potato, Tea.
- **Geographic Domain:** Major APMC wholesale mandis across India (WB, UP, PB, DL, AS).
- **Target Horizons:** Short-Term (1–7 days), 14-day calendar, 30-day calendar.
- **Target Outputs:**
  - Modal Wholesale Price ($\text{₹/quintal}$) with empirical prediction intervals (p10, p25, p50, p75, p90).
  - Volatility Regime Classification (`LOW`, `NORMAL`, `HIGH`) with calibrated posterior probabilities.

### 3.2 Member 2 Master Artifact Inventory Table

| Commodity | Version | Relative Artifact Path | Size (Bytes) | Certified SHA-256 Checksum | Release Tier | Horizon | Benchmark Performance | Model Architecture & Feature Dimensions |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: | :--- | :--- |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model.pkl` | 1,843,701 | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | **PRODUCTION** | 1–7d | Beats Persistence (-₹0.45/q MAE) | 48-feature XGBoost modal price regressor |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model_14d.pkl` | 1,572,754 | `1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab` | **OPT-IN** | 14d | Validated (MAE ₹59.87 vs ₹60.16) | 14-day direct calendar horizon price regressor |
| **Rice** | `1.0.0` | `artifacts/rice/rice_price_model_30d.pkl` | 1,492,514 | `ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343` | **EXPERIMENTAL** | 30d | Advisory Only (Parity on flat regimes) | 30-day direct calendar horizon price regressor |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_normalized_model.pkl` | 1,291,739 | `dfc83d89ae3e93d9abf225b669b82fd9e1788972797353529680f7937b614a9f` | **PRODUCTION** | 1–7d | Beats Persistence (+8.04 pp Acc) | Price-normalized 3-class XGBoost + Isotonic calibration (35 features) |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_model.pkl` | 1,762,176 | `f1acfa394e18b59cb2b25a318bbef7fe941626e8867ecccd80fc719622979253` | **RETAIN V1** | 1–7d | Baseline Reference | Uncalibrated raw volatility baseline |
| **Rice** | `1.0.0` | `artifacts/rice/rice_volatility_model_v2.pkl` | 361,371 | `8a21c26bb632f8e5910d0961d0c0b3a63c549f980c8b2822ec576b789d75446d` | **RETAIN V1** | 1–7d | Baseline Reference | Calibrated V2 volatility model |
| **Wheat** | `1.0.0` | `artifacts/wheat/wheat_price_model.pkl` | 804,457 | `9db711b7560494315cfff7775933d656b9378b89ad08152a4c4b9549787d7126` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost regressor (20 features) |
| **Wheat** | `1.0.0` | `artifacts/wheat/wheat_volatility_model.pkl` | 1,754,750 | `2f1ddebb9a46dd137f175980464e3f85004362e50fcbe7dacb09bbabb32785d4` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost classifier (19 features) |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_model.pkl` | 460,901 | `f7d388b3bae30e158de2e5513cbc756b2722026faea18fa5da9e650d56b1450c` | **PRODUCTION** | 1–7d | Beats Persistence (-₹0.19/q MAE, -₹14.61/q RMSE) | 81-feature CatBoost MAE regressor with deadband $\epsilon=3.0\text{ ₹/q}$ |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_14d_model.pkl` | 461,814 | `5a98a46eca598de1ff59aee6b14290dce76127b6449bc871e2e11f80d5fe555a` | **EXPERIMENTAL** | 14d | Advisory (Deficit vs Persistence) | 14-day direct CatBoost regressor with empirical bounds |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_price_30d_model.pkl` | 461,318 | `93e3dbdc6c64a8298d1102b0154d56a7746cd27d1fc8ded1ac843577dd8f3892` | **EXPERIMENTAL** | 30d | Advisory (Deficit vs Persistence) | 30-day direct CatBoost regressor with empirical bounds |
| **Wheat** | `2.0.0` | `artifacts/wheat/v2/wheat_volatility_model.pkl` | 355,504 | `b1ffea4abff19e17f8065e8711e78262e77643249c3131d4f92431f9f9b38348` | **PRODUCTION** | 1–7d | Beats Baseline (+25.09 pp Acc, 73.4% HIGH Prec) | 81-feature CatBoost classifier + Isotonic calibration |
| **Potato** | `1.0.0` | `artifacts/potato/potato_price_model.pkl` | 770,525 | `5ef4f95667d1231ce39c487d1156c65a9eb3e2bf482641273e8279b7c8412b3a` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost regressor (20 features) |
| **Potato** | `1.0.0` | `artifacts/potato/potato_volatility_model.pkl` | 1,754,696 | `23340d66dd7357d637e7f592edb373eb3f09efbf14bf3d576fc57364e2ea4fc9` | **RETAIN V1** | 1–7d | Baseline Reference | Historical V1 XGBoost classifier (19 features) |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_model.pkl` | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` | **PRODUCTION** | 1–7d | Matches Persistence MAE, Beats on RMSE (-₹1.05/q) | 81-feature XGBoost Pseudo-Huber regressor with deadband $\epsilon=5.0\text{ ₹/q}$ |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_14d_model.pkl` | 460,590 | `6088ac503bbfae7789c7166ca850c69c11113bf98aa9f572cefad7fd94114f17` | **EXPERIMENTAL** | 14d | Advisory (Deficit vs Persistence) | 14-day direct CatBoost regressor with empirical bounds |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_price_30d_model.pkl` | 460,694 | `de169b0211e03d036265ba4c73fe2df166532dd900ed5452ea339d13e25e992f` | **EXPERIMENTAL** | 30d | Advisory (Deficit vs Persistence) | 30-day direct CatBoost regressor with empirical bounds |
| **Potato** | `2.0.0` | `artifacts/potato/v2/potato_volatility_model.pkl` | 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` | **PRODUCTION** | 1–7d | Beats Baseline (+17.56 pp Acc, Balanced Acc 56.57%) | 81-feature XGBoost classifier + Platt Sigmoid calibration |
| **Tea** | `1.0.0` | `artifacts/tea/tea_price_model.pkl` | 772,999 | `e437c37a6b22eb01eb1fa2888cf36c05eb7614e7a6f2ea351b6dbb20eb3057e9` | **PRODUCTION** | 1–7d | Production Reference | 23-feature XGBoost modal price regressor |
| **Tea** | `1.0.0` | `artifacts/tea/tea_volatility_model.pkl` | 1,754,752 | `10bc93ea651ef8a1db576e330be4511dbb4e05d0d82847d95b542035eb4cf21e` | **PRODUCTION** | 1–7d | Production Reference | 19-feature XGBoost volatility classifier |

---

## 4. Commodity Capabilities & Cross-Module Support Matrix

The table below defines the statutory capability matrix exposed by the AgroVision-ML runtime across all modules:

| Commodity | Modal Price (1–7d) | Modal Price (14d) | Modal Price (30d) | Volatility Regime | Crop Yield Forecasting | Production Forecast ($\text{tonnes}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rice** | **PRODUCTION** | **OPT-IN** | **EXPERIMENTAL** | **PRODUCTION** | **PRODUCTION** (WB Aus, Aman, Boro) | **SUPPORTED** (Strictly requires `cultivated_area_ha`) |
| **Wheat** | **PRODUCTION** | **EXPERIMENTAL** | **EXPERIMENTAL** | **PRODUCTION** | **UNSUPPORTED** | **UNSUPPORTED** |
| **Potato** | **PRODUCTION** | **EXPERIMENTAL** | **EXPERIMENTAL** | **PRODUCTION** | **UNSUPPORTED** | **UNSUPPORTED** |
| **Tea** | **PRODUCTION** | **UNSUPPORTED** | **UNSUPPORTED** | **PRODUCTION** | **UNSUPPORTED** | **UNSUPPORTED** |

### Critical Enforcement Invariant:
- Calls to `predict_yield(commodity="wheat")`, `predict_yield(commodity="potato")`, or `predict_yield(commodity="tea")` raise an explicit `ValueError`:
  ```python
  ValueError: Commodity 'wheat' is not supported for crop yield prediction. Member 1 supports 'rice' only.
  ```
- No default values, no synthetic estimations, and no cross-commodity borrowing are permitted for unsupported yield features.

---

## 5. Artifact Verification & Checksum Audit Command

To independently verify the integrity of all production artifacts in this registry, execute the following PowerShell command in the project root:

```powershell
Get-FileHash -Path member1-yield-model/models/m1_improved_delta_ridge.pkl -Algorithm SHA256
```
Expected output:
```
Algorithm       Hash                                                               Path
---------       ----                                                               ----
SHA256          3099C2D6D113105F185111F6B1F02004229CC0E32318E9F35532841F7D4DCB56  ...
```

For Member 2 baseline artifacts:
```powershell
python -c "import json, hashlib; [print(f'{k}: OK') for k, v in json.load(open('member2-price-volatility/artifacts/baseline_sha256_checksums.json')).items() if hashlib.sha256(open(k, 'rb').read()).hexdigest() == v]"
```

---

## 6. Sign-off & Audit Certification

- **Lead Auditor:** Senior MLOps Architect & Release Auditor
- **Model Governance Authority:** Principal Agricultural ML Researcher
- **Audit Timestamp:** 2026-11-17T09:07:00+05:30
- **Registry Status:** CERTIFIED & LOCKED FOR STAGING TO PRODUCTION TRANSITION
