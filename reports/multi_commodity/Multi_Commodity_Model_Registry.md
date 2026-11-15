# AgroVision-ML — Multi-Commodity Production Model Registry
## Master Cryptographic & Architectural Model Inventory

**Document Version:** 2.0.0  
**Effective Date:** 2026-11-15  
**Registry Scope:** All Production & Baseline Commodities (Rice, Wheat, Potato, Tea)  
**Maintained By:** Principal ML Systems Architect & Production Operations  

---

## 1. Multi-Commodity Platform Architecture

The AgroVision platform enforces strict architectural separation between distinct commodities while exposing a unified prediction runtime for Member 3 (FastAPI service):
- **Rice:** Protected Production Reference system (48 features, XGBoost Regressor, XGBoost Volatility + Isotonic Calibration).
- **Wheat:** Production V2 System (81 features, CatBoost MAE Regressor with deadband, CatBoost Volatility + Isotonic Calibration).
- **Potato:** Production V2 System (81 features, XGBoost Pseudo-Huber Regressor with deadband, XGBoost Volatility + Sigmoid Calibration).
- **Tea:** Active Production Model (20 features, XGBoost Regressor, XGBoost Volatility).

---

## 2. Production Registry & Cryptographic Inventory

### 2.1 Rice (*Oryza sativa*) — Protected Production Reference

- **Status:** **FROZEN PRODUCTION** (Strictly Protected; Never Retrain/Modify)
- **Primary Artifact Directory:** `member2-price-volatility/artifacts/rice/`

| Artifact File | Size (Bytes) | SHA-256 Checksum | Model Family / Specification |
| :--- | :---: | :--- | :--- |
| `rice_price_model.pkl` | 1,843,701 | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | XGBoost Regressor (48 features) |
| `rice_price_model_14d.pkl` | 1,572,754 | `1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab` | XGBoost Regressor 14D Horizon |
| `rice_price_model_30d.pkl` | 1,492,514 | `ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343` | XGBoost Regressor 30D Horizon |
| `rice_volatility_model.pkl` | 1,762,176 | `fcfcf3794ce1ce4fb26ce5a539bc7a3f4e246736283db85c01b1311ff928ee2a` | XGBoost Classifier (Raw Baseline) |
| `rice_volatility_model_v2.pkl` | 361,371 | `f2e82500c01faee984364cb1a5e16541f40d7c71f92e02b781da24a3aa583925` | Calibrated XGBoost Classifier |
| `rice_volatility_normalized_model.pkl` | 1,291,739 | `9622d1645e75c6bf23d8c116c278e9f5bc3a85bca31a78b53fb4f7c8ec17454f` | Normalized Volatility Classifier |
| `rice_price_final_config.json` | 1,736 | `508ad9d8fddab3d0baa65cee1d7f82c546fbc33253bf23d8ec23fe1f57643de7` | 48-feature config |
| `rice_holdout_residuals.json` | 237,477 | `dfc008a15b4b29abff61393b567c2d8c2211d9a294fc6a360d117a763084d599` | Empirical error quantiles |

---

### 2.2 Wheat (*Triticum aestivum*) — Active Production V2

- **Status:** **ACTIVE PRODUCTION V2** (V1 preserved in `artifacts/wheat/` as baseline)
- **V2 Artifact Directory:** `member2-price-volatility/artifacts/wheat/v2/`

| Artifact File | Size (Bytes) | SHA-256 Checksum | Model Family / Specification |
| :--- | :---: | :--- | :--- |
| `wheat_price_model.pkl` | 460,901 | `d280b080f55fb3beee73809ee59990b7952fb58f629c4aa0292fe8db1ff18d6e` | CatBoost MAE Champion ($\epsilon=3.0$, 81 feats) |
| `wheat_price_14d_model.pkl`| 461,814 | `1bc19df770519fbb11470cefeee0028292850ae35741f2a36b32c0216ae9cbbe` | Direct 14D CatBoost Regressor |
| `wheat_price_30d_model.pkl`| 461,318 | `15e98547ca79b4fe247594d2dfd87b92789f2cfbc4c405a2feff98eb820a44cb` | Direct 30D CatBoost Regressor |
| `wheat_volatility_model.pkl`| 355,504 | `86f675661b3b24f5933fffa8eb3cc7efda3ba17f739665f8c634625b1ae950a2` | Isotonic Calibrated CatBoost Classifier |
| `wheat_feature_config.json` | 4,631 | `beeead23eb811634b07cf0fba8c9197fe8a3ba081fe7248f72a441dfdfd0a1b8` | V2 81-feature runtime definition |
| `wheat_volatility_config.json`| 444 | `9d554a9c379bfbe2d6a543f49747514a3e29f3ffbfaf713583cae44a499313db` | Volatility percentage thresholds |
| `wheat_residual_distributions.json`| 1,430 | `ca0cb18fc80ff43bb3aa84f50f2fb6e54ee0d55e08b688d011fbe244da026e63` | Empirical holdout quantiles (p05-p95) |
| `wheat_v2_metrics.json` | 5,877 | `26b89694e99f57d6928e3b5e4aa3417387b9b03943fcf09600e12d4d5a99905b` | Multi-population performance metrics |
| `manifest.json` | 1,366 | `668702b8bbcb75a898b3687c4f4ca5cb3bf441b804da2f8547e24749f506e001` | Cryptographic directory manifest |

*Wheat Historical Baseline (V1):*
- `artifacts/wheat/wheat_price_model.pkl`: SHA-256 `3a8f59bf8ce3a00508ffb6b45a6c3f58a741c099c08796123aafebe4b6113b2c` (Preserved bit-for-bit).
- `artifacts/wheat/wheat_volatility_model.pkl`: SHA-256 `6e74bc4e0c06ba1f3918a9fc4e9951684c311c8d50451f211322e6cf4848bb28` (Preserved bit-for-bit).

---

### 2.3 Potato (*Solanum tuberosum*) — Active Production V2

- **Status:** **ACTIVE PRODUCTION V2** (V1 preserved in `artifacts/potato/` as baseline)
- **V2 Artifact Directory:** `member2-price-volatility/artifacts/potato/v2/`

| Artifact File | Size (Bytes) | SHA-256 Checksum | Model Family / Specification |
| :--- | :---: | :--- | :--- |
| `potato_price_model.pkl` | 584,465 | `3fa4e1cffdb870c40e53a992bc93d62ea75fcce06b2520638575a7c296568285` | XGBoost Pseudo-Huber Champion ($\epsilon=5.0$) |
| `potato_price_14d_model.pkl`| 460,590 | `5ca64a2e55883ea54fe9a1286b5c3080ff559f2010834316238b97d264421b92` | Direct 14D CatBoost Regressor |
| `potato_price_30d_model.pkl`| 460,694 | `6cb803ba97e937d53086eb2d69956d49ca71752b04c86bf4157d6e676104fa20` | Direct 30D CatBoost Regressor |
| `potato_volatility_model.pkl`| 2,031,230 | `d0dd4eb005527a0505cefa875e5332fbc1be2b36997d9fc0888981440785fffc` | Sigmoid Calibrated XGBoost Classifier |
| `potato_feature_config.json` | 4,621 | `d12be4078832a8ba32448375e2ecb568770c0c70e28f3a8b273b5a7698cbfa2d` | V2 81-feature runtime definition |
| `potato_volatility_config.json`| 444 | `a739d48b1aa6e20ddbe980a3a4131b790d97033bbcc498fe87968dd4892cfa71` | Volatility percentage thresholds |
| `potato_residual_distributions.json`| 1,416 | `7f794713ea7970897f1bb653be161c02e1c9fe341e3a6a9be71e612cb353272d` | Empirical holdout quantiles (p05-p95) |
| `potato_v2_metrics.json` | 5,958 | `42927289ef32a0d783109a9ea0a1c7201c13bc5a932d8816c87bb3cbeeb06745` | Multi-population performance metrics |
| `manifest.json` | 1,376 | `b783d7373fef40df5f0962dc2d1ce2b9183d3b7623a9d94921b7ba94b9aa1ef4` | Cryptographic directory manifest |

*Potato Historical Baseline (V1):*
- `artifacts/potato/potato_price_model.pkl`: SHA-256 `5ef4f95667d1231ce39c487d1156c65a9eb3e2bf482641273e8279b7c8412b3a` (Preserved bit-for-bit).
- `artifacts/potato/potato_volatility_model.pkl`: SHA-256 `23340d66dd7357d637e7f592edb373eb3f09efbf14bf3d576fc57364e2ea4fc9` (Preserved bit-for-bit).

---

### 2.4 Tea (*Camellia sinensis*) — Active Production

- **Status:** **ACTIVE PRODUCTION**
- **Artifact Directory:** `member2-price-volatility/artifacts/tea/`

| Artifact File | Size (Bytes) | SHA-256 Checksum | Model Family / Specification |
| :--- | :---: | :--- | :--- |
| `tea_price_model.pkl` | 594,157 | `fd92e2175c2eecfa43e4905cfbe091a134a41b52f6797d7f7bc865fc20f837e2` | XGBoost Regressor (20 features) |
| `tea_volatility_model.pkl` | 1,685,122 | `b5aaefc46440db6cf08dbf84f3e69f826359f518e246be6cf5f76bc0a41d0615` | XGBoost Classifier (20 features) |
| `tea_feature_config.json` | 1,787 | `7d75bc314fb5f40ee78c8a1b52a51f08bc1fcf0b08e5e8e8071e7aeb294e7978` | Feature configuration |

---

## 3. Runtime Resolution & Compatibility Matrix

The prediction runtime (`member2-price-volatility/src/models/predict.py` and `shared/predict_bridge.py`) implements automatic model resolution:

```
predict(data, commodity="...")
├── commodity="rice"   ──> RiceInferencePipeline (48 feats, XGBoost, zero lookahead)
├── commodity="wheat"  ──> Wheat V2 (81 feats, CatBoost MAE, deadband eps=3.0)
│                          [Optionally: version="v1" loads historical V1 XGBoost]
├── commodity="potato" ──> Potato V2 (81 feats, XGBoost PseudoHuber, deadband eps=5.0)
│                          [Optionally: version="v1" loads historical V1 XGBoost]
└── commodity="tea"    ──> Tea V1 (20 feats, XGBoost)
```

Direct horizon forecasting (`predict_price_horizon(data, horizon_days=14/30)`) is natively supported across **Rice, Wheat, and Potato**, complete with empirical non-parametric confidence bands.
