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
| `wheat_price_model.pkl` | 460,901 | `f7d388b3bae30e158de2e5513cbc756b2722026faea18fa5da9e650d56b1450c` | CatBoost MAE Champion ($\epsilon=3.0$, 81 feats) |
| `wheat_price_model_v2.pkl` | 460,901 | `f7d388b3bae30e158de2e5513cbc756b2722026faea18fa5da9e650d56b1450c` | Standardized V2 model binary alias |
| `wheat_price_14d_model.pkl`| 461,814 | `5a98a46eca598de1ff59aee6b14290dce76127b6449bc871e2e11f80d5fe555a` | Direct 14D CatBoost Regressor |
| `wheat_price_30d_model.pkl`| 461,318 | `93e3dbdc6c64a8298d1102b0154d56a7746cd27d1fc8ded1ac843577dd8f3892` | Direct 30D CatBoost Regressor |
| `wheat_volatility_model.pkl`| 355,504 | `b1ffea4abff19e17f8065e8711e78262e77643249c3131d4f92431f9f9b38348` | Isotonic Calibrated CatBoost Classifier |
| `wheat_volatility_model_v2.pkl`| 355,504 | `b1ffea4abff19e17f8065e8711e78262e77643249c3131d4f92431f9f9b38348` | Standardized V2 volatility binary alias |
| `wheat_feature_config.json` | 4,631 | `2f17e848a86b0bcde532420616f789cf5d54efe7765e66058394ae5d60b88b06` | V2 81-feature runtime definition |
| `wheat_price_model_v2_config.json` | 4,631 | `2f17e848a86b0bcde532420616f789cf5d54efe7765e66058394ae5d60b88b06` | Standardized V2 price config alias |
| `wheat_volatility_config.json`| 444 | `c2c3a75ae1aa0732c9208c52164776bd01f8ac640908c00b1bf00c0840045d85` | Volatility percentage thresholds |
| `wheat_volatility_model_v2_config.json`| 444 | `c2c3a75ae1aa0732c9208c52164776bd01f8ac640908c00b1bf00c0840045d85` | Standardized V2 volatility config alias |
| `wheat_residual_distributions.json`| 1,430 | `1e0e857282d6bee6befd995f3b08151a82ab32e577ecd6e7711eda8f1839f4c0` | Empirical holdout quantiles (p05-p95) |
| `wheat_v2_metrics.json` | 5,877 | `7828526e8ab7d76c78b31147788aa607525079c88b11d44ccacbdb648e6200e5` | Multi-population performance metrics |
| `manifest.json` | 2,011 | `9431f48c5843d230434de84e2e49989c13d2eb46c622d763d07fa95967b5b6c7` | Cryptographic directory manifest |

*Wheat Historical Baseline (V1):*
- `artifacts/wheat/wheat_price_model.pkl`: SHA-256 `9db711b7560494315cfff7775933d656b9378b89ad08152a4c4b9549787d7126` (Preserved bit-for-bit).
- `artifacts/wheat/wheat_volatility_model.pkl`: SHA-256 `2f1ddebb9a46dd137f175980464e3f85004362e50fcbe7dacb09bbabb32785d4` (Preserved bit-for-bit).

---

### 2.3 Potato (*Solanum tuberosum*) — Active Production V2

- **Status:** **ACTIVE PRODUCTION V2** (V1 preserved in `artifacts/potato/` as baseline)
- **V2 Artifact Directory:** `member2-price-volatility/artifacts/potato/v2/`

| Artifact File | Size (Bytes) | SHA-256 Checksum | Model Family / Specification |
| :--- | :---: | :--- | :--- |
| `potato_price_model.pkl` | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` | XGBoost Pseudo-Huber Champion ($\epsilon=5.0$) |
| `potato_price_model_v2.pkl` | 584,465 | `f88323ac8aac1fac0278cbeca9463eec0d75e35fa4e5758fe74e15c75fe6e3a8` | Standardized V2 model binary alias |
| `potato_price_14d_model.pkl`| 460,590 | `6088ac503bbfae7789c7166ca850c69c11113bf98aa9f572cefad7fd94114f17` | Direct 14D CatBoost Regressor |
| `potato_price_30d_model.pkl`| 460,694 | `de169b0211e03d036265ba4c73fe2df166532dd900ed5452ea339d13e25e992f` | Direct 30D CatBoost Regressor |
| `potato_volatility_model.pkl`| 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` | Sigmoid Calibrated XGBoost Classifier |
| `potato_volatility_model_v2.pkl`| 2,031,230 | `6aa4beae4eb4a6a7f33a2e9b11f44c95292cd576b7816a1d8d797c3c4c289b31` | Standardized V2 volatility binary alias |
| `potato_feature_config.json` | 4,621 | `f5cf139d37ebc8988a2257da72bbc3422abdcc6fe0562b12ed2ae73a096d4b53` | V2 81-feature runtime definition |
| `potato_price_model_v2_config.json` | 4,621 | `f5cf139d37ebc8988a2257da72bbc3422abdcc6fe0562b12ed2ae73a096d4b53` | Standardized V2 price config alias |
| `potato_volatility_config.json`| 444 | `18fcf076a400adf5f44bb15303a4f056102900951754c18c3cdc61cd77642ddb` | Volatility percentage thresholds |
| `potato_volatility_model_v2_config.json`| 444 | `18fcf076a400adf5f44bb15303a4f056102900951754c18c3cdc61cd77642ddb` | Standardized V2 volatility config alias |
| `potato_residual_distributions.json`| 1,416 | `5d8595ec7dd51932f71ce13681870afd55d78a519011bd7288990bd264c8d877` | Empirical holdout quantiles (p05-p95) |
| `potato_v2_metrics.json` | 5,958 | `69dbbe597e26568206e62d89f7d9512c0fa872bd6fc87d1c769984028ffea11a` | Multi-population performance metrics |
| `manifest.json` | 2,026 | `d8f7aad6b8a1d5d6c3d30e754bdd804dcff6f8db22e899f3b930f951fbad2eef` | Cryptographic directory manifest |

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
