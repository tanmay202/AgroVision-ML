# AgroVision-ML — Member 2: Artifacts Catalog & Directory Structure

This directory stores all trained machine learning models, feature configurations, residual error distributions, and canonical group mappings for Member 2.

---

## 1. Directory Structure

Artifacts are separated cleanly by commodity subdirectories, while also maintaining flat root references for backward compatibility:

```text
artifacts/
├── rice/                              ← Separated Rice models & configurations
│   ├── rice_price_model.pkl           ← Frozen production 48-feature price model
│   ├── rice_price_final_config.json   ← Feature list & training parameters
│   ├── rice_price_model_14d.pkl       ← Direct 14-day calendar horizon price model
│   ├── rice_price_model_14d_config.json
│   ├── rice_price_model_30d.pkl       ← Direct 30-day calendar horizon price model
│   ├── rice_price_model_30d_config.json
│   ├── rice_volatility_model.pkl      ← Legacy nominal volatility classifier
│   ├── rice_volatility_normalized_model.pkl ← Champion normalized volatility classifier
│   ├── rice_volatility_normalized_config.json
│   ├── rice_feature_config.json
│   ├── rice_group_mappings.json       ← 87 mandis and 34 varieties
│   ├── rice_holdout_residuals.json    ← Short-term empirical residual distribution
│   ├── rice_holdout_residuals_7d.json ← 7-day empirical residual distribution
│   ├── rice_holdout_residuals_14d.json← 14-day empirical residual distribution
│   ├── rice_holdout_residuals_30d.json← 30-day empirical residual distribution
│   └── README.md
│
├── wheat/                             ← Separated Wheat models & configurations
│   ├── wheat_price_model.pkl          ← Trained 20-feature XGBoost price regressor
│   ├── wheat_volatility_model.pkl     ← Trained 19-feature class-balanced classifier
│   ├── wheat_feature_config.json      ← Complete feature list & test metrics
│   ├── wheat_volatility_config.json   ← Thresholds & parameters
│   ├── wheat_group_mappings.json      ← 38 mandis and 16 varieties
│   └── README.md
│
├── potato/                            ← Separated Potato models & configurations
│   ├── potato_price_model.pkl         ← Trained 20-feature XGBoost price regressor
│   ├── potato_volatility_model.pkl    ← Trained 19-feature class-balanced classifier
│   ├── potato_feature_config.json     ← Complete feature list & test metrics
│   ├── potato_volatility_config.json  ← Thresholds & parameters
│   ├── potato_group_mappings.json     ← 88 mandis and 13 varieties
│   └── README.md
│
├── tea/                               ← Separated Tea models & configurations
│   ├── tea_price_model.pkl            ← Baseline 20-feature price regressor
│   ├── tea_volatility_model.pkl       ← Baseline 19-feature volatility classifier
│   ├── tea_feature_config.json
│   ├── tea_volatility_config.json
│   ├── tea_group_mappings.json        ← 13 mandis
│   └── README.md
│
├── v3/                                ← Prospective shadow candidate models
│   ├── rice_volatility_v3_14d_model.pkl   ← Frozen candidate 14D classifier
│   └── rice_volatility_v3_14d_config.json ← Candidate config with isotonic calibrator
│
├── legacy/                            ← Archived exploratory models
│
├── model_handoff_manifest.json        ← Certified deliverables handoff manifest
└── README.md                          ← This document
```

---

## 2. Commodity Artifact Summary

| Commodity | Artifact Directory | Price Model File | Volatility Model File | Feature Config | Group Mappings |
|:---|:---|:---|:---|:---|:---|
| **Rice** | `artifacts/rice/` | `rice_price_model.pkl` | `rice_volatility_normalized_model.pkl` | `rice_price_final_config.json` | `rice_group_mappings.json` (87 mandis) |
| **Wheat** | `artifacts/wheat/` | `wheat_price_model.pkl` | `wheat_volatility_model.pkl` | `wheat_feature_config.json` | `wheat_group_mappings.json` (38 mandis) |
| **Potato** | `artifacts/potato/` | `potato_price_model.pkl` | `potato_volatility_model.pkl` | `potato_feature_config.json` | `potato_group_mappings.json` (88 mandis) |
| **Tea** | `artifacts/tea/` | `tea_price_model.pkl` | `tea_volatility_model.pkl` | `tea_feature_config.json` | `tea_group_mappings.json` (13 mandis) |

---

## 3. Backward Compatibility Invariant

Existing callers and automated unit tests importing directly from `artifacts/` (e.g., `artifacts/rice_price_model.pkl`) remain 100% supported. The configuration loader automatically resolves artifacts from both the commodity subfolder (`artifacts/{commodity}/`) and the top-level `artifacts/` folder.
