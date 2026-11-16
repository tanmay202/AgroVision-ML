# AgroVision-ML — Production Release Governance & Model Lifecycle Policy

**Milestone:** Milestone 2: Real-World Integration, APMC Feed, and Staging Deployment Readiness  
**Target Environment:** Staging / Pre-Production  
**Date:** November 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

This governance document formalizes the release criteria, verification gates, promotion requirements, and model lifecycle policies governing AgroVision-ML. It enforces strict separation between **Production** models (currently serving farmer advisories) and **Experimental / Shadow** models undergoing prospective evaluation.

Under this policy, no experimental model can be promoted to production without statistically significant, empirical non-inferiority evidence gathered over complete prospective forecasting horizons.

---

## 2. Release Freeze & Version Tagging Policy

### 2.1 Release Baseline Tag: `v2.0.0-staging`
- **Git Commit:** `f35db19` (verified clean working tree).
- **Release Tag:** `v2.0.0-staging` (annotated tag).
- **Protected Artifacts Baseline:** 63 locked binary model files, scalers, encoders, and configuration files verified bit-for-bit via SHA-256 (`member2-price-volatility/artifacts/baseline_sha256_checksums.json`).
- **Release Freeze Invariant:** Zero model retraining or modification of protected artifacts is permitted during the staging validation phase.

---

## 3. Current Unified Model Registry & Release Status

| Commodity | Prediction Task | Active Model | Version | Release Status | Public Default | Farmer Facing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rice** | Short-term Modal Price (1–7d) | XGBoost (48 features) | `1.0.0` | **Production** | YES | YES |
| **Rice** | Price Volatility Class | Normalized Multi-Class ML | `2.0.0` | **Production** | YES | YES |
| **Rice** | 14-Day Horizon Price | Direct XGBoost | `1.0.0` | **Opt-In** | NO | NO (Opt-In Only) |
| **Rice** | 30-Day Horizon Price | Direct XGBoost | `1.0.0` | **Experimental** | NO | NO |
| **Wheat** | Short-term Modal Price (1–7d) | CatBoost V1 | `1.0.0` | **Production** | YES | YES |
| **Wheat** | 14-Day Horizon Price | CatBoost V2 (Direct) | `2.0.0` | **Experimental** | NO | NO (Shadow Only) |
| **Wheat** | 30-Day Horizon Price | CatBoost V2 (Direct) | `2.0.0` | **Experimental** | NO | NO (Shadow Only) |
| **Potato** | Short-term Modal Price (1–7d) | CatBoost V1 | `1.0.0` | **Production** | YES | YES |
| **Potato** | 14-Day Horizon Price | CatBoost V2 (Direct) | `2.0.0` | **Experimental** | NO | NO (Shadow Only) |
| **Potato** | 30-Day Horizon Price | CatBoost V2 (Direct) | `2.0.0` | **Experimental** | NO | NO (Shadow Only) |
| **Tea** | Next-Day Modal Price (1–7d) | XGBoost | `1.0.0` | **Production** | YES | YES |
| **West Bengal Rice** | Crop Yield Intensity (t/ha) | Improved Delta Ridge | `2.1.0` | **Production** | YES | YES |

---

## 4. Production Promotion Gates

To promote any experimental model (e.g., Wheat 14d/30d, Potato 14d/30d, or Rice 30d) to production default, the model must pass all five mandatory promotion gates:

```
+-------------------------------------------------------------------------------+
|                         GATE 1: SAMPLE SIZE THRESHOLD                         |
|  - Minimum 30 prospective evaluated forecasts on distinct APMC mandi feeds.   |
|  - All forecasts must be evaluated strictly after complete target windows.    |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                    GATE 2: MATCHED NON-INFERIORITY BENCHMARK                  |
|  - Model MAE must not exceed Persistence baseline: Delta_MAE <= 0.0 Rs./Qtl.  |
|  - Model Win Rate must be >= 50.0% on identical matched forecast origins.     |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                     GATE 3: RESIDUAL & INTERVAL VALIDATION                    |
|  - Valid empirical holdout error quantiles present (p10, p25, p50, p75, p90). |
|  - Prediction intervals must be strictly monotonic (p10 <= p50 <= p90).       |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                    GATE 4: LATENCY & SLA VERIFICATION                         |
|  - P95 inference latency must be < 500ms under simulated production load.     |
|  - Zero memory leaks or file descriptor exhaustion over 1,000 requests.       |
+-------------------------------------------------------------------------------+
                                        |
                                        v
+-------------------------------------------------------------------------------+
|                    GATE 5: INDEPENDENT CODEOWNER APPROVAL                     |
|  - Complete regression test suite passes (0 failures).                        |
|  - Signed sign-off from agricultural domain lead and ML engineering lead.     |
+-------------------------------------------------------------------------------+
```

---

## 5. Model Deprecation & Lifecycle Governance

### 5.1 Semantic Versioning Convention
Models follow `MAJOR.MINOR.PATCH`:
- **MAJOR:** Breaking changes to feature schema, target variable definition, or API response contract.
- **MINOR:** Retrained model on updated historical data passing non-inferiority benchmarks with backward-compatible schema.
- **PATCH:** Hyperparameter fine-tuning or bug fixes without altering input/output interfaces.

### 5.2 Deprecation Notice Policy
When a model or contract key is superseded:
1. **Deprecation Period:** A minimum 1-major-version grace period is mandatory (e.g., the legacy alias `yield_arrivals_tonnes` was deprecated in v2.0 with a warning and is scheduled for permanent removal in v3.0).
2. **Deprecation Warnings:** Programmatic warnings (`warnings.warn(..., DeprecationWarning)`) are emitted upon access without breaking existing client requests.
3. **Migration Guide:** A formal migration guide must be published detailing canonical replacement keys.

### 5.3 Automated Rollback Trigger Conditions
A promoted model is automatically demoted to shadow mode and rolled back to the previous verified release if any of the following occur in staging or production:
1. Production prediction error rate exceeds **5.0%** over a rolling 1-hour window.
2. Inference P95 latency exceeds **1,000ms** for more than 5 consecutive minutes.
3. Feature distribution drift exceeds $4.0\sigma$ without an identified external market shock.
4. Model prediction residual exhibits systematic directional bias ($\text{Mean Residual} > 2.0 \times \text{Historical Std}$).

---

## 6. Audit Trail & Reproducibility Guarantees

1. **Immutable Artifacts:** Model artifact weights are never modified in-place. New models must be saved under new filenames or versioned directories (`artifacts/{commodity}/v{N}/`).
2. **Traceable Origins:** Every prediction response includes `model_name`, `model_version`, and `release_status`.
3. **Point-in-Time Shadow Ledger:** All shadow evaluations are logged with deterministic forecast IDs derived from origin inputs, ensuring 100% retrospective auditability.

---

## 7. Governance Sign-off

This release governance policy is binding for all staging evaluations and future production deployments of AgroVision-ML.

**Verdict:** `GOVERNANCE APPROVED`
