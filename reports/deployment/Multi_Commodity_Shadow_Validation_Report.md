# AgroVision-ML — Multi-Commodity Prospective Shadow Validation Report

**Milestone:** Milestone 2: Real-World Integration, APMC Feed, and Staging Deployment Readiness  
**Target Environment:** Staging / Pre-Production  
**Date:** November 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

Prospective shadow evaluation provides controlled, non-disruptive, real-world evaluation of candidate machine learning models against actual materialized agricultural mandi observations. Candidate models run in parallel with production pipelines, generating forecasts from actual point-in-time data without exposing predictions to farmers, agricultural extension officers, or public APIs.

This report establishes the shadow monitoring architecture across all four commodities (**Rice**, **Wheat**, **Potato**, and **Tea**), details the delayed outcome matching protocol, and documents the ledger schema and benchmarking framework.

---

## 2. Multi-Commodity Shadow Architecture

```
                               Incoming Mandi Observation
                                           |
                                           v
                 +---------------------------------------------------+
                 |           Mandi Observations Ingestion            |
                 |      (Normalized, Canonical APMC Records)         |
                 +---------------------------------------------------+
                                           |
                     +---------------------+---------------------+
                     |                     |                     |
                     v                     v                     v
            [ Rice Shadow DB ]    [ Wheat Shadow DB ]  [ Potato Shadow DB ]  [ Tea Shadow DB ]
            (shadow_evaluation.db)(shadow_wheat.db)    (shadow_potato.db)   (shadow_tea.db)
                     |                     |                     |                  |
                     +---------------------+---------------------+------------------+
                                           |
                                           v
                 +---------------------------------------------------+
                 |             Point-in-Time Generation              |
                 |  - Idempotent deterministic forecast IDs          |
                 |  - Strict historical window (Zero future leakage) |
                 |  - Recorded initial state: PENDING                |
                 +---------------------------------------------------+
                                           |
                          Target Horizon Passes (t + H days)
                                           |
                                           v
                 +---------------------------------------------------+
                 |             Delayed Outcome Evaluator             |
                 |  - Evaluates only after target window elapses     |
                 |  - Matched observation -> EVALUATED               |
                 |  - Empty window closed -> TARGET_UNAVAILABLE      |
                 |  - Window still active -> PENDING                 |
                 +---------------------------------------------------+
                                           |
                                           v
                 +---------------------------------------------------+
                 |            Matched Benchmarking Engine            |
                 |  - Model MAE/RMSE/R2 vs. Persistence Baseline     |
                 |  - Non-inferiority margin validation              |
                 |  - Win rate (%) on identical matched origins      |
                 +---------------------------------------------------+
```

---

## 3. Strict Commodity Isolation

Each commodity operates on an independent, physically isolated SQLite database to prevent cross-commodity data pollution or locking contention:

| Commodity | Database Path | Active Horizons | Candidate Model | Production Default |
| :--- | :--- | :--- | :--- | :--- |
| **Rice** | `data/processed/shadow_evaluation.db` | 14d, 30d | XGBoost (v1.0.0) | Calendar Persistence / V1 Short |
| **Wheat** | `data/processed/shadow_evaluation_wheat.db` | 14d, 30d | CatBoost V2 (v2.0.0) | V1 Production Short-term |
| **Potato** | `data/processed/shadow_evaluation_potato.db` | 14d, 30d | CatBoost V2 (v2.0.0) | V1 Production Short-term |
| **Tea** | `data/processed/shadow_evaluation_tea.db` | 1d (Next-Day) | XGBoost (v1.0.0) | 1-Day Lag Persistence |

---

## 4. Shadow Ledger Schema (`shadow_forecasts`)

The unified schema enforces capture of all required audit dimensions:

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `forecast_id` | `TEXT PRIMARY KEY` | Deterministic SHA-256 hash of `(commodity, market, variety, origin_date, horizon, version)` |
| `commodity` | `TEXT NOT NULL` | Commodity identifier (`rice`, `wheat`, `potato`, `tea`) |
| `market` | `TEXT NOT NULL` | Canonical APMC wholesale mandi name |
| `variety` | `TEXT NOT NULL` | Canonical commodity variety |
| `origin_date` | `TEXT NOT NULL` | Point-in-time origin date (YYYY-MM-DD) |
| `horizon_days` | `INTEGER NOT NULL` | Calendar forecast horizon (1 for Tea; 14 or 30 for Rice/Wheat/Potato) |
| `origin_modal_price`| `REAL NOT NULL` | Realized price at forecast origin (Rs./Quintal) |
| `predicted_price_change`| `REAL NOT NULL` | Predicted price delta from origin |
| `predicted_modal_price`| `REAL NOT NULL` | Predicted future price |
| `requested_target_date`| `TEXT NOT NULL` | Target date (`origin_date + horizon_days`) |
| `target_window_start`| `TEXT NOT NULL` | Earliest allowable matching date |
| `target_window_end` | `TEXT NOT NULL` | Latest allowable matching date (`target_date + 3 days`) |
| `model_name` | `TEXT NOT NULL` | Artifact name (e.g., `wheat-price-14d-catboost`) |
| `model_version` | `TEXT NOT NULL` | Semantic version (`1.0.0` or `2.0.0`) |
| `volatility_pred_class`| `TEXT NOT NULL` | Predicted volatility class (`LOW`, `MEDIUM`, `HIGH`) |
| `volatility_probabilities_json`| `TEXT` | Class probability distribution JSON |
| `persistence_pred_price`| `REAL NOT NULL` | Persistence benchmark price (= `origin_modal_price`) |
| `created_at` | `TEXT NOT NULL` | ISO-8601 UTC creation timestamp |
| `status` | `TEXT NOT NULL` | Lifecycle state: `PENDING`, `EVALUATED`, `TARGET_UNAVAILABLE` |
| `matched_observation_date`| `TEXT` | Actual date of realized mandi report |
| `matched_actual_price` | `REAL` | Actual realized price (Rs./Quintal) |
| `actual_horizon_offset` | `INTEGER` | Realized calendar offset (`matched_date - origin_date`) |
| `prediction_residual` | `REAL` | Prediction error: `round(actual_price - pred_price, 2)` |
| `actual_volatility_outcome`| `TEXT` | Realized volatility outcome (`LOW`, `MEDIUM`, `HIGH`) |
| `absolute_error` | `REAL` | Model absolute error: `abs(pred_price - actual_price)` |
| `percentage_error` | `REAL` | Model MAPE: `(abs_error / actual_price) * 100` |
| `persistence_absolute_error`| `REAL` | Persistence absolute error: `abs(origin_price - actual_price)` |
| `persistence_percentage_error`| `REAL`| Persistence MAPE |
| `evaluated_at` | `TEXT` | ISO-8601 UTC timestamp of outcome matching |
| `finalization_reason` | `TEXT` | `MATCHED_WINDOW_OBSERVATION` or `WINDOW_CLOSED_WITHOUT_OBSERVATION` |
| `cohort` | `TEXT` | `PROSPECTIVE` or `RETROSPECTIVE` |

---

## 5. Delayed Outcome Evaluation Protocol

To prevent lookahead bias and premature finalization, shadow forecasts are evaluated strictly after their target forecasting window has elapsed:

1. **Target Window Definition:**
   For a forecast generated at origin $t$ with horizon $H$:
   $$\text{Window} = [t + H, t + H + \Delta_{\text{tol}}]$$
   where $\Delta_{\text{tol}} = 3$ calendar days (accommodating APMC weekend/holiday reporting lags).

2. **State Transition Rules:**
   - **`PENDING`:** Current evaluation date $T_{\text{eval}} \le t + H + \Delta_{\text{tol}}$ and no valid mandi observation has appeared yet.
   - **`EVALUATED`:** The first valid observation appears in the window $[t + H, t + H + \Delta_{\text{tol}}]$. The actual price is locked, error metrics and residuals are calculated, and the record transitions immediately to `EVALUATED`.
   - **`TARGET_UNAVAILABLE`:** The evaluation date $T_{\text{eval}} > t + H + \Delta_{\text{tol}}$ has elapsed without any reported trade in the target window. The forecast is finalized as unavailable and excluded from model error statistics.

---

## 6. Realized Volatility Classification Protocol

Realized volatility outcomes are determined from actual realized price movements:
- Continuous percentage excursion:
  $$\Delta_{\text{pct}} = \frac{|P_{\text{actual}} - P_{\text{origin}}|}{P_{\text{origin}} + \epsilon} \times 100$$
- Discrete classification:
  - **`LOW`:** $\Delta_{\text{pct}} \le 2.0\%$ (stable market regime)
  - **`MEDIUM`:** $2.0\% < \Delta_{\text{pct}} \le 6.0\%$ (moderate seasonal variation)
  - **`HIGH`:** $\Delta_{\text{pct}} > 6.0\%$ (high market volatility / supply shock)

---

## 7. Matched Non-Inferiority Benchmarking

Candidate models are evaluated against the standard **Calendar Persistence Baseline** (predicting zero price change, $P_{t+H} = P_t$) using strictly identical matched origins:

$$\Delta_{\text{MAE}} = \text{MAE}_{\text{Model}} - \text{MAE}_{\text{Persistence}}$$

### Promotion Gate Criteria:
1. **Sample Size:** Minimum 30 prospective evaluated observations across distinct APMC mandis.
2. **Deficit Check:** $\Delta_{\text{MAE}} \le 0.0$ (model MAE must not exceed persistence).
3. **Win Rate:** Percentage of origins where $\text{AE}_{\text{Model}} \le \text{AE}_{\text{Persistence}}$ must exceed **50.0%**.
4. **Current Status:**
   - Rice 14d/30d: **Under Shadow Evaluation** (Opt-In / Experimental).
   - Wheat 14d/30d: **Under Shadow Evaluation** (Experimental).
   - Potato 14d/30d: **Under Shadow Evaluation** (Experimental).
   - Tea 1d: **Production Validated** (Short-Term).

---

## 8. Farmer Safety & Advisory Isolation

Under strict governance rules:
- **Zero Advisory Contamination:** Unvalidated experimental horizon forecasts are never rendered on farmer-facing advisory cards.
- **Zero Silent Fallback:** If a requested commodity or horizon artifact is unavailable, an explicit HTTP 503 error is emitted; the system never substitutes an unvalidated model.
- **Operational Switch Isolation:** Shadow evaluation can run continuously with switches enabled (`AGROVISION_ENABLE_{COMMODITY}_SHADOW=true`) while public client horizon endpoints remain disabled.

---

## 9. Validation Sign-off

The prospective shadow monitoring infrastructure has been verified across all four commodities with complete test coverage in `test_multi_commodity_shadow_tea.py`, `test_shadow_evaluation.py`, and `test_volatility_shadow.py`.

**Verdict:** `SHADOW VALIDATION READY`
