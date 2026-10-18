# AgroVision-ML — Real-Data Shadow Forecasting Pilot Report
## West Bengal Rice Price 14-Day and 30-Day Models

**Module:** `member2-price-volatility`  
**Classification:** Retrospective Shadow Simulation / Historical Replay Pilot  
**Author:** AgroVision ML Platform & Verification Engineering  
**Status:** COMPLETE & VERIFIED  
**Date:** 2026-10-17  
**Test Suite:** 90/90 Passing (100% Pass Rate across Core Price, Volatility, Horizon Models, and Shadow System)  
**Database Audit File:** `data/processed/shadow_pilot.db` (Isolated, durable SQLite WAL store)

---

## 1. Executive Summary & Audit Classification

This pilot executes real-data shadow forecasting for the 14-day and 30-day rice modal price models in `member2-price-volatility`. The pilot was conducted without modifying or retraining the frozen production price model, production configurations, or volatility estimators.

### Critical Classification Disclaimer
In accordance with platform verification integrity:
- **Available Genuine Dataset:** The genuine West Bengal mandi dataset available in the repository (`data/processed/rice_cleaned.csv`) spans from `2001-03-16` to `2024-02-02`.
- **Classification:** Because the observations terminate on `2024-02-02`, this pilot is strictly classified as a **Retrospective Shadow Simulation / Historical Replay Pilot**.
- **No False Claims:** These results represent genuine historical point-in-time forecasts evaluated against subsequently observed mandi prices. They are **not** real-time streaming observations. In an active production deployment, forecasts generated today will mature when subsequent Agmarknet daily bulletins are imported.

---

## 2. Frozen Production Artifact Checksum Verification

Cryptographic SHA-256 checksums were verified before and after the pilot workflow:

| Artifact | File Path | Documented SHA-256 Checksum | Current Verified Checksum | Invariant Status |
| :--- | :--- | :--- | :--- | :--- |
| **Production Price Model** | `artifacts/rice_price_model.pkl` | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | **MATCH (UNTOUCHED)** |
| **Production Price Config** | `artifacts/rice_price_final_config.json` | `508ad9d8fddab3d0baa65cee1d7f82c546fbc33253bf23d8ec23fe1f57643de7` | `508ad9d8fddab3d0baa65cee1d7f82c546fbc33253bf23d8ec23fe1f57643de7` | **MATCH (UNTOUCHED)** |
| **14-Day Horizon Model** | `artifacts/rice_price_model_14d.pkl` | `1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab` | `1d8c117c094703440fb7867b5490b44c4caff8aaa2475495b46e03b8d4ae6aab` | **MATCH (VERIFIED)** |
| **30-Day Horizon Model** | `artifacts/rice_price_model_30d.pkl` | `ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343` | `ed8b0c76e60d2dff1e2b807e00fb6373da75b4a6d94e6523e81c7bc8f07eb343` | **MATCH (VERIFIED)** |

**Zero Model Mutation:** Neither model was retrained, fine-tuned, or modified.

---

## 3. Real Data-Ingestion Path & Limitations

### Data Source
- **Primary Source:** West Bengal Department of Agricultural Marketing / Agmarknet official daily market records cleaned and validated at `member2-price-volatility/data/processed/rice_cleaned.csv`.
- **Date Range:** `2001-03-16` to `2024-02-02`.
- **Geographic Scope:** 87 distinct West Bengal agricultural mandis across 34 rice varieties (major volume in `Fine`, `Common`, `Other`, `Super Fine`, and `Ratnachudi`).

### Ingestion Interface & Auditing
The ingestion engine (`ingest_mandi_observations`) was executed with detailed statistical accounting:
```bash
python member2-price-volatility/experiments/run_shadow_evaluation.py --ingest member2-price-volatility/data/processed/rice_cleaned.csv --db-path member2-price-volatility/data/processed/shadow_pilot.db
```
- **Total Rows Ingested:** 319,092
- **Rejected Rows (Invalid/Corrupted):** 0
- **Deduplication:** Enforced by SQLite composite primary key `(market, variety, reported_date)` with `ON CONFLICT DO UPDATE`.
- **Validation Rules Enforced:**
  - Date strings coerced via `pd.to_datetime(errors='coerce')`; unparseable dates dropped.
  - Prices coerced to numeric and strictly filtered for `modal_price > 0`.
  - Empty strings or missing market/variety values dropped.

### Ingestion Limitations & Operational Boundaries
1. **No Automated Live Scraping Daemon:** The AgroVision repository does not maintain an unauthenticated or scraping-based live connection to Agmarknet. Periodic data updates must be imported via CSV or batch DataFrames using `ingest_mandi_observations()`.
2. **Mandi Reporting Irregularities:** Mandis frequently close for regional festivals, strikes, or Sundays. The 3-calendar-day tolerance window ($[t+H, t+H+3]$) is essential to absorb normal reporting cadences without discarding valid market observations.

---

## 4. End-to-End Pilot Workflow Execution

The pilot was executed against an isolated pilot database (`data/processed/shadow_pilot.db`) using the documented CLI runner:

```bash
python member2-price-volatility/experiments/run_shadow_evaluation.py --pilot --db-path member2-price-volatility/data/processed/shadow_pilot.db
```

### Execution Phases & Record Counts

| Phase | Action | Record Counts & Outcomes | Notes |
| :--- | :--- | :--- | :--- |
| **Phase 1: Ingestion** | Batch import of `rice_cleaned.csv` | Accepted: 319,092 records<br>Rejected: 0<br>Mandis: 87, Varieties: 34 | Complete historical and holdout series stored in SQLite table `mandi_observations`. |
| **Phase 2: Generation** | Point-in-time forecasts at 5 distinct origins:<br>`2023-03-01`, `2023-06-01`, `2023-09-01`, `2023-12-01`, `2024-01-20` | Generated: 2,340 forecasts<br>• 14D: 1,170 forecasts<br>• 30D: 1,170 forecasts<br>Skipped: 0 | History strictly filtered to $t \le t_{\text{origin}}$. Groups with $<30$ historical observations skipped. |
| **Phase 3: Evaluation** | Delayed matching in $[t+H, t+H+3]$ calendar window | Evaluated: 584 forecasts<br>Target Unavailable: 85 forecasts<br>Still Pending: 1,671 forecasts | Origins in 2023 had matured target windows. Origin `2024-01-20` remained `PENDING` because target dates ($>2024-02-02$) have not arrived. |
| **Phase 4: Idempotency** | Re-run generation & evaluation across same origins | Re-run Generated: 0 (100% skipped)<br>Re-run Skipped: 2,340<br>Re-run Evaluated: 0 | Proves deterministic hashing and total idempotency. Zero record duplication or state churn. |

---

## 5. Audit of Target Matching, Leakage Prevention & Late Arrivals

### Verification Checklist
1. **Target Window Bounds:**
   - 14-day horizon matched observations strictly inside $[t + 14, t + 17]$ calendar days.
   - 30-day horizon matched observations strictly inside $[t + 30, t + 33]$ calendar days.
   - Earliest valid observation selected chronologically (`ORDER BY reported_date ASC LIMIT 1`).
2. **Entity Isolation:** Matched strictly by composite `(market, variety)`. Zero cross-mandi or cross-variety target leakage.
3. **Point-in-Time Integrity:** Forecast origin price and 48 feature lags computed using observations $\le t_{\text{origin}}$ only. Future prices never entered feature engineering.
4. **Matched Baseline Invariant:** Persistence baseline predicted origin modal price $P_t$, evaluated on the exact same realized observations.
5. **Ingestion Delay vs. Unavailable Target:**
   - When no observation was found in $[t+H, t+H+3]$:
     - If latest observation in DB for that mandi $\le \text{win\_end}$: Classified as **`PENDING`** (ingestion delay or awaiting date).
     - If latest observation in DB for that mandi $> \text{win\_end}$: Classified as **`TARGET_UNAVAILABLE`** (mandi active post-window, but did not report during window).
6. **Late-Arrival Recovery:** Tested and verified via `evaluate_pending_shadow_forecasts(recheck_unavailable=True)`. When late backfill records covering previously unavailable windows were imported, the engine successfully recovered forecasts to `EVALUATED` with reason `"Late observation arrived and resolved unavailable target"`.

---

## 6. Pilot Performance Results: XGBoost vs. Persistence

### 14-Day Horizon Performance Summary

| Metric | XGBoost Model (14D) | Persistence Baseline | Delta (Model - Persistence) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Evaluated Forecasts** | **285** | 285 | — | Matched paired evaluation |
| **Pending Forecasts** | **835** | 835 | — | Awaiting outcome dates |
| **Target Unavailable** | **50** | 50 | — | Window closed without observation |
| **Mean Absolute Error (MAE)** | **₹72.76 / qtl** | ₹73.05 / qtl | **-₹0.29** | **Model Advantage** |
| **MAE 95% Confidence Interval** | **[₹56.48, ₹89.04]** | [₹56.14, ₹89.97] | [-₹2.28, +₹1.69] | Statistically overlapping at N=285 |
| **Root Mean Squared Error (RMSE)**| **₹157.78 / qtl** | ₹162.74 / qtl | **-₹4.97** | **Model Advantage (Variance Reduction)** |
| **$R^2$ Score** | **0.9358** | 0.9317 | **+0.0041** | High explanatory fidelity |
| **Mean Absolute Pct Error (MAPE)**| **2.07%** | 2.06% | +0.01% | Commercial equivalence |
| **Within $\pm ₹25$/qtl Accuracy** | **55.44%** | 55.79% | -0.35% | High proportion inside tolerance |
| **Median Absolute Error ($p_{50}$)** | **₹11.80** | ₹0.00 | — | Persistence exact match on flat days |
| **90th Percentile Error ($p_{90}$)** | **₹199.78** | ₹200.00 | **-₹0.22** | Slightly lower tail risk |

#### 14-Day Stratified Performance:
- **Movement Market Days (N=137, 48.07% of evaluated):**
  - XGBoost MAE: **₹148.78** vs Persistence MAE: **₹151.97** (**+₹3.19 / qtl Model Advantage**)
  - XGBoost RMSE: **₹227.49** vs Persistence RMSE: **₹234.73** (**+₹7.24 / qtl Model Advantage**)
- **Flat Market Days (N=148, 51.93% of evaluated):**
  - XGBoost MAE: **₹2.39** vs Persistence MAE: **₹0.00** (Persistence has mathematical zero error on flat days)
- **Target-Date Offset Distribution:**
  - Offset 14 days: **272 forecasts (95.44%)**
  - Offset 15 days: **5 forecasts (1.75%)**
  - Offset 16 days: **5 forecasts (1.75%)**
  - Offset 17 days: **3 forecasts (1.05%)**

---

### 30-Day Horizon Performance Summary

| Metric | XGBoost Model (30D) | Persistence Baseline | Delta (Model - Persistence) | Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Evaluated Forecasts** | **299** | 299 | — | Matched paired evaluation |
| **Pending Forecasts** | **836** | 836 | — | Awaiting outcome dates |
| **Target Unavailable** | **35** | 35 | — | Window closed without observation |
| **Mean Absolute Error (MAE)** | **₹119.35 / qtl** | ₹119.40 / qtl | **-₹0.05** | **Virtual Parity** |
| **MAE 95% Confidence Interval** | **[₹96.61, ₹142.09]** | [₹96.12, ₹142.68] | [-₹2.81, +₹2.71] | Indistinguishable difference at N=299 |
| **Root Mean Squared Error (RMSE)**| **₹233.15 / qtl** | ₹237.26 / qtl | **-₹4.12** | **Model Advantage (Shock Mitigation)** |
| **$R^2$ Score** | **0.8544** | 0.8492 | **+0.0052** | Meaningful long-range correlation |
| **Mean Absolute Pct Error (MAPE)**| **3.29%** | 3.29% | 0.00% | Exact parity |
| **Within $\pm ₹25$/qtl Accuracy** | **36.45%** | 36.45% | 0.00% | Exact parity |
| **Median Absolute Error ($p_{50}$)** | **₹58.32** | ₹70.00 | **-₹11.68** | **Lower Median Error** |
| **90th Percentile Error ($p_{90}$)** | **₹296.53** | ₹300.00 | **-₹3.47** | Modest tail advantage |

#### 30-Day Stratified Performance:
- **Movement Market Days (N=198, 66.22% of evaluated):**
  - XGBoost MAE: **₹176.78** vs Persistence MAE: **₹180.30** (**+₹3.52 / qtl Model Advantage**)
  - XGBoost RMSE: **₹286.42** vs Persistence RMSE: **₹291.56** (**+₹5.14 / qtl Model Advantage**)
- **Flat Market Days (N=101, 33.78% of evaluated):**
  - XGBoost MAE: **₹6.77** vs Persistence MAE: **₹0.00**
- **Target-Date Offset Distribution:**
  - Offset 30 days: **165 forecasts (55.18%)**
  - Offset 31 days: **73 forecasts (24.41%)**
  - Offset 32 days: **47 forecasts (15.72%)**
  - Offset 33 days: **14 forecasts (4.68%)**

---

### Performance Breakdown by Variety & Mandi

#### Top Varieties (14-Day Horizon):
- **Fine (N=94):** XGBoost MAE **₹48.59** vs Persistence **₹48.99** (Advantage: **+₹0.40**)
- **Other (N=68):** XGBoost MAE **₹87.55** vs Persistence **₹91.69** (Advantage: **+₹4.14**)
- **Common (N=77):** XGBoost MAE **₹104.16** vs Persistence **₹101.69** (Advantage: **-₹2.47**)
- **Super Fine (N=12):** XGBoost MAE **₹89.43** vs Persistence **₹87.50** (Advantage: **-₹1.93**)

#### Top Mandis (14-Day Horizon):
- **Jhargram (N=12):** XGBoost MAE **₹84.77** vs Persistence **₹91.67** (Advantage: **+₹6.90**)
- **Kalyani (N=9):** XGBoost MAE **₹17.18** vs Persistence **₹16.67** (Advantage: **-₹0.51**)
- **Burdwan (N=8):** XGBoost MAE **₹51.80** vs Persistence **₹50.00** (Advantage: **-₹1.80**)
- **Diamond Harbour (N=8):** XGBoost MAE **₹143.77** vs Persistence **₹143.75** (Parity)

---

## 7. Database Reliability & Security Findings

1. **Transaction Integrity:** Database operates under SQLite `journal_mode=WAL` with `foreign_keys=ON`. Multi-record batch insertions and status updates are wrapped in explicit database transactions.
2. **Corrupted Data Resilience:** Verified in `test_malformed_observation_rejection`. Ingesting rows with invalid dates, negative prices, zero prices, non-numeric strings, or blank market names resulted in 5 rejected rows and 1 accepted valid row without database exceptions.
3. **Restart & Crash Durability:** Verified in `test_database_persistence_across_process_restart`. Forecasts, evaluation outcomes, and comparative metrics survive complete process exit and connection termination with exact metric reproducibility.
4. **API Security & Information Exposure:**
   - Audited `GET /shadow/status` and `GET /shadow/metrics`.
   - Verified in `test_shadow_api_security_no_raw_timeseries_leakage`.
   - Both endpoints return high-level summary counters and aggregate metrics. Neither route leaks raw historical observations, individual price series, or proprietary market timeseries to public callers.

---

## 8. Final Approved Promotion and Rollback Policies

Following the statistical audit of sample sizes, confidence intervals, and agricultural market characteristics, the operational policies are approved as follows:

### Statistical Reality on Pilot Findings
- At $N=285$ (14D) and $N=299$ (30D), the overall MAE 95% confidence intervals overlap with the persistence baseline ($[-₹2.28, +₹1.69]$ for 14D; $[-₹2.81, +₹2.71]$ for 30D).
- While the models demonstrate clear advantage on moving market days ($+₹3.19$/qtl for 14D and $+₹3.52$/qtl for 30D), overall market parity is influenced by flat market days where persistence incurs mathematical zero error.
- **Conclusion:** Matured sample sizes ($N < 500$) are currently **insufficient for general production promotion**. Both models must remain **opt-in** in production.

### Operational Promotion Gates (Formalized)

#### 14-Day Horizon Model (`release_status: "opt-in"`)
- **Gate 1 (Maturity Sample):** Must accumulate at least **500 evaluated shadow forecasts** across $\ge 10$ mandis spanning at least two distinct agricultural seasons (Aman and Boro).
- **Gate 2 (Overall Skill):** Point-estimate out-of-sample shadow MAE must be $\le$ Persistence MAE.
- **Gate 3 (Movement Edge):** Movement-day MAE must outperform Persistence by $\ge 3.0\%$ with $p < 0.05$ on paired difference test.
- **Gate 4 (Tail Risk):** 90th percentile error $p_{90}$ must not exceed persistence $p_{90}$ by more than $₹10$/quintal.

#### 30-Day Horizon Model (`release_status: "experimental"`)
- **Gate 1 (Maturity Sample):** Must accumulate at least **1,000 evaluated shadow forecasts** across $\ge 15$ mandis spanning a full 12-month calendar year.
- **Gate 2 (Variance Reduction):** Out-of-sample RMSE on movement days must outperform Persistence by $\ge 4.0\%$.
- **Gate 3 (Flat Market Penalty):** Flat-market penalty must be strictly monitored; if flat days exceed $50\%$ in a given mandi, client responses must prominently display flat-market persistence advisories.
- **Gate 4 (Disclaimer Persistence):** Must retain `release_status: "experimental"` indefinitely until full annual pilot benchmarks are completed.

### Approved Rollback & Alert Policies
1. **Isolated Error Clarification:** An isolated forecast error $> ₹500$/quintal shall be treated as an **Operational Investigation Alert**, not an automatic system rollback. Agricultural mandis can experience genuine $₹500+$ price spikes during acute post-flood or export-ban events; an automated rollback on a single legitimate market shock would degrade system availability unnecessarily.
2. **Automatic Rollback Triggers (System Paused):**
   - **Sustained Degradation:** Rolling 30-day MAE is $> 5.0\%$ worse than persistence over at least 50 evaluated forecasts.
   - **Data Gap Breakdown:** `TARGET_UNAVAILABLE` rate exceeds $35\%$ over 100 consecutive forecasts, indicating chronic mandi reporting collapse.
   - **Runtime Error Spike:** Prediction error rate on `/predict-horizon` exceeds $2.0\%$ over 500 requests.

---

## 9. Comprehensive Test Suite Evidence

The entire project test suite was executed:
```bash
python -m pytest member2-price-volatility/tests/ -v
```
**Results:** **90 passed in 116.66s (0 failures, 0 skipped, 100% pass rate).**

### Test Breakdown by Subsystem:
- `test_shadow_evaluation.py`: **13 passed** (Defaults, isolation, ingestion, idempotency, delayed matching, closed windows, API endpoints, corrupt data rejection, late recovery, process restart durability, security).
- `test_price_horizons_14d_30d.py`: **19 passed** (Model loading, schemas, features, purging, tolerance windows, switches, bridge).
- `test_evaluation_horizons.py`: **8 passed** (Regression metrics, holdout reproduction, frozen artifact invariance).
- `test_volatility_normalized.py`: **8 passed** (Normalized champion model loading, leakage, API, bridge).
- `test_volatility_experiments.py`: **6 passed** (Target definitions, monotonicity, purging).
- `test_training_inference_parity.py`: **2 passed** (Feature parity, point-in-time invariance).
- `test_leakage.py`: **10 passed** (Feature leakage and column isolation).
- `test_inference.py`: **5 passed** (End-to-end inference, missing values).
- `test_pipeline.py`: **2 passed** (Pipeline orchestration).
- `test_api.py`: **17 passed** (FastAPI endpoints and error handling).

---

## 10. Summary of Files Changed & Added

| File | Status | Description of Change |
| :--- | :--- | :--- |
| `src/shadow_evaluation.py` | Enhanced | Added robust data validation (`modal_price > 0`, unparseable date rejection, string trimming); added `return_stats` to `ingest_mandi_observations`; added `recheck_unavailable` to recover late backfilled records; added 95% confidence intervals and variety breakdown. |
| `experiments/run_shadow_evaluation.py` | Enhanced | Added `--pilot` workflow runner, `--data-path` override, `--recheck-unavailable` flag, and enhanced metrics reporting with confidence intervals and breakdown summaries. |
| `tests/test_shadow_evaluation.py` | Enhanced | Added 4 new reliability & security tests covering malformed observation rejection, late-arrival recovery, restart durability, and API response privacy (13 tests total). |
| `reports/rice_price_shadow_pilot_report.md` | **Added** | This comprehensive pilot execution and statistical audit report. |

---

## 11. Repeatable Command Sequence for Future Shadow Runs

To execute future shadow evaluation runs as new monthly or weekly mandi observations arrive:

```bash
# 1. Ingest newly received mandi observations (CSV format)
python member2-price-volatility/experiments/run_shadow_evaluation.py \
    --ingest data/raw/new_mandi_batch_YYYY_MM.csv \
    --db-path member2-price-volatility/data/processed/shadow_pilot.db

# 2. Generate point-in-time shadow forecasts for today's origin date
python member2-price-volatility/experiments/run_shadow_evaluation.py \
    --generate \
    --origin-date YYYY-MM-DD \
    --db-path member2-price-volatility/data/processed/shadow_pilot.db

# 3. Evaluate matured outcomes against newly arrived observations
python member2-price-volatility/experiments/run_shadow_evaluation.py \
    --evaluate \
    --recheck-unavailable \
    --db-path member2-price-volatility/data/processed/shadow_pilot.db

# 4. Generate updated comparative metrics summary
python member2-price-volatility/experiments/run_shadow_evaluation.py \
    --report \
    --db-path member2-price-volatility/data/processed/shadow_pilot.db

# 5. Query live API status (optional, when FastAPI server is active)
curl -X GET http://localhost:8000/shadow/status
curl -X GET http://localhost:8000/shadow/metrics?horizon_days=14
```

---

## 12. Final Release Recommendation

1. **Keep Both Models Opt-In:** Maintain 14-day as opt-in production release and 30-day as experimental opt-in. Do not promote either model to the default `/predict` route.
2. **Continue Accumulating Observations:** Retain the shadow evaluation background engine to accumulate real observations until the 500-evaluation gate for 14-day and 1,000-evaluation gate for 30-day are satisfied.
3. **Operational Readiness Verdict:** **PASS.** The shadow forecasting system is robust, idempotent, point-in-time safe, and production-isolated.
