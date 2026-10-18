# AgroVision-ML — Live Shadow-Mode Evaluation Readiness Report
## Rice Price 14-Day and 30-Day Forecasting Models

**Module:** `member2-price-volatility`  
**Author:** AgroVision ML Platform Engineering  
**Status:** COMPLETE & VERIFIED  
**Date:** 2026-10-17  
**Test Suite:** 86/86 Passing (100% Pass Rate across Price, Volatility, Horizons, and Shadow Evaluation)  
**Frozen Production Model Integrity:** SHA-256 Verified (`b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993`)

---

## 1. Executive Summary & Purpose

Following the development, independent auditing, and opt-in API integration of the 14-day and 30-day rice modal price forecasting models, this report establishes the **Live Shadow-Mode Evaluation Architecture**.

Shadow-mode evaluation provides an isolated, production-safe environment to continuously assess horizon model performance against genuine subsequent mandi observations without exposing experimental forecasts to end users or impacting production decision workflows.

### Core Architectural Guarantees
1. **Production Invariant Protection:** The original production 1–7 day price model (`artifacts/rice_price_model.pkl`) and its 48-feature configuration remain completely untouched and unmodified.
2. **Fresh Installation Opt-In Default:** All horizon forecasting switches default to `false` in fresh installations, ensuring that experimental models are never exposed to external clients without deliberate administrator action.
3. **Decoupled Internal Shadow Evaluation:** Shadow evaluation can operate independently via `AGROVISION_ENABLE_SHADOW_EVALUATION=true` while public-facing client endpoints (`POST /predict-horizon`) remain disabled (`503 Service Unavailable`).
4. **Point-in-Time Temporal Safety:** Shadow forecasts are generated strictly using observations available on or before the forecast origin date ($t \le t_{\text{origin}}$), ensuring zero lookahead leakage.
5. **Delayed Outcome Matching:** Real-world mandi observations arriving days or weeks later are ingested into a transactional SQLite database. The matching engine evaluates forecasts only when their target window ($[t + H, t + H + 3]$ calendar days) arrives.
6. **Honest Persistence Benchmarking:** Every shadow forecast is paired against a persistence baseline on the exact same observation, measuring real out-of-sample skill across overall data, flat markets, and moving markets.

---

## 2. Frozen Production Artifact Integrity Verification

Cryptographic SHA-256 checksums were verified before and after the implementation of the shadow-mode evaluation suite:

| Artifact | Path | Documented SHA-256 | Current Verified SHA-256 | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Production Price Model** | `artifacts/rice_price_model.pkl` | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | **MATCH (UNTOUCHED)** |
| **Production Price Config** | `artifacts/rice_price_final_config.json` | `508ad9d8fddab3d0baa65cee1d7f82c546fbc33253bf23d8ec23fe1f57643de7` | `508ad9d8fddab3d0baa65cee1d7f82c546fbc33253bf23d8ec23fe1f57643de7` | **MATCH (UNTOUCHED)** |
| **14-Day Horizon Model** | `artifacts/rice_price_model_14d.pkl` | `56e4c73523f20d20ef353c7aebda86f14a600a9cf186a8369680ee1ea34cf695` | `56e4c73523f20d20ef353c7aebda86f14a600a9cf186a8369680ee1ea34cf695` | **MATCH (VERIFIED)** |
| **30-Day Horizon Model** | `artifacts/rice_price_model_30d.pkl` | `61f6c4491c10d32c5aaee11f4d9b4b0aee03e4811a43d9cbb288001be3e46ca7` | `61f6c4491c10d32c5aaee11f4d9b4b0aee03e4811a43d9cbb288001be3e46ca7` | **MATCH (VERIFIED)** |

**Conclusion:** Neither the production price model, production configuration, nor the newly trained horizon models were modified or retrained.

---

## 3. Data Ingestion Interface & Operational Boundaries

### Ingestion Interface
In real agricultural deployments across India, mandi data is published with irregular reporting cadence, varying holiday schedules, and latency. The shadow evaluation framework provides a clean, robust ingestion interface:

- **Functions:** `ingest_mandi_observations(data, db_path=None)` in `src/shadow_evaluation.py`.
- **Accepted Inputs:**
  - Path to a CSV file (e.g. `data/raw/rice_daily_mandi.csv`).
  - Pandas `DataFrame` containing incoming batches.
- **Normalization:** Automatically standardizes varied column headers:
  - Market: `Market Name`, `market_name`, `market` $\rightarrow$ `market`
  - Variety: `Variety`, `variety` $\rightarrow$ `variety`
  - Date: `Reported Date`, `reported_date`, `date` $\rightarrow$ `reported_date` (ISO `YYYY-MM-DD`)
  - Modal Price: `Modal Price (Rs./Quintal)`, `modal_price`, `price` $\rightarrow$ `modal_price` (Float)
  - Arrivals: `Arrivals (Tonnes)`, `arrivals_tonnes`, `arrivals` $\rightarrow$ `arrivals` (Float)

### Realistic Operating Assumptions
- **No Fabricated Streaming Infrastructure:** The implementation does not simulate fake WebSocket or streaming feeds. Ingestion operates via scheduled batch jobs or periodic sync tasks matching government data publishing schedules (e.g. Agmarknet daily bulletin updates).
- **Deduplication:** Enforced at the database engine level via composite primary key `(market, variety, reported_date)`. Duplicate entries are cleanly updated via `ON CONFLICT DO UPDATE`.

---

## 4. Shadow-Mode Configuration & Deployment Policy

### Operational Switches
Operational policy is controlled via environment variables defined in `src/config.py`:

| Switch / Variable | Default (Fresh Install) | Description | Recommended Operational Setting |
| :--- | :--- | :--- | :--- |
| `AGROVISION_ENABLE_SHADOW_EVALUATION` | `false` | Controls internal shadow forecast generation and delayed evaluation. | **`true`** (Allows internal monitoring) |
| `AGROVISION_ENABLE_HORIZON_FORECASTING` | `false` | Master public switch for `POST /predict-horizon`. | **`false`** until shadow validation is complete |
| `AGROVISION_ENABLE_HORIZON_14D` | `false` | Enables 14-day model on client API. | **`false`** (Opt-in canary when ready) |
| `AGROVISION_ENABLE_HORIZON_30D` | `false` | Enables 30-day model on client API. | **`false`** (Experimental only) |
| `AGROVISION_SHADOW_DB_PATH` | `data/processed/shadow_evaluation.db` | Path to persistent SQLite evaluation store. | Standard disk volume |

### Precedence & Decoupling Architecture
```
                         +-----------------------------------+
                         | Incoming Mandi Batch Observations |
                         +-----------------+-----------------+
                                           |
                                           v
                       +---------------------------------------+
                       |   ingest_mandi_observations()         |
                       | (Normalized SQLite Database Storage)  |
                       +-------------------+-------------------+
                                           |
             +-----------------------------+-----------------------------+
             |                                                           |
             v                                                           v
  [Client-Facing API Layer]                                  [Internal Shadow Engine]
  Endpoint: POST /predict-horizon                            Module: src/shadow_evaluation.py
  Controlled by:                                             Controlled by:
  - AGROVISION_ENABLE_HORIZON_FORECASTING                    - AGROVISION_ENABLE_SHADOW_EVALUATION
  - AGROVISION_ENABLE_HORIZON_14D                            
  - AGROVISION_ENABLE_HORIZON_30D                            
             |                                                           |
             |                                                           |
    IF FALSE: Returns HTTP 503                                 IF TRUE: Generates & stores
    (Client requests safely rejected)                          point-in-time forecasts; evaluates
                                                               outcomes when observations arrive.
```

### Emergency Rollback Procedures
1. **Immediate Client Blackout:** Set `AGROVISION_ENABLE_HORIZON_FORECASTING=false`. Client requests to `/predict-horizon` immediately fail fast with `503 Service Unavailable` without impacting `/predict` or server uptime.
2. **Selective Horizon Rollback:** Set `AGROVISION_ENABLE_HORIZON_30D=false` while keeping 14-day enabled if only the 30-day model shows drift.
3. **Shadow Engine Pause:** Set `AGROVISION_ENABLE_SHADOW_EVALUATION=false`. Stops background forecast generation and database writes.

---

## 5. Forecast Record Schema & Idempotency Design

### Database Schema
The shadow database resides in SQLite (with Write-Ahead Logging `WAL` mode enabled for concurrency and crash-resilience):

```sql
CREATE TABLE IF NOT EXISTS shadow_forecasts (
    forecast_id TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    variety TEXT NOT NULL,
    origin_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    origin_modal_price REAL NOT NULL,
    predicted_price_change REAL NOT NULL,
    predicted_modal_price REAL NOT NULL,
    requested_target_date TEXT NOT NULL,
    target_window_start TEXT NOT NULL,
    target_window_end TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL, -- 'PENDING', 'EVALUATED', 'TARGET_UNAVAILABLE'
    matched_observation_date TEXT,
    matched_actual_price REAL,
    actual_horizon_offset INTEGER,
    absolute_error REAL,
    percentage_error REAL,
    persistence_pred_price REAL,
    persistence_absolute_error REAL,
    persistence_percentage_error REAL,
    evaluated_at TEXT,
    finalization_reason TEXT
);
```

### Deterministic Idempotency
To prevent duplicate forecast entries when batch jobs run multiple times over the same historical data, every forecast record receives a deterministic unique hash:
$$\text{forecast\_id} = \text{SHA256}(\text{market} \mathbin{\Vert} \text{variety} \mathbin{\Vert} \text{origin\_date} \mathbin{\Vert} \text{horizon\_days} \mathbin{\Vert} \text{model\_version})[:16]$$
Repeated runs detect existing `forecast_id` records and skip duplicate prediction computations.

---

## 6. Delayed Outcome Matching & Finalization Policy

### Matching Window Semantics
Consistent with the training target generation policy defined in `src/horizon_targets.py`:
- For an origin date $t$ and horizon $H \in \{14, 30\}$, the requested target date is $t + H$.
- The allowable observation window is $[t + H, t + H + 3]$ calendar days (tolerance of up to 3 days to accommodate weekend/holiday mandi closures).
- When newly reported observations arrive, the matching query selects the **first chronologically valid observation** within the window for the corresponding `(market, variety)` pair.

### State Transition Lifecycle
```
                 +-----------------------+
                 | Forecast Generated    |
                 | status = 'PENDING'    |
                 +-----------+-----------+
                             |
         +-------------------+-------------------+
         | (Target window not yet elapsed)       | (Target window elapsed)
         v                                       v
+------------------+             +-------------------------------+
| Remains PENDING  |             | Search for mandi observation  |
| (Awaiting dates) |             | in [t + H, t + H + 3]         |
+------------------+             +---------------+---------------+
                                                 |
                                 +---------------+---------------+
                                 |                               |
                     (Observation found)            (No observation found)
                                 v                               v
                     +-----------------------+       +----------------------------+
                     | status = 'EVALUATED'  |       | status =                   |
                     | Records actual price, |       |   'TARGET_UNAVAILABLE'     |
                     | actual offset (14-17  |       | Window expired with no     |
                     | or 30-33), & errors.  |       | observation reported.      |
                     +-----------------------+       +----------------------------+
```

This strict tripartite lifecycle guarantees:
1. Predictions are **never prematurely evaluated** before their target dates arrive.
2. Market closures or missing data do **not distort model metrics** with artificial zero errors.
3. Actual calendar offsets are explicitly audited ($14, 15, 16, 17$ for 14d; $30, 31, 32, 33$ for 30d).

---

## 7. Shadow Evaluation Live Metrics & Persistence Comparison

Metrics are computed via `compute_shadow_evaluation_metrics()` and surfaced via the CLI and `GET /shadow/metrics`.

### Evaluated Metrics
- **Mean Absolute Error (MAE):** $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- **$R^2$ Score:** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Mean Absolute Percentage Error (MAPE):** $\frac{100\%}{N} \sum \frac{|y_i - \hat{y}_i|}{y_i}$
- **Within $\pm ₹25$/qtl Accuracy:** Fraction of forecasts whose absolute error is $\le ₹25.00$
- **Median Error ($p_{50}$) & 90th Percentile Error ($p_{90}$):** Robust quantile spread.
- **Market Stratification:**
  - **Flat Market Days:** Mandi observations where $y_{\text{actual}} = y_{\text{origin}}$ (zero price change).
  - **Movement Market Days:** Mandi observations where $y_{\text{actual}} \ne y_{\text{origin}}$.

### Live Shadow Demonstration Verification
Running `python member2-price-volatility/experiments/run_shadow_evaluation.py --demo` verified the complete end-to-end flow using historical mandi records:
- Ingestion of 2,425 mandi records into SQLite.
- Point-in-time generation of 14d and 30d shadow forecasts at origin `2023-03-01`.
- Delayed evaluation matching outcomes at `2023-03-15` (14d offset = 14) and `2023-04-03` (30d offset = 33).
- Paired comparison against matched Persistence baselines.

---

## 8. Promotion and Rollback Decision Framework

To ensure disciplined ML platform operations, promote or roll back models according to the following objective gates:

```
                            [ MODEL MATURITY GATES ]

       STAGE 0                 STAGE 1                  STAGE 2                STAGE 3
+-------------------+   +--------------------+   +-------------------+   +-------------------+
| Shadow Evaluation |-->| Canary / Opt-In    |-->| General Opt-In    |-->| Default Horizon   |
| (Disabled in API) |   | (Selected Mandis)  |   | (Full Catalog)    |   | (Production Route)|
+-------------------+   +--------------------+   +-------------------+   +-------------------+
```

### Promotion Criteria

#### For 14-Day Model (Current State: Stage 1 Opt-In Ready)
1. **Sample Size Gate:** Minimum 500 shadow evaluations across at least 10 distinct mandis over 90 calendar days.
2. **Skill Gate:** Out-of-sample shadow MAE must be $\le$ Persistence MAE overall.
3. **Movement Advantage Gate:** Shadow MAE on movement days must outperform Persistence MAE by $\ge 3.0\%$.
4. **Tail Error Gate:** 90th percentile error must be lower than or equal to Persistence $p_{90}$.
5. **Stability Gate:** $0\%$ unhandled inference exceptions or schema formatting errors in production logs.

#### For 30-Day Model (Current State: Stage 0 / Experimental)
1. **Sample Size Gate:** Minimum 1,000 shadow evaluations across at least 15 distinct mandis over 180 calendar days.
2. **Variance Reduction Gate:** Out-of-sample RMSE on movement days must outperform Persistence by $\ge 4.0\%$.
3. **Flat Market Parity Gate:** Flat market MAE penalty relative to persistence must be $\le ₹1.50$/quintal.
4. **Experimental Disclaimer:** Must retain `release_status="experimental"` and operational warning in all API client responses until Stage 2 is achieved.

### Automatic Rollback Triggers
An immediate rollback to `disabled` status must be triggered if:
- **Model Drift / Degradation:** 14-day shadow MAE exceeds persistence MAE by $> 5.0\%$ over any rolling 30-day window ($N \ge 50$).
- **Extreme Tail Failure:** Maximum absolute prediction error exceeds $₹500$/quintal in any single forecast.
- **Data Gap Exceedance:** `TARGET_UNAVAILABLE` rate exceeds $35\%$ of total generated forecasts, indicating chronic mandi reporting breakdown.
- **Latency Violation:** 95th percentile inference latency exceeds 250ms under standard operational load.

---

## 9. Verification & Test Suite Execution Results

The full test suite across the entire `member2-price-volatility` repository was executed:

```bash
python -m pytest member2-price-volatility/tests/ -v
```

### Test Suite Summary (86 Tests Passing)
- **`test_shadow_evaluation.py` (9 tests):**
  - `test_fresh_installation_defaults_are_disabled`: Verifies all switches default to `false`.
  - `test_disabled_horizon_endpoint_returns_503_while_legacy_returns_200`: Confirms API boundary isolation.
  - `test_shadow_evaluation_can_run_independently_of_client_api`: Proves internal evaluation runs while public API is disabled.
  - `test_ingest_mandi_observations`: Validates batch normalization and upserting.
  - `test_forecast_generation_idempotency`: Verifies deterministic hashing and duplicate avoidance.
  - `test_pending_forecasts_not_evaluated_prematurely`: Ensures forecasts stay `PENDING` before target window opens.
  - `test_forecast_evaluates_when_observation_arrives`: Confirms transition to `EVALUATED` and correct offset recording.
  - `test_closed_window_without_observation_becomes_unavailable`: Tests transition to `TARGET_UNAVAILABLE`.
  - `test_shadow_api_endpoints`: Verifies `GET /shadow/status` and `GET /shadow/metrics`.
- **`test_price_horizons_14d_30d.py` (19 tests):** Horizon models, schemas, targets, purging, parity, switches.
- **`test_evaluation_horizons.py` (8 tests):** Regression metrics, holdout reproduction, frozen artifact invariance.
- **`test_volatility_normalized.py` (8 tests):** Normalized volatility champion loading, leakage, API, bridge.
- **`test_volatility_experiments.py` (6 tests):** Target definitions, purging logic, monotonicity.
- **`test_training_inference_parity.py` (2 tests):** Feature parity, point-in-time invariance.
- **`test_leakage.py` (10 tests):** Temporal and target leakage prevention.
- **`test_inference.py` (5 tests):** End-to-end inference, missing values, ordering.
- **`test_pipeline.py` (2 tests):** Full pipeline orchestration.
- **`test_api.py` (17 tests):** API endpoints and error schemas.

**Total Result: 86 passed in 100.27s (0 failures, 0 skipped).**

---

## 10. Summary of Files Changed and Added

| File | Change Type | Role in Shadow Mode Architecture |
| :--- | :--- | :--- |
| `src/config.py` | Modified | Added `is_shadow_evaluation_enabled()`, `get_shadow_db_path()`, `set_shadow_evaluation_enabled()`. Updated horizon switches to default to `false` for fresh installs. |
| `src/inference.py` | Modified | Updated `predict_horizon()` to support `bypass_switches=True` for internal callers (like shadow evaluation) while keeping public API restricted. |
| `src/shadow_evaluation.py` | **Added** | Core shadow engine: SQLite WAL persistence, observation ingestion, idempotent forecast generator, delayed target matching, and comparative persistence metrics. |
| `src/api.py` | Modified | Added `GET /shadow/status` and `GET /shadow/metrics` observability routes. |
| `experiments/run_shadow_evaluation.py` | **Added** | Standalone CLI runner with `--ingest`, `--generate`, `--evaluate`, `--report`, and `--demo` options. |
| `tests/test_shadow_evaluation.py` | **Added** | Complete unit and integration test suite (9 test cases) verifying all shadow mechanics and safety guarantees. |
| `examples/client_horizon_integration.py` | Modified | Updated to showcase fresh-install disabled state (HTTP 503) followed by operator opt-in activation. |
| `README.md` | Modified | Added Section 33 with CLI instructions, endpoint references, and operational deployment policies. |
| `.gitignore` | Modified | Added ignore patterns for SQLite database files (`*.db`, `*.sqlite3`, `*.db-wal`, `*.db-shm`). |
| `reports/rice_price_shadow_evaluation_readiness.md` | **Added** | This comprehensive platform readiness deliverable. |

---

## 11. Final Certification

The live shadow-mode evaluation architecture is fully implemented, thoroughly tested, and certified ready for operational deployment. It enables the AgroVision ML platform team to rigorously benchmark 14-day and 30-day rice price forecasts against live mandi data in complete isolation from production decision systems.
