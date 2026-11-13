# Member 2 Volatility V3 — Operational Runbook & Shadow Monitoring Manual

**Module:** Member 2 — Price Volatility Machine Learning  
**Candidate Model:** `rice_volatility_v3_14d_model.pkl`  
**Production Default:** Calendar Persistence (14D)  
**System Role:** Isolated Prospective Shadow Monitoring Pipeline  
**Status:** **ACTIVE SHADOW PIPELINE** (Zero Farmer-Facing Exposure)

---

## 1. System Overview & Production Protection Guarantees

The **Volatility V3 Prospective Shadow Monitoring System** runs alongside the active production advisory without altering public routing, farmer endpoints, or existing models.

```
                    ┌──────────────────────────────────────────────┐
                    │       Ingested Daily Mandi Observation       │
                    └──────────────────────┬───────────────────────┘
                                           │
                    ┌──────────────────────┴───────────────────────┐
                    │    Active Production Routing (UNCHANGED)     │
                    │         Calendar Persistence (14D)           │
                    │     (or Normalized V2 on explicit opt-in)    │
                    └──────────────────────┬───────────────────────┘
                                           │ (Farmer-Facing Advisory)
                                           ▼
                                    [PUBLIC ADVISORY]
                                           
                    ┌──────────────────────────────────────────────┐
                    │     Shadow Evaluation Worker (Asynchronous)  │
                    │       V3 14D XGBoost (Frozen Champion)       │
                    │   - 38 Point-in-Time Features (History <= t) │
                    │   - Raw & Isotonic Calibrated Probabilities  │
                    │   - History Pre-Validation (N >= 14 obs)     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │  SQLite Store: shadow_volatility_forecasts   │
                    │   - forecast_id (SHA256 deterministic UUID)  │
                    │   - origin_date, origin_price, horizon=14    │
                    │   - persistence_pred_class                   │
                    │   - v3_pred_class, v3_calibrated_proba       │
                    │   - status = 'PENDING'                       │
                    │   - target_window_end = origin_date + 14d    │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           │ Wait Strictly >= 14 Calendar Days
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │       Nightly Delayed Outcome Resolution     │
                    │   - Query trades in (t, t + 14]              │
                    │   - Compute forward excursion anchored by Pt │
                    │   - Classify (LOW <= 0.014%, HIGH > 2.667%)  │
                    │   - Status -> 'EVALUATED' or 'WINDOW_EMPTY'  │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Internal Governance Dashboard & Gate Check │
                    │   - Pooled & Per-Mandi Macro F1              │
                    │   - Paired Cluster t-stat & Bootstrap 95% CI │
                    │   - Predefined Margin Non-Inferiority Test   │
                    │   - Rejection of p > 0.10 alone              │
                    └──────────────────────────────────────────────┘
```

### Invariant Rules
1. **Zero Exposure:** Farmer-facing `/predict` endpoint defaults strictly to Persistence (`v1`) and Normalized V2 (`v2_normalized`) upon explicit request. V3 predictions are never returned to farmers or public APIs.
2. **Frozen Artifacts:** `rice_volatility_v3_14d_model.pkl` and `rice_volatility_v3_14d_config.json` are frozen in `artifacts/v3/`. All production model hashes remain 100% bit-for-bit identical.
3. **Idempotency:** Forecast IDs are generated as:
   $$\text{forecast\_id} = \text{SHA256}(\text{market} \mathbin{\Vert} \text{variety} \mathbin{\Vert} \text{origin\_date} \mathbin{\Vert} 14 \mathbin{\Vert} \text{"rice\_volatility\_v3\_14d\_v1"})$$
   Repeated ingestion of the same observation is an idempotent no-op.
4. **Delayed Outcome Resolution:** No forecast is evaluated before its full 14-calendar-day window has elapsed ($T_{current} \ge \text{origin\_date} + 14\text{ days}$).
5. **Safe History Validation:** Explicit history validation enforces $\ge 14$ valid historical observations. Degraded histories cannot trigger tree all-NaN default false alarms.

---

## 2. Configuration & Operational Controls

Shadow monitoring is governed by environment variables and programmatically accessible switches:

| Variable | Default | Purpose |
|---|---|---|
| `AGROVISION_ENABLE_VOLATILITY_SHADOW` | `false` | Master operational switch for volatility shadow prediction and evaluation. |
| `AGROVISION_SHADOW_VOLATILITY_ONLY` | `false` | Convenience flag ensuring public prediction routes remain frozen while shadow runs. |
| `AGROVISION_SHADOW_DB_PATH` | `data/processed/shadow_evaluation.db` | Path to persistent SQLite database. |

To activate shadow monitoring in production/staging environments:
```bash
export AGROVISION_ENABLE_VOLATILITY_SHADOW=true
export AGROVISION_SHADOW_DB_PATH="/path/to/data/processed/shadow_evaluation.db"
```

---

## 3. Database Schema

The pipeline uses three structured tables in SQLite (WAL mode):

### 3.1 `shadow_volatility_forecasts`
```sql
CREATE TABLE IF NOT EXISTS shadow_volatility_forecasts (
    forecast_id TEXT PRIMARY KEY,
    market TEXT NOT NULL,
    variety TEXT NOT NULL,
    origin_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL DEFAULT 14,
    origin_price REAL NOT NULL,
    persistence_pred_class TEXT NOT NULL,
    v3_pred_class TEXT NOT NULL,
    v3_prob_low REAL NOT NULL,
    v3_prob_medium REAL NOT NULL,
    v3_prob_high REAL NOT NULL,
    v3_calibrated_prob_low REAL NOT NULL,
    v3_calibrated_prob_medium REAL NOT NULL,
    v3_calibrated_prob_high REAL NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'rice_volatility_v3_14d_v1',
    created_at TEXT NOT NULL,
    target_window_end TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('PENDING', 'EVALUATED', 'WINDOW_EMPTY')),
    matched_trades_count INTEGER,
    actual_min_price REAL,
    actual_max_price REAL,
    actual_pct_range REAL,
    actual_class TEXT CHECK(actual_class IN ('LOW', 'MEDIUM', 'HIGH', NULL)),
    evaluated_at TEXT,
    cohort TEXT NOT NULL DEFAULT 'PROSPECTIVE'
);

CREATE INDEX IF NOT EXISTS idx_shadow_vol_status 
ON shadow_volatility_forecasts (status, target_window_end);

CREATE INDEX IF NOT EXISTS idx_shadow_vol_market 
ON shadow_volatility_forecasts (market, cohort, status);

CREATE INDEX IF NOT EXISTS idx_shadow_vol_origin 
ON shadow_volatility_forecasts (origin_date, horizon_days);
```

### 3.2 `mandi_observations`
Stores daily raw incoming mandi records (`market`, `variety`, `reported_date`, `modal_price`, `arrivals`, `created_at`).

### 3.3 `shadow_volatility_audit_log`
Tracks nightly job executions, timestamp, count of generated, evaluated, empty, and pending records, plus execution duration and error details.

---

## 4. Daily Operational CLI Commands

A complete command-line runner is located at `member2-price-volatility/experiments/v3/run_volatility_shadow_pipeline.py`.

### 4.1 Ingest Observations & Generate Shadow Forecasts
```bash
python experiments/v3/run_volatility_shadow_pipeline.py \
  --ingest data/incoming/daily_mandi_feed_latest.csv \
  --predict \
  --cohort PROSPECTIVE \
  --force
```

### 4.2 Run Delayed Outcome Evaluation (Nightly Cron Job)
```bash
python experiments/v3/run_volatility_shadow_pipeline.py \
  --evaluate \
  --as-of-date $(date +%Y-%m-%d)
```

### 4.3 Generate Internal Governance Dashboard & Report
```bash
python experiments/v3/run_volatility_shadow_pipeline.py \
  --report \
  --cohort PROSPECTIVE \
  --output-report reports/volatility_shadow_monitoring_report.md
```

---

## 5. Internal Observability API Endpoints

The FastAPI server provides authenticated internal observability endpoints:

### 5.1 `GET /shadow/volatility/status`
Returns real-time operational status, model versions, active production default, and forecast counts:
```json
{
  "status": "ok",
  "shadow_volatility_enabled": true,
  "model_version": "rice_volatility_v3_14d_v1",
  "production_default": "Calendar Persistence (14D)",
  "cohort": "PROSPECTIVE",
  "counts": {
    "total_forecasts": 137,
    "pending_forecasts": 63,
    "evaluated_forecasts": 74,
    "window_empty_forecasts": 0,
    "participating_mandis": 52
  }
}
```

### 5.2 `GET /shadow/volatility/metrics?cohort=PROSPECTIVE`
Returns full comparative performance:
- Pooled Macro F1, Accuracy, Balanced Accuracy
- Per-class Precision, Recall, and F1
- HIGH volatility recall, precision, and critical HIGH-to-LOW error counts
- Multiclass Brier score and class-wise ECE
- Per-mandi win rates and cluster-robust $t$-test / bootstrap confidence intervals
- Promotion governance gate assessment and final verdict

### 5.3 `POST /shadow/volatility/evaluate`
Triggers delayed outcome resolution for all pending forecasts whose 14-day window has elapsed.

---

## 6. Promotion Governance & Statistical Decision Rule

Promotion from shadow monitoring to active production consideration is strictly gated by 6 objective criteria:

| Gate | Requirement | Operational Rationale |
|---|---|---|
| **1. Prospective Duration** | $\ge 90$ calendar days | Ensures observation across seasonal cycles and market shifts. |
| **2. Prospective Sample Size** | $\ge 2,000$ completed forecasts | Ensures sufficient statistical power for cluster disaggregation. |
| **3. Market Coverage** | $\ge 35$ reporting mandis | Guarantees representation across small, medium, and large volume tiers. |
| **4. Macro ECE** | $\le 5.0\%$ | Guarantees that predicted class probabilities are reliable for risk management. |
| **5. Market Win Rate** | $\ge 50.0\%$ of mandis | Prevents pooled metrics from masking widespread deficits in individual markets. |
| **6. Corrected Non-Inferiority Test** | $\text{CI}_{lower} > -\delta_{margin}$ AND $\bar{\Delta} \ge 0$ | Proves ML is not inferior to Persistence within predefined margin $\delta_{margin} = 0.01$. |

> [!WARNING] **Statistical Rule on Non-Inferiority**
> An insignificance result ($p > 0.10$) by itself **does NOT prove non-inferiority**.  
> The lower bound of the clustered 95% bootstrap confidence interval for $\bar{\Delta} = \text{MacroF1}_{V3} - \text{MacroF1}_{Persistence}$ must strictly exceed $-\delta_{margin}$.  
> Any failure or inconclusive evidence results in **CONTINUE_SHADOW_MONITORING**, not automatic promotion.
