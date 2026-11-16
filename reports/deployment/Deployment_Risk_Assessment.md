# AgroVision-ML — Staging Deployment Risk Assessment & Mitigation Framework

**Milestone:** Milestone 2: Real-World Integration, APMC Feed, and Staging Deployment Readiness  
**Target Environment:** Staging / Pre-Production  
**Date:** November 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

Operating an automated agricultural intelligence and price forecasting system in real-world staging environments involves operational, statistical, and infrastructural risks. Agricultural market reporting in India is inherently decentralized, subject to sudden reporting halts, local market holidays, spelling variances, and API quota limits.

This Risk Assessment establishes a formal risk taxonomy, evaluates impact and likelihood, defines programmatic mitigation controls, and outlines step-by-step disaster recovery and rollback procedures.

---

## 2. Risk Matrix Taxonomy

| ID | Risk Category | Risk Scenario | Probability | Impact | Risk Level |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **R-01** | Feed Reliability | Upstream `data.gov.in` downtime or connectivity timeout | High | Medium | **MEDIUM** |
| **R-02** | Rate Limiting | HTTP 429 quota exhaustion from OGD portal | Medium | Low | **LOW** |
| **R-03** | Schema Drift | Upstream changes column headers or date formatting | Medium | High | **HIGH** |
| **R-04** | Stale Reporting | APMC mandi fails to report for > 14 consecutive days | High | Medium | **MEDIUM** |
| **R-05** | Distribution Drift | Sudden supply shock shifts prices beyond 3 standard deviations | Low | High | **HIGH** |
| **R-06** | Data Fabrication | Ingestion engine attempts to interpolate missing daily reports | Low | Critical | **CRITICAL** |
| **R-07** | Concurrency | SQLite database file contention during simultaneous writes | Low | Medium | **LOW** |
| **R-08** | Artifact Integrity| Model binary corruption or missing dependency file | Very Low | Critical | **HIGH** |
| **R-09** | Advisory Safety | Premature exposure of experimental 14d/30d models to farmers | Medium | Critical | **CRITICAL** |
| **R-10** | Security | API credentials or tokens leaked in exception responses | Low | High | **HIGH** |

---

## 3. Detailed Risk Analysis & Mitigation Controls

### R-01: Upstream Mandi Feed Downtime
- **Mechanism:** The National Informatics Centre (NIC) or `data.gov.in` platform undergoes maintenance or suffers temporary outages.
- **Mitigation Control:**
  1. Requests use a strict 15.0-second timeout window.
  2. Failure to connect logs an audit record with status `CONNECTION_ERROR` and raises `DataConnectorNetworkError`.
  3. Cached historical observations in local SQLite database remain untouched and continue to serve inference pipelines.
- **Recovery:** Ingestion retries on next scheduled cron cycle; historical data continuity is maintained via idempotent primary keys.

### R-02: Upstream API Rate Limiting (HTTP 429)
- **Mechanism:** Exceeding portal request threshold (2.0 RPS) triggers HTTP 429 Too Many Requests.
- **Mitigation Control:**
  1. Programmatic token bucket rate limiter clamps outgoing requests to $\le 2.0$ req/sec.
  2. Built-in exponential backoff retries up to 3 times ($1.0\text{s} \to 2.0\text{s} \to 4.0\text{s}$).
- **Recovery:** If retries are exhausted, the batch pauses without corrupting state and resumes on next window.

### R-03: Upstream Schema Drift & New Mandi Aliases
- **Mechanism:** Upstream adds new unmapped mandis or alters date formatting.
- **Mitigation Control:**
  1. Date parsing uses mixed-format parsing (`pd.to_datetime(..., format="mixed")`) supporting ISO, DD/MM/YYYY, and MM/DD/YYYY.
  2. New/unmapped mandis are recorded explicitly in `unmapped_markets` and preserved in the audit log rather than dropped silently.
- **Recovery:** Data engineering reviews `unmapped_markets` from `/v2/ingestion/status/{commodity}` and appends aliases to canonical dictionaries.

### R-04: Stale Feed & Delayed Mandi Reporting
- **Mechanism:** Local market strikes or regional festivals lead to zero reports for over a week.
- **Mitigation Control:**
  1. `audit_commodity_freshness_and_drift()` calculates reporting calendar lag: $\Delta_{\text{lag}} = T_{\text{today}} - T_{\text{max\_obs}}$.
  2. Feeds with $\Delta_{\text{lag}} > 14$ days trigger `is_stale: True` and register alerts in the `ObservabilityEngine`.
  3. Inference pipelines reject stale history when history window requirements cannot be met.
- **Recovery:** System automatically clears stale status upon ingestion of new valid daily reports.

### R-05: Statistical Distribution Drift
- **Mechanism:** Drought, flood, or tariff policy causes prices to spike outside historical training distributions.
- **Mitigation Control:**
  1. Automated drift detection compares batch mean and median against locked training baselines (e.g., Rice mean 2,120.40, Std 185.30).
  2. Deviations exceeding $3.0\sigma$ or $3.0 \times \text{IQR}$ flag `distribution_drift_detected: True`.
  3. Prediction intervals widen automatically via empirical holdout residual distributions.
- **Recovery:** Drift alerts trigger review by the modeling team for retraining consideration under version governance.

### R-06: Data Fabrication / Zero-Movement Substitution
- **Mechanism:** Inadvertent feature engineering interpolating missing daily reports as zero price change.
- **Mitigation Control:**
  1. Strict code invariants in `multi_commodity_ingestion.py` prohibit filling missing dates with dummy prices.
  2. Gaps remain true calendar gaps; models utilize actual historical trading sessions.
  3. Regression test `test_no_zero_movement_interpolation` guarantees compliance.
- **Recovery:** If detected, corrupted rows are purged using primary key timestamps.

### R-07: Concurrency & Database Locking
- **Mechanism:** Simultaneous writes from concurrent ingestion tasks and shadow evaluations lock SQLite.
- **Mitigation Control:**
  1. SQLite configured with Write-Ahead Logging (`PRAGMA journal_mode=WAL;`).
  2. Thread-safe re-entrant locks (`threading.RLock`) in memory collectors.
  3. Isolated database files per commodity (`shadow_evaluation_{commodity}.db`).
- **Recovery:** WAL handles concurrent reads with zero blocking; write contention times out with automated retry.

### R-08: Model Artifact Integrity & Corruption
- **Mechanism:** Disk failure or accidental modification corrupts model `.pkl` or `.json` artifacts.
- **Mitigation Control:**
  1. SHA-256 checksum baseline (`baseline_sha256_checksums.json`) locks all 63 protected artifacts.
  2. Startup verification validates file existence and hashes before serving requests.
  3. Missing artifacts fail fast with explicit HTTP 503 errors; silent fallback between commodities is strictly prohibited.
- **Recovery:** Restore bit-for-bit identical binary artifacts from Git release tag `v2.0.0-staging`.

### R-09: Experimental Model Exposure to Farmers
- **Mechanism:** An experimental 14-day or 30-day Wheat/Potato forecast is displayed on a public advisory interface.
- **Mitigation Control:**
  1. Unified predictor returns `is_experimental: True` and `release_status: "experimental"` on all non-production horizons.
  2. Member 3 API contract explicitly disallows displaying experimental forecasts on farmer cards.
  3. Operational master switch `AGROVISION_ENABLE_HORIZONS=false` instantly disables all horizon endpoints across the platform.
- **Recovery:** Flip environment switch to disable horizon endpoints without stopping short-term advisory services.

### R-10: Credential & API Token Leakage
- **Mechanism:** Upstream error dumps query parameters containing the OGD API key into logs or HTTP response bodies.
- **Mitigation Control:**
  1. `DataGovInMultiCommodityConnector` strips API keys from exception strings using regex replacement (`*REDACTED_DATA_GOV_IN_KEY*`).
  2. Automated test `test_v2_security_no_api_key_leakage` validates that responses never contain secret strings.
- **Recovery:** Immediate API key revocation and reissue via `data.gov.in` portal; update `.env.staging`.

---

## 4. Disaster Recovery & Rollback Procedures

### 4.1 Immediate Rollback Procedure
If staging exhibits critical stability issues or data corruption:
```bash
# 1. Stop staging service
kill $(pgrep -f "uvicorn api_v2:app")

# 2. Reset codebase to verified release tag
git checkout tags/v2.0.0-staging

# 3. Verify artifact hashes
python -c "import json,hashlib; d=json.load(open('member2-price-volatility/artifacts/baseline_sha256_checksums.json')); fails=[k for k,v in d.items() if hashlib.sha256(open('member2-price-volatility/artifacts/'+k,'rb').read()).hexdigest()!=v]; print('FAIL COUNT:', len(fails))"

# 4. Restart in safe mode with horizon endpoints disabled
export AGROVISION_ENABLE_HORIZONS=false
./staging/start_staging.sh
```

### 4.2 Database Recovery Procedure
If an isolated shadow or ingestion database suffers corruption:
1. SQLite WAL mode checkpoints automatically on close.
2. If corrupted, delete the specific commodity database file (e.g., `shadow_evaluation_wheat.db`).
3. Re-run `init_commodity_shadow_db("wheat")` to recreate schema cleanly.
4. Point-in-time forecasts can be regenerated deterministically using `generate_commodity_shadow_forecasts` from raw mandi observations.

---

## 5. Assessment Sign-off

All 10 critical operational risks have been addressed with verified programmatic mitigation controls, robust error boundaries, and verified disaster recovery procedures.

**Verdict:** `RISK MITIGATIONS VERIFIED`
