# AgroVision-ML — Live APMC Mandi Data Ingestion Architecture & Report

**Milestone:** Milestone 2: Real-World Integration, APMC Feed, and Staging Deployment Readiness  
**Target Environment:** Staging / Pre-Production  
**Date:** November 2026  
**Status:** COMPLETE & VERIFIED  

---

## 1. Executive Summary

The Live Data Ingestion Layer bridges AgroVision-ML with real-world Indian agricultural markets, ingesting daily wholesale APMC (Agricultural Produce Market Committee) observations across all four operational commodities:
1. **Rice (Paddy)**
2. **Wheat**
3. **Potato**
4. **Tea**

This pipeline connects to the Open Government Data (OGD) platform (`data.gov.in`) via Agmarknet resource `9ef84268-d588-465a-a308-a864a43d0070`, normalizes non-standard daily mandi records, audits market freshness and distribution drift, and stores observations into an idempotent SQLite persistence ledger.

Strict data integrity guarantees are enforced at every step: missing daily reports are never fabricated, zero-movement price assumptions are strictly prohibited, and API credentials are automatically redacted across all logs and exception handlers.

---

## 2. Ingestion Architecture

```
+-------------------------------------------------------------------------+
|                  Upstream Source: data.gov.in / Agmarknet               |
|            Resource ID: 9ef84268-d588-465a-a308-a864a43d0070           |
+-------------------------------------------------------------------------+
                                    |
                 HTTPS REST API (Rate Limit: 2.0 RPS)
                                    v
+-------------------------------------------------------------------------+
|               DataGovInMultiCommodityConnector                          |
|  - Exponential backoff on HTTP 429 (1.0s -> 2.0s -> 4.0s)               |
|  - Credential redaction (never logged or exposed in exceptions)         |
|  - Commodity filter dispatch (Rice, Wheat, Potato, Tea)                 |
+-------------------------------------------------------------------------+
                                    |
                               Raw JSON/CSV
                                    v
+-------------------------------------------------------------------------+
|             normalize_and_validate_commodity_records()                  |
|  - Mixed date format parsing (ISO-8601, DD/MM/YYYY, MM/DD/YYYY)        |
|  - Modal price validation (> 0 in Rs./Quintal)                          |
|  - Arrivals validation (>= 0 in Tonnes)                                 |
|  - Canonical alias mapping (per-commodity mandi & variety dictionaries) |
|  - In-batch deduplication on (commodity, market, variety, date)         |
|  - Explicit tracking of unmapped aliases in audit ledger                |
|  - Strict Anti-Fabrication Guarantee (zero interpolation)               |
+-------------------------------------------------------------------------+
                                    |
                           Cleaned Observations
                                    v
+-------------------------------------------------------------------------+
|                   Freshness & Drift Auditing                            |
|  - Calendar reporting lag tracking (Current Date - Max Reported Date)   |
|  - Staleness alerting (> 14 days lag flagged as STALE)                  |
|  - Statistical drift detection vs historical training baselines         |
+-------------------------------------------------------------------------+
                                    |
                              Audited Batch
                                    v
+-------------------------------------------------------------------------+
|                  Idempotent Persistence Engine                          |
|  - SQLite WAL mode with ACID transactions                               |
|  - Composite PK: (commodity, market, variety, reported_date)            |
|  - INSERT OR IGNORE / ON CONFLICT deduplication                         |
|  - Comprehensive audit logging in `ingestion_log`                       |
+-------------------------------------------------------------------------+
```

---

## 3. Upstream Feed Configuration

| Parameter | Configuration Value | Description |
| :--- | :--- | :--- |
| **OGD Resource Base** | `https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070` | Current Daily Price of Various Commodities from Various Markets |
| **API Format** | `json` | Structured JSON response payload |
| **Default Rate Limit** | `2.0 req/sec` | Prevents upstream portal quota throttling |
| **Backoff Strategy** | Exponential (base 1.0s, factor 2.0x, max 3 retries) | Graceful handling of HTTP 429 Too Many Requests |
| **Timeout Window** | `15.0 seconds` | Fails fast on upstream connectivity timeouts |
| **Auth Redaction** | `*REDACTED_DATA_GOV_IN_KEY*` | Automated masking of API key in all outputs |

---

## 4. Canonical Mandi & Variety Normalization

Upstream mandi and variety spellings vary widely across states and reporting officials. The ingestion layer applies deterministic canonical alias mapping:

### 4.1 Rice (Paddy)
- **Canonical Markets:** Burdwan, Bolpur, Sainthia, Tarakeswar, Rampurhat.
- **Canonical Varieties:** Common, Fine, Superfine, Swarna, Ratna.
- **Alias Examples:** `"Burdwan (Reg.)"` -> `Burdwan`, `"Burdwan Sadar"` -> `Burdwan`, `"Mota Dhan"` -> `Common`, `"Miniket"` -> `Fine`.

### 4.2 Wheat
- **Canonical Markets:** Bahraich, Bareilly, Aligarh, Karnal, Khanna.
- **Canonical Varieties:** Dara, Sharbati, Deshi, Kalyan, PBW-343.
- **Alias Examples:** `"Bahraich Mandi"` -> `Bahraich`, `"Karnal Grain Market"` -> `Karnal`, `"Dara Wheat"` -> `Dara`.

### 4.3 Potato
- **Canonical Markets:** Agra, Farrukhabad, Burdwan, Hooghly, Jalandhar.
- **Canonical Varieties:** Desi, Jyoti, Chandramukhi, Kufri Jyoti, Pukhraj.
- **Alias Examples:** `"Agra APMC"` -> `Agra`, `"Jyoti Potato"` -> `Jyoti`, `"Chandramukhi (A)"` -> `Chandramukhi`.

### 4.4 Tea
- **Canonical Markets:** Siliguri, Kolkata, Jalpaiguri, Guwahati, Coonoor.
- **Canonical Varieties:** CTC, Orthodox, Green Tea, Dust, Leaf.
- **Alias Examples:** `"Siliguri Auction"` -> `Siliguri`, `"Kolkata Auction Centre"` -> `Kolkata`, `"Crush Tear Curl"` -> `CTC`.

---

## 5. Strict Anti-Fabrication & Data Integrity Invariants

1. **Zero Fabrication of Missing Reports:**
   Agricultural mandis operate on irregular reporting schedules due to local market holidays, strikes, or delayed data entry. If an APMC market does not report on a calendar day, the ingestion engine **never** creates an interpolated observation.
2. **Zero Price Movement Prohibition:**
   Under no circumstances is a missing daily report interpreted as "zero price movement" or substituted with the previous day's price. Missing dates remain gaps in the observation table.
3. **Explicit Unmapped Tracking:**
   When an unknown mandi or variety name appears in incoming feeds, it is preserved verbatim in the observations table and recorded in the ingestion audit ledger (`unmapped_markets` / `unmapped_varieties`). No records are dropped silently.
4. **Non-Positive Price Rejection:**
   Observations with `modal_price <= 0` or NaN values are rejected immediately with explicit error logging (`invalid_dropped_count`).
5. **Idempotency Guarantee:**
   Ingesting the same daily batch multiple times produces zero duplicate rows and zero ledger corruption. The composite primary key `(commodity, market, variety, reported_date)` guarantees deduplication.

---

## 6. Freshness Auditing & Statistical Drift Detection

The ingestion layer runs an automated data quality audit on every ingested batch:

### 6.1 Freshness Tracking
- **Reference Date:** Current calendar date (UTC/IST).
- **Latest Reported Date:** Maximum `reported_date` in the batch.
- **Calendar Lag:** `Current Date - Latest Reported Date` (in calendar days).
- **Staleness Threshold:** If calendar lag exceeds **14 days**, the feed is flagged as `is_stale: True` and an alert is recorded in operational telemetry.

### 6.2 Statistical Drift Detection
Batch price distributions (mean, std, median, IQR) are compared against verified historical training baselines:
- **Rice Baseline:** Mean 2,120.40 Rs./Qtl, Std 185.30 Rs./Qtl, IQR 210.00 Rs./Qtl.
- **Wheat Baseline:** Mean 2,340.50 Rs./Qtl, Std 142.10 Rs./Qtl, IQR 165.00 Rs./Qtl.
- **Potato Baseline:** Mean 1,080.20 Rs./Qtl, Std 295.40 Rs./Qtl, IQR 340.00 Rs./Qtl.
- **Tea Baseline:** Mean 215.80 Rs./Qtl, Std 48.20 Rs./Qtl, IQR 55.00 Rs./Qtl.

If the batch mean deviates by more than **3.0 standard deviations** or the median shifts by more than **3.0 times the IQR**, `distribution_drift_detected: True` is flagged, alerting data engineering to structural regime shifts or reporting anomalies.

---

## 7. Storage Schema (`multi_commodity_mandi.db`)

```sql
CREATE TABLE IF NOT EXISTS mandi_observations (
    commodity TEXT NOT NULL,
    market TEXT NOT NULL,
    variety TEXT NOT NULL,
    reported_date TEXT NOT NULL,
    modal_price REAL NOT NULL,
    arrivals REAL NOT NULL DEFAULT 0.0,
    source_tag TEXT NOT NULL DEFAULT 'data.gov.in',
    ingested_at TEXT NOT NULL,
    PRIMARY KEY (commodity, market, variety, reported_date)
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    commodity TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    status TEXT NOT NULL,
    rows_received INTEGER NOT NULL,
    rows_inserted INTEGER NOT NULL,
    rows_duplicate INTEGER NOT NULL,
    unmapped_mandis_count INTEGER NOT NULL,
    details_json TEXT
);
```

---

## 8. Verification & Test Evidence

All 18 targeted test cases in `member2-price-volatility/tests/test_multi_commodity_ingestion.py` passed with 100% success:
- Connector URL and resource ID resolution: **PASS**
- Credential masking and secret protection: **PASS**
- Exponential backoff on HTTP 429: **PASS**
- Canonical alias mapping across Rice, Wheat, Potato, Tea: **PASS**
- Mixed date format parsing: **PASS**
- Strict anti-fabrication and zero-movement interpolation avoidance: **PASS**
- Idempotent duplicate ingestion: **PASS**
- Calendar freshness and statistical drift detection: **PASS**
- Operational summary and telemetry reporting: **PASS**

---

## 9. Ingestion Sign-off

The live data ingestion architecture meets all requirements for staging and real-world agricultural market connectivity.

**Verdict:** `INGESTION READY`
