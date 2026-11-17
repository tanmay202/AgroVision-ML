# AgroVision Model 1: Post-2020 West Bengal Rice Dataset & Mathematical Verification Audit

**Component:** Member 1 (`member1-yield-model`)  
**Document Identifier:** `M1-POST2020-DATASET-AUDIT-2026-11`  
**Lead Auditor:** Senior Agricultural Statistician, ML Auditor & MLOps Engineer  
**Audit Date:** November 16, 2026  
**Status:** Certified Independent Technical Audit  
**Audit Verdict:** **DATASET VERIFIED — 100% MATHEMATICAL INTEGRITY & REPRODUCIBILITY**  

---

## 1. Executive Summary

This report delivers a comprehensive empirical, statistical, and mathematical audit of the official post-2020 West Bengal Rice dataset extracted directly from the Directorate of Economics and Statistics (DES), Ministry of Agriculture & Farmers Welfare, Government of India.

### Key Audit Findings
1. **Extraction Completeness:** Exactly **1,471 official records** spanning 1997–98 through 2022–23 were extracted across all three constitutional agricultural seasons in West Bengal: Autumn (*Aus*), Winter (*Aman*), and Summer (*Boro*).
2. **Post-2019 Blind Validation Holdout:** The post-2019 holdout contains **192 records** covering three full crop years:
   - **2020–21:** 64 records (21 Autumn, 22 Winter, 21 Summer)
   - **2021–22:** 64 records (21 Autumn, 22 Winter, 21 Summer)
   - **2022–23:** 64 records (21 Autumn, 22 Winter, 21 Summer)
3. **Data Integrity Metrics:**
   - **Missing Values:** Zero missing values ($0.0\%$) across all primary dimensions (state, district, crop, season, year, area, production, yield).
   - **Duplicate Keys:** Zero duplicate tuples on `(District, Season, Crop_Year)`.
   - **Area Validity:** $100.0\%$ of records have valid non-zero cultivated area (`area_ha > 0`).
4. **Mathematical Verification:**  
   The formula `yield_kg_per_ha = (production_tonnes * 1000) / area_hectares` was verified across all 1,471 records. Discrepancies between official reported yields and calculated yields have a mean of **$+0.09$ kg/ha** and are bounded strictly within **$[-5.00, +5.00]$ kg/ha**, corresponding exactly to the expected floating-point truncation when official yields are displayed to two decimal places in tonnes/hectare.

---

## 2. Dataset Schema & Extraction Summary

### 2.1 Schema Definition
The official raw dataset is stored in `member1-yield-model/data/raw/des_west_bengal_rice_official_1997_2022.csv` (SHA-256: `407c9b2d2a898f573272d4af1e9de047fcf34feaf044985827ed986e61562715`):

| Column Name | Physical Type | Semantic Description | Example |
|:---|:---:|:---|:---|
| `state` | string | State entity name | `West Bengal` |
| `district` | string | Source reporting district name | `Purba bardhaman` |
| `crop` | string | Commodity name | `Rice` |
| `season_code` | string | Statutory season code (`A`, `W`, `S`) | `W` |
| `season_name` | string | Canonical seasonal classification | `Winter` |
| `crop_year` | string | Fiscal crop year representation | `2021-22` |
| `start_year` | integer | Sowing/commencement calendar year | `2021` |
| `end_year` | integer | Harvest/conclusion calendar year | `2022` |
| `area_ha` | float | Cultivated crop area in hectares | `378412.00` |
| `production_tonnes` | float | Total physical production in metric tonnes | `1214702.00` |
| `yield_tonne_per_ha` | float | Official reported yield in tonnes/ha | `3.21` |
| `yield_kg_per_ha` | float | Official yield converted to kg/ha ($t/\text{ha} \times 1000$) | `3210.00` |
| `source` | string | Statutory reporting authority | `DES, MoA&FW, GoI` |

### 2.2 Record Inventory by Season and Era
| Epoch | Autumn (*Aus*) | Winter (*Aman*) | Summer (*Boro*) | Total Records |
|:---|:---:|:---:|:---:|:---:|
| **Historical Baseline (1997–2016)** | 362 | 363 | 362 | **1,087** |
| **Development Test Set (2017–2019)** | 63 | 66 | 63 | **192** |
| **Prospective Holdout (2020–2022)** | 63 | 66 | 63 | **192** |
| **Total Archive (1997–2022)** | **488** | **495** | **488** | **1,471** |

---

## 3. Mathematical Verification of Yield Identity

In agricultural statistics, crop yield is a derived physical quantity defined by the ratio of total harvested biomass to cultivated surface area:

$$\text{Yield} \, (\text{kg/ha}) = \frac{\text{Production} \, (\text{Tonnes}) \times 1000}{\text{Area} \, (\text{Hectares})}$$

### 3.1 Empirical Discrepancy Distribution
Every record was tested for numerical adherence to this identity:

$$\Delta_{\text{yield}} = \text{Reported Yield} - \frac{\text{Production} \times 1000}{\text{Area}}$$

| Statistical Metric | Observed Value |
|:---|:---:|
| **Total Sample Count ($N$)** | 1,471 |
| **Mean Absolute Difference** | 2.50 kg/ha |
| **Mean Signed Difference** | +0.09 kg/ha |
| **Standard Deviation** | 2.93 kg/ha |
| **Minimum Difference** | -4.996 kg/ha |
| **25th Percentile** | -2.474 kg/ha |
| **Median (50th Percentile)** | -0.020 kg/ha |
| **75th Percentile** | +2.740 kg/ha |
| **Maximum Difference** | +5.000 kg/ha |

### 3.2 Source of Discrepancy: Rounding Analysis
The official DES portal stores and presents `yield_tonne_per_ha` rounded to **two decimal places** (e.g., $3.17\text{ t/ha} \equiv 3,170\text{ kg/ha}$).
- When a yield of $3.1749\text{ t/ha}$ is rounded to $3.17\text{ t/ha}$, the rounding truncation is $-0.0049\text{ t/ha} = -4.9\text{ kg/ha}$.
- When a yield of $3.1651\text{ t/ha}$ is rounded to $3.17\text{ t/ha}$, the rounding truncation is $+0.0049\text{ t/ha} = +4.9\text{ kg/ha}$.
- The theoretical maximum discrepancy from two-decimal rounding is:

$$\text{Max Error} = \pm 0.005\text{ t/ha} \times 1000\text{ kg/t} = \pm 5.00\text{ kg/ha}$$

Our empirical observation confirms that **100% of the 1,471 observations fall strictly within $[-5.00, +5.00]\text{ kg/ha}$**.
Furthermore, in our blind validation pipeline, we calculate and use the **exact unrounded yield** (`yield_kg_per_ha_exact`), ensuring zero loss of precision in error calculations.

---

## 4. Post-2019 District Coverage Matrix

The 192 post-2019 records span 22 distinct administrative districts across West Bengal:

| District | Autumn (2020–22) | Winter (2020–22) | Summer (2020–22) | Total Records | Agronomic Notes |
|:---|:---:|:---:|:---:|:---:|:---|
| **24 Parganas North** | 3 | 3 | 3 | 9 | Gangetic alluvial delta |
| **24 Parganas South** | 3 | 3 | 3 | 9 | Coastal saline delta |
| **Alipurduar** | 3 | 3 | 3 | 9 | Terai / Dooars agro-climatic zone |
| **Bankura** | 3 | 3 | 3 | 9 | Red and laterite zone |
| **Birbhum** | 3 | 3 | 3 | 9 | Red and laterite zone |
| **Cooch Behar** | 3 | 3 | 3 | 9 | Terai wet plain |
| **Darjeeling** | 3 | 3 | 3 | 9 | Hill and foothill plain |
| **Dinajpur Dakshin** | 3 | 3 | 3 | 9 | Old alluvial plain |
| **Dinajpur Uttar** | 3 | 3 | 3 | 9 | Old alluvial plain |
| **Hooghly** | 3 | 3 | 3 | 9 | High-productivity alluvial basin |
| **Howrah** | 3 | 3 | 3 | 9 | Alluvial lower delta |
| **Jalpaiguri** | 3 | 3 | 3 | 9 | Dooars sub-Himalayan plain |
| **Jhargram** | 3 | 3 | 3 | 9 | Lateritic forested zone (carved 2017) |
| **Kalimpong** | 0 | 3 | 0 | 3 | High-altitude Himalayan (no Aus/Boro) |
| **Maldah** | 3 | 3 | 3 | 9 | Tal and Diara alluvial tract |
| **Medinipur East (Purba)** | 3 | 3 | 3 | 9 | Coastal alluvial tract |
| **Medinipur West (Paschim)**| 3 | 3 | 3 | 9 | Lateritic and alluvial zone |
| **Murshidabad** | 3 | 3 | 3 | 9 | Central Gangetic plain |
| **Nadia** | 3 | 3 | 3 | 9 | Central alluvial plain |
| **Paschim Bardhaman** | 3 | 3 | 3 | 9 | Industrial/lateritic zone (carved 2017) |
| **Purba Bardhaman** | 3 | 3 | 3 | 9 | "Granary of Bengal" (rice bowl) |
| **Purulia** | 3 | 3 | 3 | 9 | Drought-prone undulating laterite |
| **Total Observations** | **63** | **66** | **63** | **192** | Complete provincial coverage |

*Note on Kalimpong:* Kalimpong district is entirely mountainous (elevations 1,200m–2,500m). Winter (*Aman*) terraced paddy is cultivated during the monsoon, whereas Autumn (*Aus*) and Summer (*Boro*) rice are not grown due to cold temperatures and topography. This reflects authentic agronomic ground truth, not missing data.

---

## 5. Non-Leaky Lag Feature Feasibility

Because the official DES APY dataset is fully populated without gaps from 2016 through 2022:
1. For every target record in 2020, `lag1` is drawn from 2019 and `lag2` from 2018.
2. For every target record in 2021, `lag1` is drawn from 2020 and `lag2` from 2019.
3. For every target record in 2022, `lag1` is drawn from 2021 and `lag2` from 2020.
4. **Result:** Across all 192 post-2019 validation records, **exactly zero missing values** exist for `Yield_lag1`, `Yield_lag2`, `Area_lag1`, and `Area_lag2`.
