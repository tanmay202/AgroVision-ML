# AgroVision Model 1 Expansion — Potato Data Feasibility Audit Report

**Document ID:** AV-M1-EXP-POTATO-DATA-001  
**Version:** 1.0.0  
**Effective Date:** 2026-11-17  
**Crop:** Potato (*Solanum tuberosum*)  
**Geographic Domain:** West Bengal (23 Administrative Districts)  
**Classification:** Agricultural Data Engineering & Feasibility Audit  
**Author:** Senior Agricultural ML Researcher & MLOps Engineer  
**Feasibility Determination:** **DATA SUFFICIENT FOR HISTORICAL MODELING (1997–2019); POST-2020 VALIDATION DATA GAP**

---

## 1. Executive Summary

As part of the AgroVision-ML Member 1 expansion from single-crop rice yield forecasting to multi-crop yield intelligence, an exhaustive data feasibility audit was conducted on official district-level potato statistics in West Bengal.

The primary target is biological crop productivity intensity (Yield in $\text{kg/ha}$ and $\text{tonnes/ha}$) and volume (Production in $\text{tonnes}$ conditional on cultivated area in $\text{hectares}$). Mandi arrivals and APMC market prices are strictly segregated and not used as biological yield targets.

West Bengal is India's second-largest potato producer, contributing ~30% of national production. The audit verified 23 continuous agricultural years (1997–98 to 2019–20) spanning all 23 administrative districts of West Bengal. A critical unit scaling anomaly was discovered in the raw Ministry of Agriculture / Directorate of Economics and Statistics (DES) APY dataset, where older records reported yield in $\text{tonnes/ha}$ while newer concatenated blocks reported production scaled by $100\times$. A mathematically proven unit-harmonization protocol was established, reconciling 103 duplicate observation pairs to machine-precision floating-point identity ($< 1.1 \times 10^{-11}\text{ kg/ha}$).

---

## 2. Dataset Provenance & Ingestion Footprint

### 2.1 Raw Data Sources
- **Source File:** `member1-yield-model/data/raw/Potato.csv` (42,757 bytes)
- **Origin Master:** Extracted from national crop database `member1-yield-model/data/raw/yield.csv` (575,879 records, 41.9 MB), compiled from official Area, Production, and Yield (APY) statistics published by the Directorate of Economics and Statistics (DES), Ministry of Agriculture and Farmers Welfare, Government of India.
- **Geographic Filter:** State = `'West Bengal'`, Crop = `'Potato'`.
- **Target Variable:** Biological harvest yield ($\text{kg/ha}$), distinct from commercial market arrivals or cold-storage stocks.

### 2.2 Record Inventory & Deduplication Summary

| Metric | Raw Dataset Count | Clean Deduplicated Count | Discrepancy & Handling |
| :--- | :---: | :---: | :--- |
| **Total Row Count** | 568 | 465 | 103 duplicate district-year pairs identified and deduplicated |
| **Crop Years** | 23 (1997–2019) | 23 (1997–2019) | Full 23-year historical temporal continuity |
| **Districts Covered** | 23 | 23 | All 23 LGD-standardized districts represented |
| **Season Labels** | 4 distinct labels | 4 distinct labels | `Rabi` (220), `Whole Year` (190), `Winter` (38), `Summer` (17) |
| **Missing Values** | 0 nulls across all cols | 0 nulls | 100% complete across Area, Production, and Yield |

---

## 3. Critical Unit Scaling Discovery & Reconciliation Protocol

### 3.1 The Two-Scale Concatenation Anomaly
An empirical examination of raw yields in `Potato.csv` revealed a sharp bimodal distribution mirroring the anomaly uncovered in wheat:
- **Block A (121 records, rows 0–120):** Yields clustered between $2.27$ and $41.35$. Inspection showed Area was in hectares, Production was in raw metric tonnes ($P$), yielding $\text{Yield} = P / A$ in **metric tonnes per hectare** ($\text{t/ha}$).
- **Block B (447 records, rows 121–567):** Yields clustered between $226.9$ and $4,500.6$. Here, Production was entered as $100 \times P$ (an erroneous unit conversion artifact multiplying tonnes by 100 instead of 10). Consequently, $\text{Yield} = (100 \times P) / A = 100 \times \text{Yield}_{\text{t/ha}} = \text{Yield}_{\text{kg/ha}} / 10$.

### 3.2 Mathematical Identity on 103 Duplicate Pairs
For the 103 district-year-season pairs present in both Block A and Block B from 1997 to 2014, the relationship was tested:

$$y_{\text{Block A}} \times 1000.0 \stackrel{?}{=} y_{\text{Block B}} \times 10.0$$

```
========================================================================================
             POTATO 103-PAIR MATHEMATICAL IDENTITY VERIFICATION
========================================================================================
Sample Pair (Hooghly, 2002, Season: Whole Year):
  Block A (row 14):  Area = 76785.0 ha, Prod = 2642140.0 t,   Yield = 34.409585 t/ha
  Block B (row 145): Area = 76785.0 ha, Prod = 264214000.0,   Yield = 3440.958521
  Harmonized Yield A (y * 1000.0): 34409.585218 kg/ha
  Harmonized Yield B (y * 10.0):   34409.585218 kg/ha
  Absolute Difference:             0.000000000000000 kg/ha
----------------------------------------------------------------------------------------
Duplicate Pairs Count: 103
Maximum Absolute Error Across All 103 Pairs: 1.091394e-11 kg/ha
Mean Absolute Error Across All 103 Pairs:    2.039741e-12 kg/ha
========================================================================================
```

The maximum error across all 103 pairs is $< 1.1 \times 10^{-11}\text{ kg/ha}$, conclusively proving mathematical equivalence.

### 3.3 Harmonization Invariant
To guarantee data integrity across all 465 deduplicated observations:
1. When raw yield $< 100.0$:
   $$\text{Yield}_{\text{kg/ha}} = \text{Yield}_{\text{raw}} \times 1000.0, \quad \text{Production}_{\text{tonnes}} = \text{Production}_{\text{raw}}$$
2. When raw yield $\ge 100.0$:
   $$\text{Yield}_{\text{kg/ha}} = \text{Yield}_{\text{raw}} \times 10.0, \quad \text{Production}_{\text{tonnes}} = \text{Production}_{\text{raw}} / 100.0$$
3. Exact mathematical identity holds:
   $$\text{Yield}_{\text{kg/ha}} \equiv \frac{\text{Production}_{\text{tonnes}} \times 1000.0}{\text{Area}_{\text{hectares}}}$$

---

## 4. Seasonal Taxonomy & Agronomic Profiling

### 4.1 Plain vs. Hill Agro-Ecological Segregation
Potato cultivation in West Bengal falls into two distinct agro-ecological regimes:
1. **Plains Winter/Rabi Potato ($N=448$ records):**
   - Labeled variably in APY as `Rabi` (220), `Whole Year` (190), or `Winter` (38).
   - Grown across 21 southern and northern plains districts (Hooghly, Purba Bardhaman, Paschim Medinipur, Bankura, Jalpaiguri, etc.).
   - Sown: Early to mid November.
   - Harvested: February to mid March.
   - Duration: 80–100 days.
   - Represents $>99.5\%$ of West Bengal total potato tonnage.
2. **Himalayan Summer Potato ($N=17$ records):**
   - Strictly confined to `Darjeeling` (13) and `Kalimpong` (4) hill tracts.
   - Sown: March – April.
   - Harvested: August – September.
   - Represents $<0.5\%$ of state production.
   - Segregated from the main winter forecasting engine.

### 4.2 Clean Harmonized Yield Distribution

#### All West Bengal Records ($N=465$)

| Statistic | Value ($\text{kg/ha}$) | Value ($\text{tonnes/ha}$) | Agronomic Context |
| :--- | :---: | :---: | :--- |
| **Sample Count ($N$)** | 465 | 465 | All 23 districts, 1997–2019 |
| **Mean** | $23,464.10$ | $23.46$ | High-yielding subtropical tuber cultivation |
| **Standard Deviation** | $7,982.20$ | $7.98$ | High inter-district technology gradient |
| **Minimum** | $2,269.06$ | $2.27$ | South 24 Parganas (coastal saline soil stress, 1997) |
| **25th Percentile (p25)** | $17,964.72$ | $17.96$ | Lower commercial quartile |
| **Median (p50)** | $22,899.53$ | $22.90$ | Median state productivity |
| **75th Percentile (p75)** | $29,174.89$ | $29.17$ | Intensive commercial belt (Hooghly/Bardhaman) |
| **Maximum** | $45,005.57$ | $45.01$ | Hooghly (2017, peak intensive certified seed tract) |

#### Plains Main-Crop Subset ($N=448$)

| Statistic | Value ($\text{kg/ha}$) | Value ($\text{tonnes/ha}$) | Agronomic Context |
| :--- | :---: | :---: | :--- |
| **Sample Count ($N$)** | 448 | 448 | Rabi, Whole Year, and Winter plains records |
| **Mean** | $23,967.07$ | $23.97$ | Commercial plains mean yield |
| **Standard Deviation** | $7,666.44$ | $7.67$ | Commercial variation |
| **Median (p50)** | $23,154.88$ | $23.15$ | Symmetric median |
| **Maximum** | $45,005.57$ | $45.01$ | Hooghly intensive belt |

---

## 5. Weather Feasibility & Agronomic Risk Indices

### 5.1 Weather Windows
- Potato vegetative and tuberization stages in the Indo-Gangetic plains are highly compressed:
  - Planting: November 1–20.
  - Tuber Initiation (Stolon tipping): 20–35 days after planting (late November – early December).
  - Tuber Bulking: 35–75 days after planting (December – January).
  - Senescence & Dehaulming: Late January – early February.
  - Harvesting: February 10 – March 15.
- Selected Operational Cutoff: **January 31** (captures 92 days from November 1; covers complete tuber initiation and the bulk of tuber expansion).

### 5.2 Agronomic Weather Risk Features
Unlike rice or wheat, potato yield in West Bengal is driven by two specific microclimatic hazards:
1. **Cool Night Requirement for Tuber Induction:**
   - Night temperatures ($T_{\text{min}}$) below $16^\circ\text{C}$ and $14^\circ\text{C}$ are essential for stolon partitioning to tubers.
   - Feature: `cool_night_days_16` and `cool_night_days_14` (count of days with $T_{\text{min}} \le 16^\circ\text{C}$ and $\le 14^\circ\text{C}$).
2. **Late Blight Disease Epidemic Risk (*Phytophthora infestans*):**
   - High humidity combined with moderate winter temperatures creates devastating Late Blight outbreaks in West Bengal (e.g., the 2013-14 epiphytotic).
   - Feature: `late_blight_risk_days` ($\text{RH} \ge 80\%$ and $10^\circ\text{C} \le T_{\text{mean}} \le 22^\circ\text{C}$).
3. **Growing Degree Days:**
   - Calculated with base temperature $5^\circ\text{C}$ (`gdd_base5`).

---

## 6. Data Feasibility Conclusion & Release Governance

1. **Feasibility Verdict:** **`DATA SUFFICIENT FOR HISTORICAL MODELING`**.
2. **Strength:** Complete, unit-harmonized, zero-missing, 23-year historical dataset covering all 23 districts with rigorous meteorological alignment.
3. **Identified Gap:** Official post-2020 DES APY potato harvest statistics for West Bengal (2020–21 through 2023–24) are pending ingestion. Consequently, while backtesting is validated, operational release governance must remain strictly **`EXPERIMENTAL`** until independent blind holdout validation is executed.
