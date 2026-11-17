# AgroVision Model 1 Expansion — Wheat Data Feasibility Audit Report

**Document ID:** AV-M1-EXP-WHEAT-DATA-001  
**Version:** 1.0.0  
**Effective Date:** 2026-11-17  
**Crop:** Wheat (*Triticum aestivum*)  
**Geographic Domain:** West Bengal (23 Administrative Districts)  
**Classification:** Agricultural Data Engineering & Feasibility Audit  
**Author:** Senior Agricultural ML Researcher & MLOps Engineer  
**Feasibility Determination:** **DATA SUFFICIENT FOR HISTORICAL MODELING (1997–2019); POST-2020 VALIDATION DATA GAP**

---

## 1. Executive Summary

As part of the AgroVision-ML Member 1 expansion from single-crop rice yield forecasting to multi-crop yield intelligence, an exhaustive data feasibility audit was conducted on official district-level wheat statistics in West Bengal.

The primary target is biological crop productivity intensity (Yield in $\text{kg/ha}$ and $\text{tonnes/ha}$) and volume (Production in $\text{tonnes}$ conditional on cultivated area in $\text{hectares}$). Mandi arrivals and APMC market prices are strictly segregated and not used as biological yield targets.

The audit verified 23 continuous agricultural years (1997–98 to 2019–20) spanning all 23 administrative districts of West Bengal. A critical unit scaling anomaly was discovered in the raw Ministry of Agriculture / Directorate of Economics and Statistics (DES) APY dataset, where older records reported yield in $\text{tonnes/ha}$ while newer concatenated blocks reported production scaled by $100\times$. A mathematically proven unit-harmonization protocol was established, reconciling 101 duplicate observation pairs to machine-precision floating-point identity ($< 1.4 \times 10^{-12}\text{ kg/ha}$).

---

## 2. Dataset Provenance & Ingestion Footprint

### 2.1 Raw Data Sources
- **Source File:** `member1-yield-model/data/raw/Wheat.csv` (41,119 bytes)
- **Origin Master:** Extracted from national crop database `member1-yield-model/data/raw/yield.csv` (575,879 records, 41.9 MB), compiled from official Area, Production, and Yield (APY) statistics published by the Directorate of Economics and Statistics (DES), Ministry of Agriculture and Farmers Welfare, Government of India.
- **Geographic Filter:** State = `'West Bengal'`, Crop = `'Wheat'`.
- **Target Variable:** Biological harvest yield ($\text{kg/ha}$), distinct from commercial market arrivals.

### 2.2 Record Inventory & Deduplication Summary

| Metric | Raw Dataset Count | Clean Deduplicated Count | Discrepancy & Handling |
| :--- | :---: | :---: | :--- |
| **Total Row Count** | 544 | 443 | 101 duplicate district-year pairs identified and deduplicated |
| **Crop Years** | 23 (1997–2019) | 23 (1997–2019) | Full 23-year historical temporal continuity |
| **Districts Covered** | 23 | 23 | All 23 LGD-standardized districts represented |
| **Season Labels** | 1 (`'Rabi'`: 544) | 1 (`'Rabi'`: 443) | 100% Rabi season across all records |
| **Missing Values** | 0 nulls across all cols | 0 nulls | 100% complete across Area, Production, and Yield |

---

## 3. Critical Unit Scaling Discovery & Reconciliation Protocol

### 3.1 The Two-Scale Concatenation Anomaly
An forensic examination of raw yields in `Wheat.csv` revealed a multimodal distribution:
- **Block A (119 records, rows 0–118):** Yields clustered between $0.80$ and $3.50$. Inspection showed Area was in hectares, Production was in raw metric tonnes ($P$), yielding $\text{Yield} = P / A$ in **metric tonnes per hectare** ($\text{t/ha}$).
- **Block B (425 records, rows 119–543):** Yields clustered between $80.0$ and $424.0$. Here, Production was entered as $100 \times P$ (an amateur unit conversion artifact common in merged Indian open datasets attempting to convert tonnes to quintals but multiplying by 100 instead of 10). Consequently, $\text{Yield} = (100 \times P) / A = 100 \times \text{Yield}_{\text{t/ha}} = \text{Yield}_{\text{kg/ha}} / 10$.

### 3.2 Mathematical Identity on 101 Duplicate Pairs
For the 101 district-year pairs present in both Block A and Block B from 1997 to 2014, the relationship was tested:

$$y_{\text{Block A}} \times 1000.0 \stackrel{?}{=} y_{\text{Block B}} \times 10.0$$

```
========================================================================================
             WHEAT 101-PAIR MATHEMATICAL IDENTITY VERIFICATION
========================================================================================
Sample Pair (24 Paraganas North, 2001):
  Block A (row 4):   Area = 16621.0 ha, Prod = 34176.0 t,   Yield = 2.056194 t/ha
  Block B (row 119): Area = 16621.0 ha, Prod = 3417600.0,   Yield = 205.619397
  Harmonized Yield A (y * 1000.0): 2056.194 kg/ha
  Harmonized Yield B (y * 10.0):   2056.194 kg/ha
  Absolute Difference:             0.000000000000000 kg/ha
----------------------------------------------------------------------------------------
Maximum Absolute Error Across All 101 Pairs: 1.364242e-12 kg/ha
========================================================================================
```

### 3.3 Harmonization Invariant
To guarantee data integrity across all 443 deduplicated observations:
1. When raw yield $< 10.0$:
   $$\text{Yield}_{\text{kg/ha}} = \text{Yield}_{\text{raw}} \times 1000.0, \quad \text{Production}_{\text{tonnes}} = \text{Production}_{\text{raw}}$$
2. When raw yield $\ge 10.0$:
   $$\text{Yield}_{\text{kg/ha}} = \text{Yield}_{\text{raw}} \times 10.0, \quad \text{Production}_{\text{tonnes}} = \text{Production}_{\text{raw}} / 100.0$$
3. Exact mathematical identity holds:
   $$\text{Yield}_{\text{kg/ha}} \equiv \frac{\text{Production}_{\text{tonnes}} \times 1000.0}{\text{Area}_{\text{hectares}}}$$

---

## 4. Agronomic Profile & Descriptive Statistics (West Bengal Wheat)

### 4.1 Agronomic Calendar
- **Crop Season:** Rabi (Winter)
- **Sowing Period:** November 15 – December 15
- **Critical Vegetative / Tillering:** December – January
- **Heading / Anthesis:** Late January – mid February
- **Grain Filling / Ripening:** February – early March
- **Harvesting Period:** Mid March – mid April

### 4.2 Clean Harmonized Yield Distribution ($N=443$)

| Statistic | Value ($\text{kg/ha}$) | Value ($\text{tonnes/ha}$) | Agronomic Context |
| :--- | :---: | :---: | :--- |
| **Sample Count ($N$)** | 443 | 443 | 23 districts over 23 crop years (1997–2019) |
| **Mean** | $2,306.54$ | $2.31$ | Standard Indo-Gangetic alluvial plain yield |
| **Standard Deviation** | $493.35$ | $0.49$ | Moderate inter-annual and regional variability |
| **Minimum** | $807.78$ | $0.81$ | Darjeeling (1998, high-altitude terrace cultivation) |
| **25th Percentile (p25)** | $1,995.08$ | $2.00$ | Lower quartile |
| **Median (p50)** | $2,301.08$ | $2.30$ | Highly symmetric median |
| **75th Percentile (p75)** | $2,632.98$ | $2.63$ | Upper quartile |
| **Maximum** | $4,247.58$ | $4.25$ | Murshidabad (2018, intensive irrigated alluvial tract) |

---

## 5. Geographic Alignment & District Bifurcation Handling

District names were cross-walked to the 23 standard Local Government Directory (LGD) districts:
- Undivided `BARDHAMAN` records prior to 2017 are mapped consistently to `Purba Bardhaman`.
- Bifurcated districts (`Alipurduar` from `Jalpaiguri` in 2014; `Kalimpong` from `Darjeeling` in 2017; `Jhargram` from `Paschim Medinipur` in 2017; `Paschim Bardhaman` from `Bardhaman` in 2017) are represented with their historic parent series for lag construction.
- Non-growing urban district: `Kolkata` has zero wheat area and is correctly omitted from agricultural yield targets.

---

## 6. Weather Feasibility & Temporal Alignment

- **NASA POWER Archives:** 23 JSON weather files in `member1-yield-model/data/raw/weather/` provide complete daily meteorological observations from `1997-01-01` to `2019-12-31` (8,400 consecutive days per district).
- **Temporal Alignment for Rabi Wheat:**
  - Crop year $T$ corresponds to sowing in November of year $T-1$ and harvest in March/April of year $T$.
  - Daily weather series are aligned to window $[T-1\text{ Nov 15},\ T\text{ Jan 31}]$ (78 days) or $[T-1\text{ Nov 15},\ T\text{ Feb 15}]$ (92 days).
  - Complete daily weather coverage exists for all 443 observations from 1998 through 2019.

---

## 7. Data Feasibility Conclusion & Gaps

1. **Feasibility Verdict:** **`DATA SUFFICIENT FOR HISTORICAL MODELING`**.
2. **Strength:** 23 continuous years of unit-harmonized, zero-leakage, zero-missing district observations coupled with complete daily NASA POWER archives.
3. **Identified Gap:** Official post-2020 DES APY wheat harvest statistics for West Bengal (2020–21 through 2023–24) are not currently available in the repository. Consequently, while historical backtesting is rigorous, operational deployment must be classified as **`EXPERIMENTAL`** until independent blind post-2020 holdout validation is executed.
