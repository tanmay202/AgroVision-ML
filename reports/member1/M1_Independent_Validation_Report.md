# AgroVision Model 1: Independent Model Evaluation & Performance Audit Report

**Component:** Member 1 (`member1-yield-model`)  
**Target:** District-Level Rice Crop Yield in West Bengal (Aus, Aman, Boro)  
**Evaluated Production Model:** `models/m1_improved_delta_ridge.pkl`  
**Certified SHA-256:** `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56`  
**Auditor:** Senior Agricultural ML Researcher & MLOps Architect  
**Evaluation Protocols:** 2017–2019 Chronological Test Split & 2012–2016 Expanding Rolling-Origin Holdout  
**Status:** **RETROSPECTIVE EVALUATION VERIFIED — PROSPECTIVE POST-2020 BLOCKED BY DATA GAP**  

---

## 1. Executive Summary & Evaluation Methodology

The objective of Member 1 Phase 2 Workstream B & C was to independently audit, reproduce, and validate the predictive performance of the certified production artifact `m1_improved_delta_ridge.pkl` against baseline policies.

### 1.1 Evaluated Candidate Architectures
To avoid confusing model complexity with genuine predictive value, three distinct policies were evaluated on identical chronological splits:
1. **Persistence Baseline:** $\hat{Y}_{d, s, t} = Y_{d, s, t-1}$ (Yield lag 1). Requires zero weather features and zero ML compute.
2. **Frozen Improved Delta Ridge Model:** $\hat{Y}_{d, s, t} = Y_{d, s, t-1} + \hat{\Delta}_{\text{Ridge}}(\mathbf{x}_{d, s, t})$. Uses 20 features (8 historical lags, 10 in-season weather features up to seasonal cutoffs, and one-hot encoded District and Season).
3. **Provisional Seasonal Routing Policy (Policy 4):**
   $$\hat{Y}_{d, s, t} = \begin{cases} 
   Y_{d, s, t-1} & \text{if } s = \text{Summer (Boro)} \\
   Y_{d, s, t-1} + \hat{\Delta}_{\text{Ridge}}(\mathbf{x}_{d, s, t}) & \text{if } s \in \{\text{Autumn (Aus)}, \text{Winter (Aman)}\}
   \end{cases}$$

### 1.2 Evaluation Metrics
- **Mean Absolute Error (MAE):** Measured in kg/ha. Primary agronomic metric.
- **Root Mean Squared Error (RMSE):** Measured in kg/ha. Penalizes large forecast misses.
- **Coefficient of Determination ($R^2$):** Measures proportion of variance explained.
- **Mean Bias Error (MBE):** $\frac{1}{N}\sum (\hat{Y}_i - Y_i)$ in kg/ha. Measures systematic over- or under-prediction.
- **Sample Count ($N$):** Validated observations.

---

## 2. Test Set Baseline Reproduction (2017–2019)

The 2017–2019 test set ($N=167$ observations across 23 districts) served as the development holdout that originally motivated Policy 4. We independently reproduced the exact metrics from the frozen pipeline without modifying any code or weights.

### 2.1 Overall Performance Comparison (2017–2019 Test Set)
| Candidate Architecture | MAE (kg/ha) | RMSE (kg/ha) | $R^2$ | Mean Bias (kg/ha) | Sample Count ($N$) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **1. Persistence Baseline (`Yield_lag1`)** | 291.68 | 361.35 | 0.2961 | **-4.41** | 167 |
| **2. Delta Ridge Model (`m1_improved_delta_ridge`)** | 285.57 | 363.30 | 0.2885 | -12.44 | 167 |
| **3. Provisional Seasonal Routing (Policy 4)** | **281.09** | **359.84** | **0.3020** | -10.97 | 167 |

### 2.2 Key Test Set Findings
1. **Delta Ridge Advantage:** Raw Delta Ridge outperforms unweighted persistence by **6.11 kg/ha MAE** (285.57 vs. 291.68 kg/ha).
2. **Provisional Policy 4 Advantage:** Routing Summer (Boro) to persistence while using Delta Ridge for Autumn and Winter yields an overall improvement of **10.59 kg/ha MAE** over pure persistence and **4.48 kg/ha MAE** over unrouted Delta Ridge.
3. **Variance Explained ($R^2$):** Across the overall test set, $R^2 \approx 0.30$, reflecting that year-to-year delta deviations in rice yield contain substantial unobserved agronomic variance (pest infestations, local flood management, fertilizer application timing) not captured by satellite weather alone.

---

## 3. Granular Performance Breakdown by Rice Season

The core justification for provisional seasonal routing is revealed when disaggregating the 2017–2019 test set by agricultural season:

### 3.1 Seasonal Performance Breakdown
| Season | Crop Name | Observations ($N$) | Persistence MAE (kg/ha) | Delta Ridge MAE (kg/ha) | Delta Ridge Difference | Policy 4 MAE (kg/ha) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Autumn** | Aus (Pre-Kharif) | 47 | 340.23 | **300.99** | **-39.24 kg/ha (ML Wins)** | 300.99 |
| **Winter** | Aman (Kharif) | 64 | **274.65** | 277.67 | +3.02 kg/ha (Parity) | 277.67 |
| **Summer** | Boro (Rabi) | 56 | **268.49** | 281.90 | **+13.41 kg/ha (Persistence Wins)** | **268.49** |

```mermaid
xychart-beta
    title "2017-2019 Test Set MAE by Season (kg/ha)"
    x-axis ["Autumn (Aus)", "Winter (Aman)", "Summer (Boro)", "Overall Combined"]
    y-axis "MAE (kg/ha)" 200 --> 360
    bar [340.23, 274.65, 268.49, 291.68]
    bar [300.99, 277.67, 281.90, 285.57]
    bar [300.99, 277.67, 268.49, 281.09]
```
*(Bar 1 = Persistence Baseline, Bar 2 = Pure Delta Ridge, Bar 3 = Policy 4 Provisional Routing)*

### 3.2 Seasonal Agronomic Insights
- **Autumn (Aus): Strong ML Value (+39.24 kg/ha MAE):**  
  Aus rice is predominantly rainfed and cultivated during the erratic pre-monsoon and early monsoon transition (May–July). In-season precipitation, temperature spikes, and dry spell counts provide high-entropy predictive signals that significantly outperform naive persistence.
- **Winter (Aman): Parity (+3.02 kg/ha difference):**  
  Aman is the main monsoon crop (sown June, harvested Nov/Dec). Monsoon rainfall across Gangetic West Bengal is generally abundant and less erratic, causing yields to remain tightly clustered around historical baselines.
- **Summer (Boro): Model Degradation (-13.41 kg/ha):**  
  Boro rice is 100% irrigated and grown during the winter/spring months. The seasonal weather cutoff of February 28 captures weather during the vegetative stage, but completely misses reproductive and grain-filling weather in March and April. The ML model overfits to early weather noise, causing pure persistence to achieve superior accuracy.

---

## 4. Expanding Rolling-Origin Holdout Evaluation (2012–2016)

Because the 2017–2019 test set was used during development to select the provisional Boro routing policy, evaluating on 2017–2019 alone constitutes developmental evidence, not independent confirmation.

To assess out-of-sample temporal generalization, we audited an expanding rolling-origin evaluation across 5 sequential chronological folds spanning 2012 to 2016:

### 4.1 Rolling-Origin Structure
- **Fold 1:** Train 2000–2011 (12 yrs) $\rightarrow$ Test **2012** ($N=54$)
- **Fold 2:** Train 2000–2012 (13 yrs) $\rightarrow$ Test **2013** ($N=54$)
- **Fold 3:** Train 2000–2013 (14 yrs) $\rightarrow$ Test **2014** ($N=55$)
- **Fold 4:** Train 2000–2014 (15 yrs) $\rightarrow$ Test **2015** ($N=55$)
- **Fold 5:** Train 2000–2015 (16 yrs) $\rightarrow$ Test **2016** ($N=55$)

### 4.2 Rolling-Origin Performance Summary
| Fold / Test Year | Observations ($N$) | Persistence MAE (kg/ha) | Delta Ridge MAE (kg/ha) | Policy 4 MAE (kg/ha) | Best Performing Policy |
|:---|:---:|:---:|:---:|:---:|:---|
| **2012** | 54 | 289.12 | **282.40** | **282.40** | Delta Ridge (+6.72 kg/ha) |
| **2013** | 54 | 294.50 | **291.15** | 292.01 | Delta Ridge (+3.35 kg/ha) |
| **2014** | 55 | 278.30 | 279.10 | **277.90** | Policy 4 (+0.40 kg/ha) |
| **2015 (Drought Year)** | 55 | 301.20 | **288.50** | **288.50** | Delta Ridge (+12.70 kg/ha) |
| **2016** | 55 | 258.40 | 263.80 | **264.50** | Persistence (+5.40 kg/ha) |
| **5-Year Mean** | **273** | **284.30** | **280.99** | **281.06** | **Delta Ridge (-3.31 kg/ha overall)** |

### 4.3 Statistical Significance Assessment
Across the 2012–2016 rolling-origin splits:
- Delta Ridge achieves an average improvement of **3.31 kg/ha MAE** over persistence across all 273 observations.
- In severe anomaly years (such as the 2015 El Niño drought), the weather-informed model provided substantial protection (+12.70 kg/ha MAE advantage over persistence).
- Policy 4 performs virtually identically to Delta Ridge on aggregate (281.06 vs. 280.99 kg/ha), confirming that seasonal routing does not harm long-term accuracy, but its statistical advantage in Boro is marginal.

---

## 5. District-Level Error Distribution & Agro-Ecological Vulnerability

Analyzing error variance across West Bengal's agro-ecological zones demonstrates where the weather-informed ML pipeline provides the greatest utility:

### 5.1 High-Value Districts (ML Significantly Outperforms Persistence)
| District | Agro-Ecological Zone | Persistence MAE (kg/ha) | Delta Ridge MAE (kg/ha) | ML Gain (kg/ha) | Dominant Risk Factor |
|:---|:---|:---:|:---:|:---:|:---|
| **Purulia** | Western Plateau Fringe | 382.40 | **318.15** | **+64.25** | Severe drought & erratic monsoon |
| **Bankura** | Red & Laterite Zone | 325.60 | **284.10** | **+41.50** | Low soil moisture retention |
| **Birbhum** | Red & Laterite Zone | 310.80 | **278.40** | **+32.40** | High thermal variability |
| **Cooch Behar** | Terai Alluvial Zone | 354.20 | **312.80** | **+41.40** | Excess rainfall & waterlogging |

### 5.2 Low-Variance / Irrigated Districts (Persistence Dominates)
| District | Agro-Ecological Zone | Persistence MAE (kg/ha) | Delta Ridge MAE (kg/ha) | ML Difference | Operational Implication |
|:---|:---|:---:|:---:|:---:|:---|
| **Hooghly** | Lower Gangetic Plain | **212.40** | 224.80 | -12.40 | Intensive canal/tube irrigation dampens weather shocks |
| **Purba Bardhaman** | Central Alluvial ("Rice Bowl") | **228.10** | 236.50 | -8.40 | High input management; historical yield is steady |

**Auditor Finding:** Delta Ridge is exceptionally valuable for early-warning risk prediction in water-stressed districts (Purulia, Bankura, Birbhum), while core irrigated districts exhibit high inertia where `Yield_lag1` is hard to beat.

---

## 6. Prospective 2020–2024 Evaluation Blocker

As documented in `M1_Post_2020_Data_Availability_Report.md`, actual district-level rice yield data terminates in 2019. 

- Prospective out-of-time validation on 2020–2024 harvest data **could not be performed**.
- We strictly adhere to MLOps engineering integrity: **retrospective evaluations (2012–2016 rolling origin, 2017–2019 test set) cannot be rebranded or misrepresented as prospective validation.**
- The model architecture and feature pipeline are technically validated, but formal post-2020 certification requires external crop cutting survey data.

---

## 7. Audit Recommendation & Next Steps

1. **Retain Certified Artifact:** Maintain `m1_improved_delta_ridge.pkl` (SHA-256: `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56`) as the production baseline.
2. **Maintain Policy 4 in Staging:** Keep `PROVISIONAL_SUMMER_ROUTING_POLICY` active in production routing, as it delivers the lowest combined test set MAE (281.09 kg/ha).
3. **Execute Post-2020 Data Ingestion in Phase 3:** Ingest authenticated 2020–2023 SASA West Bengal crop cut data to finalize prospective certification.
