# AgroVision Model 1: Summer (Boro) Routing Shadow Comparison Report

**Component:** Member 1 (`member1-yield-model`)  
**Production Artifact:** `models/m1_improved_delta_ridge.pkl`  
**Certified Checksum:** `3099c2d6d113105f185111f6b1f02004229cc0e32318e9f35532841f7d4dcb56`  
**Document Identifier:** `BORO-ROUTING-SHADOW-2026-11`  
**Lead Auditor:** Senior Agricultural ML Researcher & MLOps Architect  
**Audit Date:** November 17, 2026  
**Status:** Certified Shadow Evaluation  
**Governance Decision:** **RETAIN PRODUCTION POLICY 4 BY DEFAULT — CANARY ROUTING PATHWAY DOCUMENTED**  

---

## 1. Executive Summary & Context

In the Member 1 production inference pipeline, Summer (*Boro*) rice is routed by default to the unweighted historical persistence baseline (`Yield_lag1`), setting $\Delta = 0$. This behavior is governed by `PROVISIONAL_SUMMER_ROUTING_POLICY` (Policy 4).

### Historical Origin & Audit Mandate
- **Developmental Origin:** The provisional routing policy was introduced based on an observed $13.41\text{ kg/ha}$ MAE advantage for persistence over Delta Ridge on the 2017–2019 test set ($209.93$ vs. $223.34\text{ kg/ha}$).
- **The Phenological Blindness Dilemma:** Boro rice is irrigated through winter and suffers its major yield risks (terminal heat shock $T_{\text{max}} > 35^\circ\text{C}$ and *Kalbaishakhi* hailstorms) during flowering and grain filling in **April and May**. Because the operational cutoff is set to **February 28**, the model is completely blind to post-cutoff weather anomalies.
- **Milestone 4 Shadow Evaluation Mandate:** Evaluate candidate Pure Delta Ridge against existing Policy 4 and the Persistence baseline on identical post-2019 holdout observations ($N=63$), verify if the candidate consistently outperforms persistence, and document the statutory requirements for candidate promotion without modifying default production routing.

---

## 2. Three-Way Shadow Evaluation Matrix (Identical Holdout N=63)

All 63 official Summer (*Boro*) observations from the post-2019 holdout (crop years 2020–21, 2021–22, and 2022–23 across 21 districts) were evaluated simultaneously across all three configurations:

| Configuration Identifier | System Role | Active Logic | MAE (kg/ha) | RMSE (kg/ha) | $R^2$ Score | Mean Bias (kg/ha) | MAPE (%) | Net Advantage vs. Persistence |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Persistence Baseline** | Benchmark | $\hat{Y} = \text{Yield\_lag1}$ | 220.76 | 289.15 | 0.5523 | +21.92 | 6.94% | Baseline ($0.00\text{ kg/ha}$) |
| **2. Production Policy 4** | **Active Production** | Boro routed to Persistence ($\Delta = 0$) | 220.76 | 289.15 | 0.5523 | +21.92 | 6.94% | Parity with Persistence |
| **3. Candidate Pure Delta Ridge** | **Shadow Candidate** | Unconstrained ML ($\hat{Y} = Y_{\text{lag1}} + \hat{\Delta}$) | **200.08** | **268.35** | **0.6144** | -21.89 | **6.22%** | **+20.68 kg/ha (Champion)** |

```mermaid
flowchart LR
    subgraph Summer Boro Holdout Evaluation N=63
        P1["Persistence Baseline\nMAE: 220.76 kg/ha\nRMSE: 289.15 kg/ha"]
        P4["Production Policy 4\nMAE: 220.76 kg/ha\nRMSE: 289.15 kg/ha"]
        CR["Candidate Pure Delta Ridge\nMAE: 200.08 kg/ha\nRMSE: 268.35 kg/ha"]
    end

    P1 ---|Identical Policy Output| P4
    P4 ==>|+20.68 kg/ha MAE Gain\nR² increases 0.5523 -> 0.6144| CR
```

---

## 3. Annual Performance Breakdown Across Holdout Crop Years

| Crop Year | Sample Count ($N$) | Persistence MAE (kg/ha) | Production Policy 4 MAE (kg/ha) | Candidate Delta Ridge MAE (kg/ha) | Candidate Advantage vs. Policy 4 | Delta Ridge $R^2$ | Delta Ridge MAPE |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **2020–21** | 21 | 307.74 | 307.74 | **292.29** | **+15.45 kg/ha** | 0.3984 | 9.53% |
| **2021–22** | 21 | 172.45 | 172.45 | **151.78** | **+20.67 kg/ha** | 0.7771 | 4.32% |
| **2022–23** | 21 | 182.08 | 182.08 | **156.17** | **+25.91 kg/ha** | 0.7287 | 4.80% |
| **3-Year Aggregate** | **63** | **220.76** | **220.76** | **200.08** | **+20.68 kg/ha** | **0.6144** | **6.22%** |

### Key Empirical Findings
1. **Unanimous Annual Superiority:** Candidate Pure Delta Ridge outperformed Policy 4 in **100% of post-2019 crop years** ($3 / 3$ consecutive annual wins).
2. **Expanding Margin of Advantage:** The candidate's advantage expanded from $+15.45\text{ kg/ha}$ in 2020–21 to $+20.67\text{ kg/ha}$ in 2021–22, reaching $+25.91\text{ kg/ha}$ in 2022–23.
3. **Variance Reduction:** Root Mean Squared Error decreased from $289.15$ to $268.35\text{ kg/ha}$ ($-20.80\text{ kg/ha}$ reduction in large errors), and $R^2$ improved by $+0.0621$ to **0.6144**.
4. **Row-Level Win Rate:** Pure Delta Ridge produced lower absolute prediction error than Persistence in **36 out of 63 observations (57.1%)**.

---

## 4. Why Did Delta Ridge Outperform Persistence Post-2020?

Despite the February 28 cutoff blindness to April heat stress:
1. **District-Level Trend Regularization:**  
   Ridge regression regularizes deviations around `Yield_lag1` using district dummy embeddings and regional historical baselines. In years where regional productivity shifted (e.g. 2021–22 high-yield regime), the model accurately scaled deltas upward rather than anchoring rigidly to prior years.
2. **Diurnal Temperature Range Signal:**  
   Winter temperature range (`InSeason_T2M_Range`) and relative humidity (`InSeason_RH2M_Mean`) during the January–February tillering stage provide an early proxy for crop vegetative vigor, allowing the model to make beneficial adjustments before the cutoff.

---

## 5. Promotion Governance & Statutory Evidence Protocol

Although Candidate Pure Delta Ridge demonstrates a $+20.68\text{ kg/ha}$ empirical advantage on the 2020–2022 holdout, **production routing will NOT be altered silently or prematurely**.

### 5.1 Non-Negotiable Criteria Required Before Promotion
Before promoting Candidate Pure Delta Ridge to default production status, the following three criteria must be fulfilled:
1. **Confirmation on Final 2023–24 Official DES APY Statistics:**  
   The candidate must be evaluated on the forthcoming 2023–24 West Bengal Rice dataset once published by DES, verifying that the $+20\text{ kg/ha}$ advantage replicates on a 4th consecutive year.
2. **Prospective Real-Time Shadow Cycle:**  
   The candidate must operate in shadow mode across a live, ongoing Boro season (e.g. 2024–25 or 2025–26) with predictions locked in an immutable audit database on or before February 28, confirming operational resilience before harvest.
3. **Stakeholder & Agricultural Governance Approval:**  
   Formal sign-off by agricultural modeling leadership approving the transition from conservative persistence routing to full weather-informed ML inference.

### 5.2 Current Production Architecture Recommendation
- **Default Production Mode:** Retain `PROVISIONAL_SUMMER_ROUTING_POLICY` (Policy 4) as default (`enable_seasonal_routing=True`).
- **Opt-In Candidate Path:** The shared prediction bridge and staging API support explicit opting-in via `enable_seasonal_routing=False` for research and canary evaluation cohorts.
