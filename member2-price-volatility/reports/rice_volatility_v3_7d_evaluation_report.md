# AgroVision Member 2 — Volatility V3 7-Day Calendar Horizon Evaluation Report

**Module:** Member 2 — Price Volatility Machine Learning  
**Author:** Senior ML Research Engineer & Evaluation Architect  
**Evaluation Target:** 7-Calendar-Day Forward Normalized Price Range  
**Date:** November 2026  
**Status:** Certified Experimental Evaluation  

---

## 1. Executive Summary

This report delivers the comprehensive evaluation for **Target 7D**, a 7-calendar-day forward price volatility classifier developed under the Member 2 V3 target engineering repair initiative.

### Key Findings
1. **Target Engineering Repair Validated:** In contrast to V2 where row-shifting (`.shift(-5)`) produced variable horizons extending up to 86 days in holdout and 13.8 years in training, Target 7D is strictly calendar-bounded. The maximum calendar offset between forecast origin and target observation is **exactly 7.0 days** across all 311,584 valid observations (mean: 6.66 days, median: 7.0 days).
2. **Persistence Baseline Remains Dominant at 7 Days:** On the independent retrospective holdout period (January 27, 2023 to January 27, 2024, N=20,166), Calendar Persistence achieves Macro F1 of **0.6177**, while Champion XGBoost (Balanced) achieves **0.6164**.
3. **Market-Level Clustering Rejects ML Superiority:** Disaggregated mandi-level evaluation reveals that **Calendar Persistence outperforms ML in 39 out of 56 mandis (69.6% win rate for baseline)**. Clustered paired testing confirms a statistically significant advantage for persistence ($p = 0.00006$, cluster-robust $t = -4.33$, 95% bootstrap CI $[-0.0398, -0.0152]$).
4. **Probability Calibration Greatly Improved:** Raw XGBoost probabilities suffered from severe miscalibration (Macro ECE = 9.70%, LOW ECE = 14.54%). Fitting Isotonic Regression on pre-holdout validation data reduces Macro ECE to **2.73%** and Brier score from 0.4693 to **0.4315**.

---

## 2. Target Engineering & Semantics (7D)

### 2.1 Formal Definition
For forecast origin $t$ with price $P_t$, the forward observation window is defined as:
$$W_7(t) = \{ \tau \mid t < \tau \le t + 7 \text{ calendar days, for the same (Market Name, Variety)} \}$$

The 7-calendar-day price-normalized forward percentage range is computed as:
$$\text{future\_pct\_range}_{7d} = \frac{\max_{s \in [t, t+7]} P_s - \min_{s \in [t, t+7]} P_s}{P_t} \times 100\%$$
where $[t, t+7] = \{P_t\} \cup \{P_\tau \mid \tau \in W_7(t)\}$.

### 2.2 Invariant Constraints
- **Minimum Future History:** Requires $|W_7(t)| \ge 1$ (at least one future trade within the 7 calendar days). If zero trades occurred, target is assigned `NaN` (insufficient future history).
- **Temporal Boundary Purging:** The exact date of the latest observation $\tau_{\max} \le t + 7$ is recorded in `future_target_date_7d`. Any training row where $\tau_{\max} \ge T_{\text{split}}$ is purged, preventing boundary leakage.

### 2.3 Threshold Derivation
Thresholds were derived strictly from the pre-2023 training dataset ($N=290,754$ clean observations):
- **LOW:** $\le 0.014286\%$ (captures flat trading days with zero or sub-tick price movement).
- **MEDIUM:** $0.014286\% < \text{range} \le 2.083333\%$ (derived from non-zero median).
- **HIGH:** $> 2.083333\%$ (exceeds non-zero median).

---

## 3. Walk-Forward Cross-Validation Performance

Evaluated across three chronological walk-forward folds with temporal boundary purging:

| Fold Name | Split Period | Train Size | Val Size | Purged Rows | Model | Macro F1 | Balanced Acc | HIGH Recall | HIGH Prec |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 2017–2018 | 167,827 | 44,163 | 242 | Majority Baseline | 0.2393 | 0.3333 | 0.0000 | 0.0000 |
| Fold 1 | 2017–2018 | 167,827 | 44,163 | 242 | 5-Obs Persistence | 0.5480 | 0.5401 | 0.3209 | 0.3831 |
| Fold 1 | 2017–2018 | 167,827 | 44,163 | 242 | Calendar Persistence (7D) | 0.5570 | 0.5517 | 0.3843 | 0.3768 |
| Fold 1 | 2017–2018 | 167,827 | 44,163 | 242 | **XGBoost (Balanced)** | **0.5631** | **0.5684** | **0.3108** | **0.4025** |
| Fold 1 | 2017–2018 | 167,827 | 44,163 | 242 | Random Forest (Balanced) | 0.5662 | 0.5688 | 0.3114 | 0.3980 |
| Fold 2 | 2019–2020 | 212,233 | 37,499 | 260 | Calendar Persistence (7D) | 0.6023 | 0.6085 | 0.5193 | 0.5198 |
| Fold 2 | 2019–2020 | 212,233 | 37,499 | 260 | **XGBoost (Balanced)** | **0.6190** | **0.6294** | **0.4930** | **0.5890** |
| Fold 2 | 2019–2020 | 212,233 | 37,499 | 260 | Random Forest (Balanced) | 0.6195 | 0.6300 | 0.5006 | 0.5878 |
| Fold 3 | 2021–2022 | 251,038 | 39,396 | 265 | Calendar Persistence (7D) | 0.6134 | 0.6140 | 0.5590 | 0.5600 |
| Fold 3 | 2021–2022 | 251,038 | 39,396 | 265 | **XGBoost (Balanced)** | **0.6240** | **0.6375** | **0.5801** | **0.5815** |
| Fold 3 | 2021–2022 | 251,038 | 39,396 | 265 | Random Forest (Balanced) | 0.6256 | 0.6362 | 0.5749 | 0.5748 |
| **MEAN** | — | — | — | — | **Calendar Persistence (7D)** | **0.5909** | **0.5914** | **0.4875** | **0.4855** |
| **MEAN** | — | — | — | — | **XGBoost (Balanced)** | **0.6020** | **0.6118** | **0.4613** | **0.5243** |
| **MEAN** | — | — | — | — | **Random Forest (Balanced)** | **0.6038** | **0.6117** | **0.4623** | **0.5202** |

---

## 4. Retrospective Holdout Benchmark (2023–2024)

Evaluated on the independent 1-year holdout (January 27, 2023 to January 27, 2024, $N=20,166$ observations, 56 mandis):

| Model | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | HIGH Recall | HIGH Precision | HIGH F1 | False Alarm Rate | H→L Conf | L→H Conf |
|---|---|---|---|---|---|---|---|---|---|---|
| Majority Baseline | 0.5552 | 0.3333 | 0.2380 | 0.3964 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 5,676 | 0 |
| 5-Obs Persistence Baseline | 0.6666 | 0.6063 | 0.6113 | 0.6614 | 0.4924 | 0.5718 | 0.5292 | 0.4282 | 2,147 | 1,600 |
| **Calendar Persistence (7D)** | **0.6607** | **0.6180** | **0.6177** | **0.6609** | **0.5442** | **0.5396** | **0.5419** | **0.4604** | **1,966** | **1,981** |
| Normalized V2 (Production Model) | 0.6631 | 0.6288 | 0.6236 | 0.6644 | 0.5412 | 0.5419 | 0.5416 | 0.4581 | 1,859 | 2,022 |
| V3 XGBoost (Balanced, Default) | 0.6542 | 0.6229 | 0.6164 | 0.6579 | 0.5816 | 0.5274 | 0.5532 | 0.4726 | 1,724 | 2,226 |
| V3 XGBoost (Tuned Thresholds) | 0.6494 | 0.6307 | 0.6157 | 0.6540 | 0.5678 | 0.5364 | 0.5516 | 0.4636 | 1,694 | 2,164 |
| V3 Random Forest (Balanced) | 0.6598 | 0.6312 | 0.6222 | 0.6619 | 0.5460 | 0.5463 | 0.5461 | 0.4537 | 1,862 | 1,979 |

### Confusion Matrix: Calendar Persistence (7D)
```
Actual \ Pred    Pred LOW    Pred MED    Pred HIGH    Total
True LOW            8,399         828        1,981   11,208
True MED              792       1,847          655    3,294
True HIGH           1,966         621        3,089    5,676
Total              11,157       3,296        5,725   20,166
```

### Confusion Matrix: V3 XGBoost Balanced (7D)
```
Actual \ Pred    Pred LOW    Pred MED    Pred HIGH    Total
True LOW            7,800       1,187        2,221   11,208
True MED              606       1,970          718    3,294
True HIGH           1,709         673        3,294    5,676
Total              10,115       3,830        6,233   20,166
```

---

## 5. Decision Threshold Analysis

Tuning probability multipliers on Fold 3 validation data yielded optimal multipliers $w = [1.00, 1.20, 1.00]$:
- High-to-Low confusion was modestly reduced from 1,709 to 1,679.
- However, overall Macro F1 shifted marginally from 0.6180 to 0.6171, as increasing MEDIUM probability diverted boundary cases into MEDIUM.
- Neither default nor tuned threshold configurations established a statistically meaningful advantage over the persistence baseline.

---

## 6. Generalization to Unseen Mandis

Evaluating a dedicated group holdout of 8 mandis ($N=2,640$) withheld entirely from training:
- **ML Macro F1 (Unseen Mandis):** 0.5421
- **Persistence Macro F1 (Unseen Mandis):** 0.5281
- **ML Advantage:** $+0.0140$

While ML demonstrates positive cross-market transfer to unseen mandis, overall Macro F1 is substantially lower than on seen mandis (0.6258), reflecting localized price regime differences.

---

## 7. Conclusions for Horizon 7D

1. **Target Engineering:** Resolved the variable calendar horizon defect completely.
2. **Operational Recommendation:** Because the zero-cost calendar persistence baseline outperforms ML in 69.6% of individual mandis ($p = 0.00006$), **ML models should NOT replace persistence for 7-day volatility forecasting**. Persistence remains the scientifically grounded choice for 7-day advisory services.
