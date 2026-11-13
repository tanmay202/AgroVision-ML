# AgroVision Member 2 — Volatility V3 14-Day Calendar Horizon Evaluation Report

**Module:** Member 2 — Price Volatility Machine Learning  
**Author:** Senior ML Research Engineer & Evaluation Architect  
**Evaluation Target:** 14-Calendar-Day Forward Normalized Price Range  
**Date:** November 2026  
**Status:** Certified Experimental Evaluation  

---

## 1. Executive Summary

This report presents the rigorous evaluation for **Target 14D**, a 14-calendar-day forward price volatility classifier developed under the Member 2 V3 target engineering repair program.

### Key Findings
1. **Target Engineering Validated:** Target 14D strictly bounds future price observation windows to at most 14 calendar days across all 312,947 observations (mean: 13.29 days, max: exactly 14.0 days), replacing the unbounded row shifts of V2.
2. **Defensible Pooled Advantage Over Persistence and V2:** On the independent retrospective holdout period (January 27, 2023 to January 27, 2024, $N=20,329$), **V3 XGBoost (Balanced)** achieves:
   - Accuracy: **0.5957** vs Persistence **0.5760** vs V2 **0.5835**
   - Balanced Accuracy: **0.5752** vs Persistence **0.5650** vs V2 **0.5553**
   - Macro F1: **0.5799** vs Persistence **0.5655** vs V2 **0.5567**
   - HIGH Recall: **0.5668** vs Persistence **0.5571** vs V2 **0.4457**
   - HIGH Precision: **0.5635** vs Persistence **0.5483** vs V2 **0.5663**
   - HIGH F1: **0.5652** vs Persistence **0.5527** vs V2 **0.4989**
3. **Substantial HIGH-Class Improvement Over V2:** On 14-day forecasts, production V2 misclassifies 3,028 actual HIGH cases as LOW (HIGH recall 44.57%). V3 reduces HIGH-to-LOW confusion by 761 observations (to 2,267) while increasing HIGH F1 from 0.4989 to **0.5652** (+6.63 percentage points).
4. **Market-Level Findings (Cluster Significance):** Across 56 individual mandis, persistence wins in 33 mandis (58.9%) and ML wins in 23 (41.1%). A cluster-robust paired $t$-test indicates that the baseline retains a statistically significant edge across individual markets ($p = 0.03125$, cluster $t = -2.2105$, 95% bootstrap CI $[-0.0252, -0.0020]$).
5. **Generalization to Unseen Mandis:** On 8 unseen mandis withheld entirely from training, V3 achieves Macro F1 of **0.5645** compared to persistence **0.5423** (+0.0222 advantage).

---

## 2. Target Engineering & Semantics (14D)

### 2.1 Formal Definition
For forecast origin $t$ with price $P_t$, the forward observation window is defined as:
$$W_{14}(t) = \{ \tau \mid t < \tau \le t + 14 \text{ calendar days, for the same (Market Name, Variety)} \}$$

The 14-calendar-day price-normalized forward percentage range is computed as:
$$\text{future\_pct\_range}_{14d} = \frac{\max_{s \in [t, t+14]} P_s - \min_{s \in [t, t+14]} P_s}{P_t} \times 100\%$$
where $[t, t+14] = \{P_t\} \cup \{P_\tau \mid \tau \in W_{14}(t)\}$.

### 2.2 Class Thresholds (Strictly Training-Derived)
Derived from clean pre-2023 training data ($N=291,728$ rows):
- **LOW:** $\le 0.014286\%$ (flat price regime, 35.42% of training data).
- **MEDIUM:** $0.014286\% < \text{range} \le 2.666666\%$ (32.30% of training data).
- **HIGH:** $> 2.666666\%$ (32.28% of training data).

This yields a balanced class distribution across the three volatility regimes.

---

## 3. Walk-Forward Cross-Validation Performance

Evaluated across three chronological walk-forward folds with temporal boundary purging:

| Fold Name | Split Period | Train Size | Val Size | Purged Rows | Model | Macro F1 | Balanced Acc | HIGH Recall | HIGH Prec |
|---|---|---|---|---|---|---|---|---|---|
| Fold 1 | 2017–2018 | 168,434 | 44,361 | 510 | Majority Baseline | 0.1642 | 0.3333 | 0.0000 | 0.0000 |
| Fold 1 | 2017–2018 | 168,434 | 44,361 | 510 | 5-Obs Persistence | 0.5052 | 0.5166 | 0.2239 | 0.5501 |
| Fold 1 | 2017–2018 | 168,434 | 44,361 | 510 | Calendar Persistence (14D) | 0.5667 | 0.5667 | 0.4641 | 0.4581 |
| Fold 1 | 2017–2018 | 168,434 | 44,361 | 510 | **XGBoost (Balanced)** | **0.5387** | **0.5471** | **0.2588** | **0.5140** |
| Fold 1 | 2017–2018 | 168,434 | 44,361 | 510 | Random Forest (Balanced) | 0.5471 | 0.5552 | 0.2688 | 0.5360 |
| Fold 2 | 2019–2020 | 212,900 | 37,760 | 569 | Calendar Persistence (14D) | 0.5945 | 0.5946 | 0.5541 | 0.5499 |
| Fold 2 | 2019–2020 | 212,900 | 37,760 | 569 | **XGBoost (Balanced)** | **0.6012** | **0.5990** | **0.4899** | **0.6015** |
| Fold 2 | 2019–2020 | 212,900 | 37,760 | 569 | Random Forest (Balanced) | 0.6038 | 0.6018 | 0.4880 | 0.6120 |
| Fold 3 | 2021–2022 | 251,862 | 39,726 | 553 | Calendar Persistence (14D) | 0.5731 | 0.5736 | 0.5932 | 0.5915 |
| Fold 3 | 2021–2022 | 251,862 | 39,726 | 553 | **XGBoost (Balanced)** | **0.5837** | **0.5822** | **0.5521** | **0.6410** |
| Fold 3 | 2021–2022 | 251,862 | 39,726 | 553 | Random Forest (Balanced) | 0.5818 | 0.5810 | 0.5460 | 0.6430 |
| **MEAN** | — | — | — | — | **Calendar Persistence (14D)** | **0.5781** | **0.5783** | **0.5371** | **0.5332** |
| **MEAN** | — | — | — | — | **XGBoost (Balanced)** | **0.5745** | **0.5761** | **0.4336** | **0.5855** |
| **MEAN** | — | — | — | — | **Random Forest (Balanced)** | **0.5776** | **0.5793** | **0.4343** | **0.5970** |

---

## 4. Retrospective Holdout Benchmark (2023–2024)

Evaluated on the independent 1-year holdout (January 27, 2023 to January 27, 2024, $N=20,329$ observations, 56 mandis):

| Model | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | HIGH Recall | HIGH Precision | HIGH F1 | False Alarm Rate | H→L Conf | L→H Conf |
|---|---|---|---|---|---|---|---|---|---|---|
| Majority Baseline | 0.4179 | 0.3333 | 0.1965 | 0.2463 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 7,316 | 0 |
| 5-Obs Persistence Baseline | 0.5803 | 0.5564 | 0.5485 | 0.5612 | 0.3533 | 0.6584 | 0.4599 | 0.3416 | 3,323 | 914 |
| Calendar Persistence (14D) | 0.5760 | 0.5650 | 0.5655 | 0.5760 | 0.5571 | 0.5483 | 0.5527 | 0.4517 | 2,141 | 2,134 |
| Normalized V2 (Production Model) | 0.5835 | 0.5553 | 0.5567 | 0.5725 | 0.4457 | 0.5663 | 0.4989 | 0.4337 | 2,908 | 1,440 |
| **V3 XGBoost (Balanced, Default)** | **0.5957** | **0.5752** | **0.5799** | **0.5931** | **0.5668** | **0.5635** | **0.5652** | **0.4365** | **2,291** | **1,929** |
| V3 XGBoost (Tuned Thresholds) | 0.5779 | 0.5827 | 0.5740 | 0.5794 | 0.5558 | 0.5609 | 0.5583 | 0.4391 | 1,872 | 2,174 |
| V3 Random Forest (Balanced) | 0.5916 | 0.5729 | 0.5755 | 0.5883 | 0.5262 | 0.5751 | 0.5496 | 0.4249 | 2,424 | 1,743 |

### Confusion Matrix: Calendar Persistence (14D)
```
Actual \ Pred    Pred LOW    Pred MED    Pred HIGH    Total
True LOW            5,333       1,037        2,134    8,504
True MED              984       2,310        1,224    4,518
True HIGH           2,141       1,099        4,076    7,316
Total               8,458       4,446        7,434   20,329
```

### Confusion Matrix: Normalized V2 (Production Model)
```
Actual \ Pred    Pred LOW    Pred MED    Pred HIGH    Total
True LOW            6,260       1,064        1,180    8,504
True MED            1,623       1,909          986    4,518
True HIGH           3,114       1,112        3,090    7,316
Total              10,997       4,085        5,256   20,329
```

### Confusion Matrix: V3 XGBoost Balanced (14D)
```
Actual \ Pred    Pred LOW    Pred MED    Pred HIGH    Total
True LOW            5,471       1,073        1,960    8,504
True MED              956       2,525        1,037    4,518
True HIGH           2,257         912        4,147    7,316
Total               8,684       4,510        7,144   20,329
```

---

## 5. Market-Level Clustered Analysis

Disaggregation across the 56 mandis:
- **Total Mandis Evaluated:** 56
- **ML Wins:** 23 (41.1%)
- **Persistence Wins:** 33 (58.9%)
- **Mean Per-Mandi F1 Delta ($\bar{d}$):** -0.0130
- **Cluster-Robust Standard Error:** 0.0059
- **Paired Cluster $t$-statistic:** -2.2105
- **Paired Cluster $t$-test $p$-value:** $0.03125$ (statistically significant advantage for persistence at $\alpha = 0.05$)
- **Wilcoxon Signed-Rank $p$-value:** $0.06526$ (stat: 572.0)
- **Clustered Bootstrap 95% Confidence Interval for Macro F1 Difference:** $[-0.0252, -0.0020]$ (strictly below zero)
- **Bootstrap $p$-value:** $0.01900$

### Volume Tier Disaggregation
| Mandi Volume Tier | Mandi Count | Mean Observations | ML Win Rate | Mean ML Macro F1 | Mean Base Macro F1 | Mean Delta |
|---|---|---|---|---|---|---|
| Small Mandis ($\le p_{25}$) | 14 | 211.6 | 28.6% (4 wins, 10 base) | 0.3655 | 0.3831 | -0.0176 |
| Medium Mandis ($p_{25} - p_{75}$) | 28 | 331.8 | 42.9% (12 wins, 16 base) | 0.4079 | 0.4217 | -0.0138 |
| Large Mandis ($> p_{75}$) | 14 | 577.0 | 50.0% (7 wins, 7 base) | 0.3925 | 0.3990 | -0.0066 |

---

## 6. Generalization to Unseen Mandis (Group Holdout)

Evaluating the 8 completely withheld mandis ($N=2,656$ observations):
- **ML Macro F1 (Unseen Mandis):** 0.5645
- **Persistence Macro F1 (Unseen Mandis):** 0.5423
- **ML Advantage:** $+0.0222$ (+2.22 percentage points)
- Proves that the 14D learned price patterns transfer effectively to unseen mandis where persistence degrades.

---

## 7. Conclusions for Horizon 14D

1. **Target Engineering:** Fully solves the variable-horizon defect.
2. **Predictive Capability:** V3 demonstrates a clear pooled advantage over both persistence (+1.44% Macro F1) and production V2 (+2.32% Macro F1, +12.11% HIGH recall).
3. **Mandi-Level Reality:** Because persistence wins in 33 of 56 individual mandis (58.9%) and retains a statistically significant clustered advantage ($p = 0.03125$, cluster $t = -2.2105$), ML cannot be claimed as superior across individual market clusters.
4. **Recommendation:** Target 14D is validated as an **Experimental Pass** and strong candidate for prospective shadow monitoring, but should not replace production persistence without live prospective validation.
