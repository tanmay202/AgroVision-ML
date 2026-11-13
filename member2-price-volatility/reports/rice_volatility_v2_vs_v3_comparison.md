# AgroVision Member 2 — Price Volatility V2 vs V3 vs Persistence Architectural & Empirical Comparison

**Module:** Member 2 — Price Volatility Machine Learning  
**Author:** Senior ML Research Engineer & Evaluation Architect  
**Scope:** Direct Comparison of V2 Normalized XGBoost, V3 Calendar Systems, and Persistence Baselines  
**Date:** November 2026  
**Status:** Certified Engineering Audit  

---

## 1. Executive Summary

This report delivers a rigorous side-by-side comparison between:
1. **Production Baseline (V1):** 5-observation & calendar persistence baselines.
2. **Production Normalized Model (V2):** XGBoost Balanced model trained on 5-observation row-shifted target (`future_pct_range`).
3. **Experimental System (V3):** Calendar-bounded 7-day and 14-day models with point-in-time features, threshold tuning, and probability calibration.

---

## 2. Structural & Architectural Comparison

| Dimension | Production Normalized V2 | Experimental System V3 (7D / 14D) | Impact / Resolution |
|---|---|---|---|
| **Target Horizon Definition** | 5 row shifts (`shift(-5)`) without calendar bounds | Fixed calendar window $[t, t+H]$ ($H \in \{7, 14\}$ days) | Eliminates calendar horizon drift entirely |
| **Max Target Horizon (Holdout)** | **86 calendar days** | **Exactly 7.0 days (7D) / 14.0 days (14D)** | Bounded temporal forecasting semantics |
| **Max Target Horizon (Training)** | **5,052 calendar days (13.84 years)** | **Exactly 7.0 days (7D) / 14.0 days (14D)** | Eliminates massive long-term multi-year target contamination |
| **Missing Trading Days** | Silently shifted across weeks/months/years | Real irregular days counted within $[t, t+H]$; NaN if 0 trades | Honest handling of market closures |
| **Feature Point-in-Time Safety** | 35 features | 38 features (added calendar-aligned past ranges) | Zero future lookahead verified |
| **Probability Calibration** | Uncalibrated (Raw softprob) | **Isotonic Regression & Platt Scaling** | ECE reduced from 9.60% to 2.79% |
| **Statistical Validation** | Pooled McNemar test only | **Clustered Paired $t$-test, Wilcoxon, & Block Bootstrap** | Correctly accounts for intra-mandi correlation |
| **Market Disaggregation** | Pooled report concealed small mandis | Explicit Small / Medium / Large tier breakdown | Exposes baseline dominance in smaller markets |
| **Unseen Mandi Testing** | Not tested | 8 withheld mandis ($N=2,640$) | Validates cross-market transferability |

---

## 3. Retrospective Holdout Benchmark Comparison (2023–2024)

Evaluated over the identical chronological holdout period (January 27, 2023 to January 27, 2024):

| Metric | Calendar Persistence (7D) | Calendar Persistence (14D) | Production V2 (XGBoost) | V3 XGBoost Balanced (7D) | V3 XGBoost Balanced (14D) | V3 RF Balanced (14D) |
|---|---|---|---|---|---|---|
| **Evaluated Target** | 7D Calendar | 14D Calendar | V3 14D Benchmark | 7D Calendar | 14D Calendar | 14D Calendar |
| **Sample Size ($N$)** | 20,166 | 20,329 | 20,329 | 20,166 | 20,329 | 20,329 |
| **Accuracy** | 0.6607 | 0.5760 | 0.5835 | 0.6542 | **0.5957** | 0.5916 |
| **Balanced Accuracy** | 0.6180 | 0.5650 | 0.5553 | 0.6229 | **0.5752** | 0.5729 |
| **Macro F1** | 0.6177 | 0.5655 | 0.5567 | 0.6164 | **0.5799** | 0.5755 |
| **Weighted F1** | 0.6609 | 0.5760 | 0.5725 | 0.6579 | **0.5931** | 0.5883 |
| **HIGH Recall** | 0.5442 | 0.5571 | 0.4457 | **0.5816** | **0.5668** | 0.5262 |
| **HIGH Precision** | 0.5396 | 0.5483 | **0.5663** | 0.5274 | 0.5635 | 0.5751 |
| **HIGH F1** | 0.5419 | 0.5527 | 0.4989 | **0.5532** | **0.5652** | 0.5496 |
| **HIGH False Alarm Rate** | 0.4604 | 0.4517 | **0.4337** | 0.4726 | 0.4365 | 0.4249 |
| **HIGH→LOW Confusion** | 1,966 | 2,141 | 2,908 | **1,724** | 2,291 | 2,424 |
| **LOW→HIGH Confusion** | 1,981 | 2,134 | **1,440** | 2,226 | 1,929 | 1,743 |
| **Macro ECE (Calibrated)** | — | — | 0.0970 | **0.0273** | **0.0325** | — |
| **Multiclass Brier Score** | — | — | 0.5122 | **0.4315** | **0.5137** | — |

---

## 4. Key Comparative Insights

### 4.1 Target Semantics Repair
V2's target was fatally ungrounded in calendar time. A farmer asking for volatility was receiving a prediction over the next 5 trading reports, which could occur over 5 days or 86 days. V3 fixes this:
- Target 7D predicts volatility strictly over the next 7 calendar days.
- Target 14D predicts volatility strictly over the next 14 calendar days.

### 4.2 HIGH Volatility Detection
In 14-day forecasts:
- V2 missed **55.43% of HIGH volatility events** (HIGH recall: 44.57%, 2,908 actual HIGH misclassified as LOW).
- V3 XGBoost Balanced captures **56.68% of HIGH volatility events** (+12.11 percentage points), reducing HIGH-to-LOW confusion by 617 cases.
- V3 increases HIGH F1 from 0.4989 (V2) to **0.5652** (+6.63 percentage points).

### 4.3 Market-Level Reality Check
- In 7D: Persistence beats V3 ML in **39 of 56 mandis (69.6%)**, with clustered significance ($p = 0.00006$).
- In 14D: Persistence beats V3 ML in **33 of 56 mandis (58.9%)**, with clustered significance ($p = 0.03125$).
- **Conclusion:** ML cannot be claimed as broadly superior to persistence across individual mandis.

---

## 5. Summary Conclusion

V3 represents a profound methodological improvement over V2 in data integrity, target validity, calibration quality, and evaluation rigor. However, because V3 does not demonstrate an undisputed clustered advantage over persistence across individual markets, **V3 should not be pushed directly to production as a total replacement for persistence**.

The persistence baseline should remain the active default for short-term 7-day volatility, while V3 14D should be staged as an experimental shadow candidate.
