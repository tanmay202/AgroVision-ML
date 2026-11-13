# Independent Forensic Audit of Member 2 Rice Market Volatility Classifier

**Auditor:** Senior Machine Learning Evaluation Engineer  
**Date:** November 12, 2026  
**Repository:** [AgroVision-ML](file:///D:/MCA/SEM3/AgroVision-ML)  
**Branch:** `main`  
**Audited Commit:** `4c446e5b8babbd267e9c39eb53b4db3671c69506`  
**Target Artifact:** `rice_volatility_normalized_model.pkl`  
**Matching Configuration:** `rice_volatility_normalized_config.json`  
**Execution Context:** Independent Forensic Verification & Validation  

---

## 1. Executive Summary

An independent, read-only forensic audit of the frozen **AgroVision Member 2 Rice Market Volatility Classification Model** (`rice_volatility_normalized_model.pkl`) was conducted. The model evaluates a 3-class price-normalized volatility target (`LOW`, `MEDIUM`, `HIGH`) over a forward 5-observation horizon ($[t+1 \dots t+5]$) normalized by the observation-day price $P_t$.

### Key Findings:
1. **Mathematical Reproducibility (100% Exact):**  
   Every reported test metric was reproduced to the exact decimal from the frozen artifact and canonical dataset (`member2-price-volatility/data/processed/rice_features_final.csv`, $N=315,079$).
   - Champion Macro F1 reproduced at **0.6247** (exact: $0.624655$).
   - Champion HIGH-volatility recall reproduced at **55.62%** (exact: $55.6153\%$).
   - Champion Accuracy reproduced at **0.6825** (exact: $0.682533$).
   - Persistence Baseline Macro F1 reproduced at **0.6021** (exact: $0.602102$).
   - Persistence Baseline HIGH recall reproduced at **51.91%** (exact: $51.9116\%$).
   - Uncorrected McNemar statistic reproduced at $\chi^2 = 18.6113$, $p = 1.6027 \times 10^{-5}$ ($N = 20,336$).
2. **Leakage & Temporal Isolation (Clean):**  
   Strict 5-day boundary temporal purging correctly eliminated 415 lookahead boundary rows from the training split ($N=292,971$, covering 2002-03-26 to 2023-01-21). Features strictly utilize information available at or before observation time $t$; no target lookahead was detected.
3. **Statistical Significance Caveat (Critical):**  
   The previously reported McNemar significance ($p = 1.60 \times 10^{-5}$) is **severely distorted by violation of the i.i.d. assumption**. Because consecutive forward 5-observation windows overlap by 4 observations (80% window overlap), the within-mandi target serial autocorrelation is **0.8338**, and 89.72% of consecutive observations share identical class labels. When clustered by mandi ($M=56$), the paired accuracy difference drops to a non-significant $p = 0.0670$ ($\alpha = 0.05$).
4. **Temporal Generalization & Drift:**  
   While the ML champion maintains a +2.89% accuracy advantage over the baseline in H1 2023 (Jan–Jul 2023, Acc 68.04% vs 65.15%), performance decays in H2 2023 (Aug 2023–Jan 2024), where the persistence baseline surpasses the ML classifier (ML Acc 68.48% vs Baseline Acc 69.12%). In Q3 2023 (Aug–Oct 2023), the baseline beats ML on both accuracy (66.95% vs 65.63%) and Macro F1 (0.5894 vs 0.5860).
5. **Operational Suitability:**  
   The champion misses **44.38%** of all HIGH-volatility episodes (recall: 55.62%), with 74% of those misses erroneously classified as `LOW` (zero volatility). Furthermore, predicted class probabilities are uncalibrated (Expected Calibration Error: 9.98% on HIGH, 16.87% on LOW) due to balanced sample weighting during XGBoost training. The classifier is safe for advisory/opt-in use via its existing `v2_normalized` routing, but is **not reliable as an autonomous risk trigger**.

---

## 2. Artifact Integrity

Artifact SHA-256 digests were computed independently prior to evaluation and re-verified following all diagnostic executions. No file modifications, training iterations, or hyperparameter alterations occurred.

| Artifact Role | File Path | SHA-256 Digest | Status |
|---|---|---|---|
| Frozen Champion Model | [rice_volatility_normalized_model.pkl](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/artifacts/rice_volatility_normalized_model.pkl) | `dfc83d89ae3e93d9abf225b669b82fd9e1788972797353529680f7937b614a9f` | Verified Identical |
| Frozen Deliverable Model | [rice_volatility_normalized_model.pkl](file:///D:/MCA/SEM3/AgroVision-ML/deliverables/member2-ml/artifacts/rice_volatility_normalized_model.pkl) | `dfc83d89ae3e93d9abf225b669b82fd9e1788972797353529680f7937b614a9f` | Verified Identical |
| Model Configuration | [rice_volatility_normalized_config.json](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/artifacts/rice_volatility_normalized_config.json) | `c57651ff4a54f7cdde0e3199c63750ac72d09d205530643d1ff670bce1396f61` | Verified Identical |
| Deliverable Config | [rice_volatility_normalized_config.json](file:///D:/MCA/SEM3/AgroVision-ML/deliverables/member2-ml/artifacts/rice_volatility_normalized_config.json) | `c57651ff4a54f7cdde0e3199c63750ac72d09d205530643d1ff670bce1396f61` | Verified Identical |
| Production Price Model | [rice_price_model.pkl](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/artifacts/rice_price_model.pkl) | `b501f6d0e94f866900741adacbf97a6e0f621219a4d64004aac380e99ec2b993` | Verified Untouched |

### Internal Artifact Metadata Verification
Inspection of the serialized dictionary inside `rice_volatility_normalized_model.pkl` confirmed:
- **Model Type:** `xgboost.sklearn.XGBClassifier`
- **Objective:** `multi:softprob` (`num_class=3`)
- **Hyperparameters:** `n_estimators=250`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.85`, `colsample_bytree=0.85`, `tree_method='hist'`
- **Weight Strategy:** `balanced` sample weighting via `sklearn.utils.class_weight.compute_sample_weight`
- **Feature Count:** Exactly 35 features, matching `EXPERIMENTAL_FEATURE_COLS` in [feature_engineering.py](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/experiments/feature_engineering.py#L18-L61)
- **Target Window:** 5 observations forward
- **Thresholds:** `low_medium_threshold = 0.014286`, `medium_high_threshold = 1.960784`
- **Training Metadata:** 292,971 observations from 2002-03-26 to 2023-01-21
- **Reported Holdout Window:** 20,336 observations from 2023-01-27 to 2024-01-27

---

## 3. Original Metric Reproduction

Evaluation was conducted over the exact held-out test window ($[2023-01-27, 2024-01-27]$) using the frozen champion model artifact and source dataset [rice_features_final.csv](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/data/processed/rice_features_final.csv). The sample size is exactly $N = 20,336$ (following boundary dropna on features and targets).

### Overall Metrics Table

| Metric | Previously Reported | Independently Reproduced | Absolute Discrepancy | Status |
|---|---|---|---|---|
| **Holdout Sample Size ($N$)** | 20,336 | 20,336 | 0 | EXACT |
| **Accuracy** | 0.6825 | 0.682533 | +0.000033 | REPRODUCED |
| **Balanced Accuracy** | 0.6320 | 0.632042 | +0.000042 | REPRODUCED |
| **Macro Precision** | Not reported | 0.618531 | — | AUDIT COMPUTED |
| **Macro Recall** | Not reported | 0.632042 | — | AUDIT COMPUTED |
| **Macro F1** | **0.6247** | **0.624655** | -0.000045 | **REPRODUCED** |
| **Weighted F1** | Not reported | 0.685751 | — | AUDIT COMPUTED |
| **MCC** | 0.4418 | 0.441834 | +0.000034 | REPRODUCED |
| **HIGH-Volatility Recall** | **55.62%** | **55.6153%** | -0.0047% | **REPRODUCED** |
| **HIGH-Volatility Precision** | 52.76% | 52.7578% | -0.0022% | REPRODUCED |
| **HIGH-Volatility F1** | 0.5415 | 0.541484 | -0.000016 | REPRODUCED |

*Note: Macro F1 ($0.6247$) is an unweighted average of per-class F1 scores and must not be conflated with overall classification accuracy ($0.6825$).*

---

## 4. Confusion Matrix

The multi-class confusion matrix for the Champion XGBoost Balanced model on the $N=20,336$ holdout observations was recomputed and confirmed to match the documented counts exactly:

### Champion XGBoost (Balanced) Holdout Confusion Matrix
| Actual \ Predicted | Predicted LOW | Predicted MEDIUM | Predicted HIGH | Total Actual (Support) |
|---|---|---|---|---|
| **Actual LOW** | **9,266** | 1,039 | 1,860 | **12,165** (59.82%) |
| **Actual MEDIUM** | 687 | **1,821** | 641 | **3,149** (15.48%) |
| **Actual HIGH** | 1,649 | 580 | **2,793** | **5,022** (24.70%) |
| **Total Predicted** | **11,602** (57.05%) | **3,440** (16.92%) | **5,294** (26.03%) | **20,336** (100.0%) |

### Persistence Baseline Holdout Confusion Matrix
| Actual \ Predicted | Predicted LOW | Predicted MEDIUM | Predicted HIGH | Total Actual (Support) |
|---|---|---|---|---|
| **Actual LOW** | **9,422** | 896 | 1,847 | **12,165** (59.82%) |
| **Actual MEDIUM** | 917 | **1,613** | 619 | **3,149** (15.48%) |
| **Actual HIGH** | 1,804 | 611 | **2,607** | **5,022** (24.70%) |
| **Total Predicted** | **12,143** (59.71%) | **3,120** (15.34%) | **5,073** (24.95%) | **20,336** (100.0%) |

### Analysis of Error Distributions:
1. **Severe HIGH-to-LOW Misclassifications:**  
   Out of 2,229 total errors on actual HIGH-volatility regimes, **1,649 (73.98%)** are predicted as `LOW` (flat price regime). Only 580 (26.02%) are predicted as `MEDIUM`. The model systematically predicts complete market tranquility when substantial volatility occurs.
2. **False Alarms on HIGH Volatility:**  
   Out of 5,294 total instances where the champion predicted `HIGH`, **1,860 (35.13%) were completely flat (`LOW`)**, and 641 (12.11%) were `MEDIUM`. Thus, **47.24% of all HIGH alerts are false alarms**.

---

## 5. Per-Class Performance

Performance across the three volatility classes reveals pronounced disparities in predictive power:

| Class | Model | Precision | Recall | F1-Score | Support ($N$) | Support (%) |
|---|---|---|---|---|---|---|
| **LOW** | **Champion XGBoost** | **0.7987** | **0.7617** | **0.7797** | **12,165** | **59.82%** |
| | Persistence Baseline | 0.7759 | 0.7745 | 0.7752 | 12,165 | 59.82% |
| | *Lift over Baseline* | *+0.0228* | *-0.0128* | *+0.0045* | — | — |
| **MEDIUM** | **Champion XGBoost** | **0.5294** | **0.5783** | **0.5527** | **3,149** | **15.48%** |
| | Persistence Baseline | 0.5170 | 0.5122 | 0.5146 | 3,149 | 15.48% |
| | *Lift over Baseline* | *+0.0124* | *+0.0661* | *+0.0381* | — | — |
| **HIGH** | **Champion XGBoost** | **0.5276** | **0.5562** | **0.5415** | **5,022** | **24.70%** |
| | Persistence Baseline | 0.5139 | 0.5191 | 0.5165 | 5,022 | 24.70% |
| | *Lift over Baseline* | *+0.0137* | *+0.0371* | *+0.0250* | — | — |

### Evaluation of Class Weakness:
- **Weakest Overall Class:** `HIGH` volatility is the weakest class by F1 score (**0.5415**) and precision (**0.5276**).
- **Secondary Weakness:** `MEDIUM` volatility has the lowest support (only 15.48% of holdout rows) and low precision (**0.5294**), though balanced weighting boosts its recall to 0.5783.
- **Accuracy Dominance:** `LOW` volatility accounts for 59.82% of the test set and exhibits the highest F1 score (**0.7797**). The model's headline accuracy of 68.25% is predominantly driven by correctly identifying flat market periods.

---

## 6. Class Imbalance Analysis

Target definitions and class thresholds were derived from the pre-2021 training set:
$$\text{Thresholds: } \text{LOW} \le 0.014286\%, \quad 0.014286\% < \text{MEDIUM} \le 1.960784\%, \quad \text{HIGH} > 1.960784\%$$

Our empirical check revealed that **exact zeros** constitute **56.62%** ($177,637$ rows) of the dataset, and there are **zero observations** in $(0.0, 0.014286\%]$. Therefore, the `LOW` class strictly isolates flat-pricing intervals ($0.0\%$ forward range over 5 days).

### Distribution Comparison Across Partitions

| Partition | Total Rows ($N$) | LOW Count (%) | MEDIUM Count (%) | HIGH Count (%) |
|---|---|---|---|---|
| **Full Dataset** | 315,079 | 177,637 (56.38%) | 63,165 (20.05%) | 74,277 (23.57%) |
| **Training Set ($< 2023-01-27$)** | 292,971 | 165,254 (56.41%) | 59,953 (20.46%) | 67,764 (23.13%) |
| **Walk-Forward Fold 1 (2017–18)** | 44,594 | 27,751 (62.23%) | 9,506 (21.32%) | 7,337 (16.45%) |
| **Walk-Forward Fold 2 (2019–20)** | 38,038 | 23,079 (60.67%) | 6,797 (17.87%) | 8,162 (21.46%) |
| **Walk-Forward Fold 3 (2021–22)** | 40,022 | 23,295 (58.21%) | 5,376 (13.43%) | 11,351 (28.36%) |
| **Retrospective Holdout (2023–24)** | 20,336 | 12,165 (59.82%) | 3,149 (15.48%) | 5,022 (24.70%) |

### Observations on Imbalance Dynamics:
1. `LOW` volatility consistently dominates every partition (ranging between 56.4% and 62.2%).
2. `HIGH` volatility prevalence expanded over time: from **16.45%** in Fold 1 up to **28.36%** in Fold 3, stabilizing at **24.70%** in the 2023–2024 holdout.
3. In Fold 1 (2017–2018), where `HIGH` volatility was lowest (16.45%), the model's HIGH recall collapsed to **31.54%** (vs 58.70% in Fold 3). This confirms that the model's recall on volatile states is highly sensitive to the underlying market regime.

---

## 7. Baseline Comparison

The Champion XGBoost (Balanced) model was compared against all standard non-informative and operational baselines on the exact $N=20,336$ holdout set:

| Baseline Candidate | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | HIGH Recall | HIGH Precision | Notes |
|---|---|---|---|---|---|---|---|
| **Always-Majority (Always LOW)** | 0.5982 | 0.3333 | 0.2495 | 0.4478 | 0.0000 | 0.0000 | Trivial zero-information baseline |
| **Stratified Dummy (Train Priors)** | 0.4238 | 0.3278 | 0.3270 | 0.4311 | 0.2280 | 0.2460 | Random sampling from class priors |
| **Uniform Random Dummy** | 0.3326 | 0.3304 | 0.3061 | 0.3591 | 0.3310 | 0.2470 | Equal 1/3 prior assignment |
| **Production Persistence Baseline** | **0.6708** | **0.6020** | **0.6021** | **0.6710** | **0.5191** | **0.5139** | Prior 5-day percentage range $[t-4 \dots t]$ |
| **Champion XGBoost (Balanced)** | **0.6825** | **0.6320** | **0.6247** | **0.6858** | **0.5562** | **0.5276** | Frozen 35-feature model |

### Comparative Insights:
- The Champion classifier easily outperforms all dummy baselines (Always-Majority accuracy: 59.82%, Macro F1: 0.2495).
- However, against the **Persistence Baseline**, the lift is moderate:
  - Accuracy: $+1.17\%$ ($0.6825$ vs $0.6708$)
  - Balanced Accuracy: $+3.00\%$ ($0.6320$ vs $0.6020$)
  - Macro F1: $+0.0226$ ($0.6247$ vs $0.6021$)
  - HIGH Recall: $+3.71\%$ ($55.62\%$ vs $51.91\%$)

---

## 8. Temporal Generalization & In-Period Drift

The frozen champion was trained strictly on data prior to `2023-01-27` (effective end date: `2023-01-21` after boundary purging). To evaluate temporal stability without data contamination, the $N=20,336$ holdout was segmented into chronological quarters and semesters:

### Quarter-by-Quarter Performance Drift

| Chronological Period | Sample Size ($N$) | ML Accuracy | Baseline Accuracy | ML Macro F1 | Baseline Macro F1 | ML HIGH Recall | Baseline HIGH Recall | Relative Advantage |
|---|---|---|---|---|---|---|---|---|
| **Q1 (2023-01-27 to 2023-04-30)** | 4,785 | 0.6792 | 0.6589 | 0.6238 | 0.5957 | 0.5348 | 0.5226 | ML outperforms (+2.03% Acc) |
| **Q2 (2023-05-01 to 2023-07-31)** | 5,643 | 0.6814 | 0.6452 | 0.6391 | 0.5900 | 0.5787 | 0.5250 | ML outperforms (+3.62% Acc) |
| **Q3 (2023-08-01 to 2023-10-31)** | 4,978 | **0.6563** | **0.6695** | **0.5860** | **0.5894** | **0.5096** | **0.5087** | **Baseline outperforms (+1.32% Acc)** |
| **Q4 (2023-11-01 to 2024-01-27)** | 4,930 | 0.7136 | 0.7130 | 0.6444 | 0.6313 | 0.6029 | 0.5171 | Tied on Acc; ML leads on Recall |

### Semester Breakdown
| Half-Year Partition | Sample Size ($N$) | ML Accuracy | Baseline Accuracy | ML Macro F1 | Baseline Macro F1 | ML HIGH Recall | Baseline HIGH Recall |
|---|---|---|---|---|---|---|---|
| **H1 (2023-01-27 to 2023-07-31)** | 10,428 | **0.6804** | 0.6515 | **0.6324** | 0.5926 | 0.5590 | 0.5239 |
| **H2 (2023-08-01 to 2024-01-27)** | 9,908 | 0.6848 | **0.6912** | 0.6149 | 0.6104 | 0.5523 | 0.5125 |

### Analysis of Generalization Limits:
1. **Performance Decay in H2:** In the second six months following the training cutoff, the Persistence Baseline matches or exceeds the ML model on overall accuracy (Baseline 69.12% vs ML 68.48%).
2. **Q3 Inversion:** During Q3 2023, the baseline is superior across accuracy (66.95% vs 65.63%) and Macro F1 (0.5894 vs 0.5860).
3. **Data Horizon Boundary:** In `rice_features_final.csv`, records exist up to 2024-02-01 (303 rows after 2024-01-27). However, because a 5-observation forward window within each mandi group is mandatory, **0 valid forward targets exist after 2024-01-27**. Thus, supplementary evaluation beyond 2024-01-27 is mathematically impossible without external data ingestion.

---

## 9. Leakage and Data Integrity Audit

A comprehensive forensic audit of features, targets, splits, and identifiers yielded the following findings:

1. **Feature Lookahead (Clean):**  
   The 35 features in [feature_engineering.py](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/experiments/feature_engineering.py) rely strictly on backward-looking operators:
   - `rolling(5)`, `rolling(7)`, `rolling(14)`, `rolling(30)` over $[t-W+1 \dots t]$
   - `shift(1)`, `shift(2)`, `shift(5)`
   - `ewm(span=5)`, `ewm(span=14)` over historical differences
   - Calendar features extracted from `Reported Date` at time $t$
   - Lagged arrivals: `arrival_lag_1` ($t-1$) and `arrival_rolling_mean_7` ($t-1$ backward)
   No future pricing, forward volatility, or contemporaneous arrival signal is leaked into feature matrices.
2. **Temporal Split & Horizon Purging (Clean):**  
   The forward volatility target uses prices $[t+1 \dots t+5]$. In [target_definitions.py](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/experiments/target_definitions.py#L116-L131), `apply_temporal_purge` checks `future_target_date >= val_start`. For the holdout starting `2023-01-27`, exactly **415 boundary rows** were purged from the training set. The latest training date is `2023-01-21`, guaranteeing that no training target observed prices on or after `2023-01-27`.
3. **Record Duplication (Clean):**  
   A duplicate check on `['Market Name', 'Variety', 'Reported Date']` over all 315,079 rows returned **0 duplicates**.
4. **Price Scale Normalization (Clean):**  
   The minimum observed price in the dataset is Rs. 240.0/Quintal; maximum is Rs. 29,000.0/Quintal. Zero or negative prices do not exist, eliminating division-by-zero artifacts.
5. **Feature/Target Boundary Alignment (Disjoint & Correct):**  
   Features terminate strictly at index $t$; targets initiate strictly at index $t+1$. No overlapping price point exists between the feature space and target calculation.

---

## 10. Statistical Significance Caveats

The previously reported McNemar significance test on paired holdout predictions yielded:
$$\text{Both Correct } (n_{00}) = 12,252, \quad \text{Both Wrong } (n_{11}) = 5,066$$
$$b \text{ (Champion correct, Baseline wrong)} = 1,628, \quad c \text{ (Baseline correct, Champion wrong)} = 1,390$$
$$\chi^2 = \frac{(|1628 - 1390| - 1)^2}{1628 + 1390} = 18.6113, \quad p = 1.6027 \times 10^{-5}$$

### Why the Conventional McNemar Test is Invalid Here:
McNemar's test relies strictly on the assumption that paired prediction errors are **independent and identically distributed (i.i.d.)**. In this dataset, this assumption is fundamentally violated:

1. **High Rolling Window Overlap:**  
   The target is defined over 5 consecutive daily observations. In the holdout set, the median spacing between observations within a mandi is **1.0 day** (mean: 1.34 days). Consequently, `target[t]` and `target[t+1]` share **4 out of 5 observations (80% mutual information)**.
2. **Extreme Serial Autocorrelation:**  
   Within-group lag-1 autocorrelation of the target label is **0.8338**. Furthermore, **89.72% of consecutive observations belong to the exact same volatility class**.
3. **Panel Clustered Structure:**  
   The 20,336 holdout observations come from **56 distinct mandis**, not 20,336 independent trials. Errors within a mandi are strongly correlated across time.

### Clustered Significance Re-Analysis:
When we aggregate paired discordance at the independent cluster level (by mandi, $M=56$):
- Mandis where ML net outperforms Baseline: **34 (60.7%)**
- Mandis where Baseline net outperforms ML: **18 (32.1%)**
- Mandis where ML ties Baseline: **4 (7.1%)**
- Mandi-level clustered mean accuracy difference: **$+1.40\%$**
- Cluster-adjusted paired $t$-statistic: **$t = 1.8320$**
- **Cluster-adjusted $p$-value:** **$p = 0.0670$**

**Conclusion:** When accounting for panel correlation and overlapping rolling targets, **the performance difference is NOT statistically significant at the conventional $\alpha = 0.05$ threshold ($p = 0.0670$)**. The conventional $p = 1.6 \times 10^{-5}$ represents a massive pseudo-replication artifact caused by counting autocorrelated observations as independent samples.

---

## 11. Operational Reliability & Inference Assessment

### 1. Probability Calibration & Confidence Bias
In [experiment_models.py](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/experiments/experiment_models.py#L38-L58), the champion was trained using `weight_strategy="balanced"`. Balanced sample weighting mathematically distorts XGBoost probability estimates away from true posterior class frequencies:
- **`LOW` Volatility:** Heavily under-confident (Expected Calibration Error: **16.87%**, Brier score: $0.1989$). At predicted probability $[0.4, 0.5)$, the empirical accuracy is **71.16%**.
- **`HIGH` Volatility:** Systematically over-confident (Expected Calibration Error: **9.98%**, Brier score: $0.1579$). At predicted probability $[0.3, 0.4)$, the empirical occurrence of HIGH volatility is only **20.51%**.
- **Lack of Post-Hoc Calibration:** No Platt scaling or isotonic regression was applied. Raw predicted probabilities from `predict_proba` cannot be interpreted as calibrated frequencies.

### 2. Inference vs Training Feature Parity
All 35 features generated by `_generate_normalized_volatility_features` in [inference.py](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/src/inference.py#L343-L433) were tested against batch extraction in `feature_engineering.py`. Parity was verified to within $< 10^{-5}$ across all columns.

### 3. Missing Value Handling
The inference pipeline handles missing inputs via:
```python
group_df[col] = pd.to_numeric(group_df[col], errors='coerce').ffill().bfill()
```
And any remaining infinite or NaN values in feature generation are filled with `0.0`. Operational stress testing verified that sporadic missing prices or arrivals are handled gracefully.

### 4. History Constraints & Cold Start
The pipeline strictly enforces a minimum history requirement:
```python
if len(group_df) < 30:
    raise InsufficientHistoryError(f"Insufficient history. Required: 30, Found: {len(group_df)}")
```
Inputs with fewer than 30 observations are rejected cleanly without runtime crashes.

### 5. Unseen-Market Behavior
In the holdout dataset, 100% of the 56 mandis were present in the training set (spatial leave-one-mandi-out cross-validation was never performed). However, our synthetic stress test confirmed that if an entirely new mandi (e.g. `NonExistentMarket_XYZ`) is provided with $\ge 30$ historical observations, the pipeline successfully computes features and returns a valid prediction because features are entirely statistical and do not depend on categorical mandi embeddings.

### 6. Architectural Safety & Backward Compatibility
In [inference.py](file:///D:/MCA/SEM3/AgroVision-ML/member2-price-volatility/src/inference.py#L470-L594), routing is version-gated:
- `predict(..., volatility_version="v1")` or default: returns the legacy persistence baseline schema.
- `predict(..., volatility_version="v2_normalized")`: invokes the normalized XGBoost classifier.
- Production price model `rice_price_model.pkl` (48 features) is completely isolated and never invoked by or coupled to the volatility classifier.

---

## 12. Final Verdict

### Explicit Answers to Forensic Mandate Questions:

1. **Does Macro F1 = 0.6247 reproduce?**  
   **YES.** Independently reproduced at **0.624655** (rounds to **0.6247**).
2. **Does HIGH-volatility recall = 55.62% reproduce?**  
   **YES.** Independently reproduced at **55.6153%** (rounds to **55.62%**).
3. **Is the classifier better than its baseline?**  
   **YES, moderately.** On the overall holdout, it improves Macro F1 by $+0.0226$ ($0.6247$ vs $0.6021$), balanced accuracy by $+0.0300$ ($0.6320$ vs $0.6020$), and HIGH recall by $+3.71\%$ ($55.62\%$ vs $51.91\%$). However, the advantage narrows to zero or reverses in the second half of the holdout period.
4. **Does it generalize to future unseen observations?**  
   **PARTIALLY / WITH LIMITATIONS.** Performance is stable in H1 2023, but degrades in H2 2023 where the persistence baseline equals or exceeds it. Furthermore, all 56 holdout mandis were present in training; cross-sectional generalization to new mandis has not been tested.
5. **Which volatility class is weakest?**  
   **`HIGH` volatility.** It exhibits the lowest F1 score (**0.5415**) and precision (**0.5276**). Nearly 44.4% of actual HIGH volatility events are missed, and 74% of those misses are misclassified as flat/LOW.
6. **Is HIGH-volatility detection reliable enough for operational use?**  
   **NO, not as an autonomous primary trigger.** With a 44.38% miss rate and a 47.24% false alarm rate (plus uncalibrated probabilities), it is unsuitable for critical alerting, though acceptable as a secondary advisory signal.
7. **Is model retraining justified by the evidence?**  
   **NO.** Retraining on the same data architecture will not solve the structural issues (80% window autocorrelation, uncalibrated probabilities, and label noise). What is required is post-hoc probability calibration (Platt scaling) and regime-adaptive monitoring, not retraining.
8. **Should the current frozen production model be retained?**  
   **YES, strictly as an opt-in versioned capability (`v2_normalized`).** The existing backward-compatible routing architecture preserves production stability while providing access to the normalized classifier.

---

### Audit Determination

$$\huge\textbf{FINAL VERDICT: PARTIAL PASS}$$

**Justification:**  
The model and reported evaluation metrics reproduce with 100% mathematical fidelity. Data leakage protections and point-in-time feature extraction are rigorously designed and fully verified. However, operational generalization is subject to severe caveats: the reported McNemar statistical significance is invalid due to massive serial target autocorrelation; the classifier misses nearly 45% of high-volatility events; the advantage over the persistence baseline evaporates in later chronological windows; and output probabilities are uncalibrated. The model is approved for retention as an opt-in release (`v2_normalized`), but should not replace the persistence baseline as the production default.

---

## 13. Remaining Questions & Gaps

1. **Spatial Generalization to Unseen Mandis:**  
   Because all 56 holdout mandis existed in the training set, spatial out-of-sample generalization could not be definitively measured. A dedicated spatial leave-one-group-out (LOGO) validation should be prioritized in future evaluations.
2. **Subsequent 2024 Data Ingestion:**  
   Dataset records end on 2024-02-01 with insufficient future horizon points to compute forward targets. Incorporating verified 2024–2026 mandi data is required to determine whether Q3 2023 baseline inversion was a temporary seasonal anomaly or permanent regime drift.
3. **Probability Calibration Post-Processor:**  
   The current artifact lacks a calibration layer. Fitting an isotonic regression or sigmoid calibrator on held-out validation folds should be prioritized before deploying class probabilities to downstream user-facing dashboards.
