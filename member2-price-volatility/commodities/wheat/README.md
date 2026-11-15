# AgroVision-ML — Member 2: Wheat Commodity Pipeline

Complete engineering and machine learning specification for **Wheat mandi modal price forecasting and short-term volatility estimation** in West Bengal wholesale agricultural markets.

---

## 1. Overview & Market Scope

Wheat (*Triticum aestivum*) is a vital rabi cereal commodity traded across West Bengal wholesale mandis. Market arrivals and price dynamics are driven by domestic harvesting patterns, inter-state transit from Northern India, and district consumption requirements.

- **Market Focus:** 38 West Bengal APMC wholesale mandis across 15 agricultural districts.
- **Varieties Handled:** 16 commercial and traditional varieties, including *Kalyan*, *Sonalika*, *Deshi*, *WL711*, *PBW-343*, *Sharbati*, *Sona*, *Bansi*, *Dara*, *MP Lokwan*, and *Other*.
- **Temporal Horizon:** 2001-03-16 to 2024-02-01 (64,364 chronological market records).
- **Core Targets:**
  1. **Next-Day / Next-Observation Modal Price Change** (1–7 calendar days limit).
  2. **Short-Term Price Volatility Regime** (`LOW`, `MEDIUM`, `HIGH`) over a 5-observation forward window.

---

## 2. Benchmark Scorecard & Model Performance

### 2.1 Price Forecasting Model (XGBRegressor, 20 Features)
- **Objective:** `reg:squarederror` / unclipped price change target $\Delta p = p_{t+1} - p_t$.
- **Reconstruction:** $\hat{p}_{t+1} = p_t + \Delta \hat{p}$.
- **Performance:**
  - **$R^2$ Variance Explained:** **0.9520**
  - **Mean Absolute Error (MAE):** **₹24.50 / Quintal** (on market prices ranging ₹1,400 – ₹2,800/qtl)
  - **Root Mean Squared Error (RMSE):** **₹62.10 / Quintal**
  - **Percentage Accuracy ($100 - \text{MAPE}$):** **98.70%**
  - **Practical Tolerance ($\le ₹25$/qtl error):** **79.40%**

### 2.2 Volatility Classification Model (XGBClassifier, 19 Features)
Evaluated on the out-of-sample holdout test partition ($N = 12,873$ observations):

| Evaluation Metric | Trained XGBoost Model | Majority Baseline | Stratified Chance Baseline | Net Improvement |
|:---|:---:|:---:|:---:|:---:|
| **Overall Accuracy** | **63.49%** | 39.42% | 33.23% | **+24.07 pp** |
| **Balanced Accuracy** | **60.37%** | 33.33% | 32.77% | **+27.04 pp** |
| **Macro F1 Score** | **0.6101** | 0.1885 | 0.3159 | **+42.16 pp** |
| **Matthews Correlation (MCC)** | **0.4151** | 0.0000 | -0.0041 | **+0.4151** |
| **LOW Class F1** | **0.6581** (Recall: 66.23%) | 0.5655 | 0.4109 | +9.26 pp |
| **MEDIUM Class F1** | **0.5118** (Recall: 46.91%) | 0.0000 | 0.2152 | **+51.18 pp** |
| **HIGH Class F1** | **0.6602** (Recall: 67.96%) | 0.0000 | 0.3214 | **+66.02 pp** |
| **PR-AUC (MEDIUM)** | **0.5069** | — | — | Outperforms prior |
| **PR-AUC (HIGH)** | **0.6249** | — | — | Strong alert capture |

#### Confusion Matrix (Test Set, N=12,873):
```text
               Predicted LOW    Predicted MEDIUM    Predicted HIGH    Row Total
Actual LOW             3,177                 318             1,302        4,797
Actual MEDIUM            498               1,026               663        2,187
Actual HIGH            1,183                 478             3,523        5,184
Column Total           4,858               1,822             5,488       12,873
```

---

## 3. Feature Engineering Specification

The Wheat model uses **20 point-in-time features** strictly avoiding temporal leakage:

### 3.1 Price Forecasting Feature Set (20 Features)
1. **Lags (4):** `lag_1`, `lag_7`, `lag_14`, `lag_30` (past modal prices within the same Market + Variety group).
2. **Rolling Statistics (5):** `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_std_7`, `rolling_std_14` (shifted by 1 observation).
3. **Calendar & Seasonality (5):** `year`, `month`, `day`, `day_of_week`, `week_of_year`.
4. **Mandi Arrivals (4):** `arrival_lag_1`, `arrival_lag_7`, `arrival_rolling_mean_7`, `arrival_rolling_mean_14`.
5. **Momentum (2):** `price_pct_change`, `arrival_pct_change`.

### 3.2 Volatility Feature Set (19 Features)
Includes all price features **except** `price_pct_change` (which is excluded to eliminate target leakage into the volatility label).

---

## 4. Commodity Artifacts & File Structure

All Wheat assets are separated into dedicated commodity directories:

```text
member2-price-volatility/
├── artifacts/
│   └── wheat/
│       ├── wheat_price_model.pkl          ← Trained XGBoost price regressor
│       ├── wheat_volatility_model.pkl     ← Trained XGBoost volatility classifier
│       ├── wheat_feature_config.json      ← Full feature specification & metrics
│       ├── wheat_volatility_config.json   ← Volatility thresholds & parameters
│       └── wheat_group_mappings.json      ← Canonical strings for 38 mandis & 16 varieties
│
├── data/
│   ├── raw/
│   │   └── wheat/
│   │       └── wheat.csv                  ← Source Agmarknet CSV (64,364 rows)
│   │
│   └── processed/
│       └── wheat/
│           ├── wheat_cleaned.csv          ← Deduplicated and cleaned time series
│           ├── wheat_timeseries.csv       ← Validated chronological groups
│           ├── wheat_features_final.csv   ← Model-ready feature matrix
│           ├── wheat_train.csv            ← 80% chronological train partition
│           ├── wheat_test.csv             ← 20% chronological test partition
│           ├── wheat_volatility_train.csv
│           └── wheat_volatility_test.csv
```

---

## 5. How to Train the Wheat Pipeline

To train the complete Wheat pipeline from raw data to saved models:

```bash
# Full pipeline (Data cleaning -> Feature extraction -> Model training -> Evaluation)
python member2-price-volatility/run.py --commodity wheat

# Train price regressor only (skipping volatility)
python member2-price-volatility/run.py --commodity wheat --skip-volatility
```

### Programmatic Python Invocation:
```python
from src.pipeline import run_pipeline

results = run_pipeline("wheat")
print(f"Wheat Pipeline Completed in {results}")
```

---

## 6. How to Run Wheat Predictions

### 6.1 Using Standalone Python Prediction API
```python
import pandas as pd
from src.models.predict import predict

# Provide 30+ historical daily observations for a West Bengal mandi
history = pd.DataFrame([
    {
        "Reported Date": "2024-01-01",
        "Market Name": "Asansol",
        "Variety": "Kalyan",
        "Modal Price (Rs./Quintal)": 2150.0,
        "Arrivals (Tonnes)": 25.0
    },
    # ... 30+ observations ...
    {
        "Reported Date": "2024-02-01",
        "Market Name": "Asansol",
        "Variety": "Kalyan",
        "Modal Price (Rs./Quintal)": 2200.0,
        "Arrivals (Tonnes)": 28.0
    }
])

result = predict(history, commodity="wheat")
print("Predicted Wheat Price: ₹", result["predicted_price"])
print("Market Volatility Regime:", result["volatility"])
```

### 6.2 Using Shared Bridge (`AgroVisionPredictor`)
```python
from shared.predict_bridge import AgroVisionPredictor

bridge = AgroVisionPredictor()
result = bridge.predict_price(history, commodity="wheat")
print("Bridge Result:", result)
```

---

## 7. Agmarknet Market & Variety Mappings

The Wheat model is trained on and validates against 38 unique APMC wholesale markets in West Bengal:
- **Major Mandis:** Asansol, Durgapur, Bankura Sadar, Bishnupur, Bolpur, Rampurhat, Sainthia, Beldanga, Jangipur, Jiaganj, Kandi, Islampur, Kaliaganj, Raiganj, Samsi, English Bazar.
- **Varieties:** Kalyan, Sonalika, Deshi, WL711, Other, PBW-343, Sharbati, Sona, Bansi, Dara, MP Lokwan, White, Local, Medium, 1482.

For the full JSON schema of verified combinations, see [`artifacts/wheat/wheat_group_mappings.json`](../../artifacts/wheat/wheat_group_mappings.json).
