# AgroVision-ML — Member 2: Potato Commodity Pipeline

Complete engineering and machine learning specification for **Potato mandi modal price forecasting and short-term volatility estimation** in West Bengal wholesale agricultural markets.

---

## 1. Overview & Market Scope

Potato (*Solanum tuberosum*) is the highest-volume horticultural cash crop in West Bengal, which ranks as India's second-largest potato-producing state. Potato prices exhibit strong seasonal volatility, driven by winter harvesting cycles, cold-storage unloading surges, monsoon demand shifts, and interstate market transfers.

- **Market Focus:** 88 West Bengal APMC wholesale mandis across 19 agricultural districts.
- **Varieties Handled:** 13 commercial and regional varieties, predominantly *Jyoti* and *Chandramukhi*, alongside *Desi*, *Local*, *F.A.Q.*, *Red*, *Shimla*, *Jalander*, *Red Nanital*, *Great Scott*, *Kufri Giriraj*, and *Other*.
- **Temporal Horizon:** 2002-01-18 to 2024-02-02 (265,047 chronological market records).
- **Core Targets:**
  1. **Next-Day / Next-Observation Modal Price Change** (1–7 calendar days limit).
  2. **Short-Term Price Volatility Regime** (`LOW`, `MEDIUM`, `HIGH`) over a 5-observation forward window.

---

## 2. Benchmark Scorecard & Model Performance

### 2.1 Price Forecasting Model (XGBRegressor, 20 Features)
- **Objective:** `reg:squarederror` / unclipped price change target $\Delta p = p_{t+1} - p_t$.
- **Reconstruction:** $\hat{p}_{t+1} = p_t + \Delta \hat{p}$.
- **Performance:**
  - **$R^2$ Variance Explained:** **0.9240**
  - **Mean Absolute Error (MAE):** **₹32.80 / Quintal** (on market prices ranging ₹400 – ₹2,200/qtl)
  - **Root Mean Squared Error (RMSE):** **₹78.40 / Quintal**
  - **Percentage Accuracy ($100 - \text{MAPE}$):** **97.60%**
  - **Practical Tolerance ($\le ₹25$/qtl error):** **74.80%**

### 2.2 Volatility Classification Model (XGBClassifier, 19 Features)
Evaluated on the out-of-sample holdout test partition ($N = 53,010$ observations):

| Evaluation Metric | Trained XGBoost Model | Majority Baseline | Stratified Chance Baseline |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **50.88%** | 43.10% | 33.40% |
| **Balanced Accuracy** | **44.42%** | 33.33% | 33.10% |
| **Macro F1 Score** | **0.4610** | 0.2010 | 0.3200 |
| **MCC Correlation** | **0.2150** | 0.0000 | 0.0010 |
| **LOW Class F1** | **0.5820** | 0.6020 | 0.4210 |
| **MEDIUM Class F1** | **0.4120** | 0.0000 | 0.2240 |
| **HIGH Class F1** | **0.4930** | 0.0000 | 0.3150 |

> **Operational Insight for Potato:** Because potato prices are heavily influenced by speculative storage releases and seasonal gluts, price percentage swings can occur abruptly. Class-balanced sample weighting ensures the model remains sensitive to HIGH volatility alerts rather than collapsing into the majority class.

---

## 3. Feature Engineering Specification

The Potato model uses **20 point-in-time features** strictly avoiding temporal leakage:

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

All Potato assets are separated into dedicated commodity directories:

```text
member2-price-volatility/
├── artifacts/
│   └── potato/
│       ├── potato_price_model.pkl          ← Trained XGBoost price regressor
│       ├── potato_volatility_model.pkl     ← Trained XGBoost volatility classifier
│       ├── potato_feature_config.json      ← Full feature specification & metrics
│       ├── potato_volatility_config.json   ← Volatility thresholds & parameters
│       └── potato_group_mappings.json      ← Canonical strings for 88 mandis & 13 varieties
│
├── data/
│   ├── raw/
│   │   └── potato/
│   │       └── potato.csv                  ← Source Agmarknet CSV (265,047 rows)
│   │
│   └── processed/
│       └── potato/
│           ├── potato_cleaned.csv          ← Deduplicated and cleaned time series
│           ├── potato_timeseries.csv       ← Validated chronological groups
│           ├── potato_features_final.csv   ← Model-ready feature matrix
│           ├── potato_train.csv            ← 80% chronological train partition
│           ├── potato_test.csv             ← 20% chronological test partition
│           ├── potato_volatility_train.csv
│           └── potato_volatility_test.csv
```

---

## 5. How to Train the Potato Pipeline

To train the complete Potato pipeline from raw data to saved models:

```bash
# Full pipeline (Data cleaning -> Feature extraction -> Model training -> Evaluation)
python member2-price-volatility/run.py --commodity potato

# Train price regressor only (skipping volatility)
python member2-price-volatility/run.py --commodity potato --skip-volatility
```

### Programmatic Python Invocation:
```python
from src.pipeline import run_pipeline

results = run_pipeline("potato")
print(f"Potato Pipeline Completed: {results}")
```

---

## 6. How to Run Potato Predictions

### 6.1 Using Standalone Python Prediction API
```python
import pandas as pd
from src.models.predict import predict

# Provide 30+ historical daily observations for a West Bengal mandi
history = pd.DataFrame([
    {
        "Reported Date": "2024-01-01",
        "Market Name": "Champadanga",
        "Variety": "Jyoti",
        "Modal Price (Rs./Quintal)": 920.0,
        "Arrivals (Tonnes)": 150.0
    },
    # ... 30+ observations ...
    {
        "Reported Date": "2024-02-01",
        "Market Name": "Champadanga",
        "Variety": "Jyoti",
        "Modal Price (Rs./Quintal)": 980.0,
        "Arrivals (Tonnes)": 160.0
    }
])

result = predict(history, commodity="potato")
print("Predicted Potato Price: ₹", result["predicted_price"])
print("Market Volatility Regime:", result["volatility"])
```

### 6.2 Using Shared Bridge (`AgroVisionPredictor`)
```python
from shared.predict_bridge import AgroVisionPredictor

bridge = AgroVisionPredictor()
result = bridge.predict_price(history, commodity="potato")
print("Bridge Result:", result)
```

---

## 7. Agmarknet Market & Variety Mappings

The Potato model is trained on and validates against 88 unique APMC wholesale markets in West Bengal:
- **Major Production Hubs:** Champadanga, Sheoraphuly, Kalipur, Pandua, ChinSurah (Hooghly); Burdwan, Kalna, Katwa, Memari (Burdwan); Garbeta, Ghatal, Medinipur Sadar, Kolaghat (Medinipur); Dhupguri, Moynaguri, Jalpaiguri Sadar (Jalpaiguri); English Bazar, Samsi, Gajol (Malda); Uluberia, Howrah (Howrah); Bara Bazar / Posta Bazar (Kolkata).
- **Varieties:** Jyoti, Chandermukhi, Desi, Local, F.A.Q., Red, Shimla, Jalander, (Red Nanital), Great Scott, Kufri Giriraj, Potato, Other.

For the full JSON schema of verified combinations, see [`artifacts/potato/potato_group_mappings.json`](../../artifacts/potato/potato_group_mappings.json).
