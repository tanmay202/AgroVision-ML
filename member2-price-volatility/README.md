# AgroVision-ML — Member 2: Rice Price Forecasting & Volatility

A time-aware machine learning pipeline for West Bengal Rice mandi price forecasting and short-term volatility estimation using historical market, variety, price, and arrival data.

---

## 1. Project Overview
Member 2 is responsible for creating a robust predictive pipeline for the Rice commodity within the broader AgroVision-ML ecosystem. This module focuses exclusively on **West Bengal, India**. 

The core objectives are:
- **Mandi Price Forecasting**: Predicting the future change in Modal Price at a Market + Variety level.
- **Volatility Estimation**: Classifying the intensity of price fluctuations over a short-term horizon.

The implementation strictly avoids data leakage through point-in-time chronological splits, extensive time-aware validation, and carefully decoupled APIs ensuring safe, production-ready inference.

---

## 2. Member 2 Deliverables

| Deliverable | Status |
|---|---|
| Clean mandi price and arrival data | ✅ Completed |
| Lag features | ✅ Completed |
| Rolling features | ✅ Completed |
| Seasonal features | ✅ Completed |
| Arrival-based features | ✅ Completed |
| Baseline price forecasting | ✅ Completed |
| Stronger price forecasting model | ✅ Completed (XGBoost) |
| Volatility definition | ✅ Completed |
| Volatility classification | ✅ Completed |
| Time-aware validation | ✅ Completed |
| Leakage prevention | ✅ Completed |
| Model export | ✅ Completed (Frozen Artifacts) |
| Inference pipeline | ✅ Completed |
| FastAPI integration | ✅ Completed |

---

## 3. System Architecture

```mermaid
flowchart TD
    A[Rice Mandi Data] --> B[Data Cleaning]
    B --> C[Time-Series Preparation]
    C --> D[Historical Feature Engineering]
    D --> E[Price Baselines]
    D --> F[Price XGBoost]
    F --> G[Frozen Rice Price Model]
    B --> H[Volatility Target]
    H --> I[Volatility Experiments]
    I --> J[Persistence Baseline]
    G --> K[Rice Inference Pipeline]
    J --> K
    K --> L[FastAPI]
    L --> M[/health]
    L --> N[/predict]
```

---

## 4. Dataset
The Rice data represents historical agricultural market records:
- **Source**: `data/processed/rice_cleaned.csv`
- **Rows**: 319,092 valid observations
- **Date Range**: 2001-03-16 to 2024-02-02
- **Markets**: 87 unique markets
- **Varieties**: 34 unique varieties

**Important Columns Used:**
- `State Name`
- `District Name`
- `Market Name`
- `Variety`
- `Group`
- `Arrivals (Tonnes)`
- `Min Price (Rs./Quintal)`
- `Max Price (Rs./Quintal)`
- `Modal Price (Rs./Quintal)`
- `Reported Date`

---

## 5. Data Cleaning
The data cleaning pipeline transforms raw CSV dumps into a unified time-series format:
- **Date parsing**: Standardized `Reported Date` into `YYYY-MM-DD`.
- **Numeric conversion**: Forced arrivals and price columns to float; dropped non-numeric garbage.
- **Deduplication**: Handled exact-date duplicates for the same Market + Variety by retaining the latest log.
- **Missing Prices**: Valid `Modal Price` rows were retained even if `Min Price` or `Max Price` were missing/invalid.
- **Extreme Movements**: Wild price swings were inspected and retained (representing true agricultural market shocks) rather than being blindly filtered out.
- **Chronological Sorting**: Enforced strict temporal ordering.

---

## 6. Time-Series Design
Data is strictly grouped by **Market Name + Variety**. 
Because agricultural markets report asynchronously, observations are grouped chronologically. Future observations are strictly walled off—they are never used to generate rolling windows, lags, or EWMA features for the current day. 

The **Forecast Horizon** predicts the price change to the *next available valid observation* within a 1 to 7 day limit. Gaps larger than 7 days break the continuity streak.

---

## 7. Price Feature Engineering
The pipeline uses **48 point-in-time features**, divided into four categories. Current and future targets are completely excluded.

**Base Features (20)**
`lag_1`, `lag_7`, `lag_14`, `lag_30`, `rolling_mean_7`, `rolling_mean_14`, `rolling_mean_30`, `rolling_std_7`, `rolling_std_14`, `year`, `month`, `day`, `day_of_week`, `week_of_year`, `arrival_lag_1`, `arrival_lag_7`, `arrival_rolling_mean_7`, `arrival_rolling_mean_14`, `arrival_pct_change`, `price_pct_change`

**Momentum Features (5)**
`price_change_1`, `price_change_7`, `price_change_14`, `accel_1_7`, `accel_1_14`

**Trend Features (13)**
`trend_slope_7`, `trend_slope_14`, `trend_slope_30`, `ewma_7`, `ewma_14`, `ewma_30`, `rolling_median_7`, `rolling_median_14`, `rolling_median_30`, `gap_mean_7`, `gap_mean_14`, `zscore_7`, `zscore_14`

**History Features (10)**
`move_count_7`, `move_count_14`, `move_count_30`, `pos_count_7`, `neg_count_7`, `streak_zero`, `streak_pos`, `streak_neg`, `last_non_zero_move`, `last_move_abs`

---

## 8. Price Target
The exact target is:
`future_price_change = future_modal_price - current_modal_price`

Where `future_modal_price` is the next valid observation (1–7 days away) for the exact same Market + Variety. 
For inference, the absolute future price is reconstructed natively inside the pipeline via:
`predicted_modal_price = current_modal_price + predicted_price_change`

---

## 9. Baseline Models
Two foundational baselines govern the evaluation process to prove the ML model is adding genuine predictive value:
- **Persistence Baseline**: Naïvely predicts that the price will not change (`future_price_change = 0`). Useful because agricultural prices often remain unchanged for days.
- **lag_1 Baseline**: Predicts that tomorrow's price movement will equal yesterday's price movement.

---

## 10. Final Frozen Rice Price Model
The production price model is a strictly frozen XGBoost Regressor.

**XGBRegressor Exact Parameters:**
- `n_estimators=1000`
- `max_depth=5`
- `learning_rate=0.05`
- `subsample=1.0`
- `colsample_bytree=0.85`
- `min_child_weight=3`
- `gamma=0`
- `reg_alpha=1`
- `reg_lambda=1`
- `objective=reg:pseudohubererror`
- `random_state=42`
- `tree_method=hist`

This model is frozen.
- **Model Artifact**: `artifacts/rice_price_model.pkl`
- **Config**: `artifacts/rice_price_final_config.json`

---

## 11. Price Model Validation
The dataset was split chronologically to prevent temporal leakage.

- **TRAIN**: 2002-05-28 to 2021-01-31 *(247,198 rows)*
- **VALIDATION**: 2021-02-01 to 2023-01-31 *(40,015 rows)*
- **FINAL HOLDOUT**: 2023-02-01 to 2024-02-01 *(20,465 rows)*

**Final Holdout Results (All Rows)**
*(Note: MAE/RMSE measure raw currency error. R² is the coefficient of determination, NOT an accuracy percentage).*

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Tuned 48-feature XGB** | **18.97** | **85.42** | **0.9806** |
| Step 5 Base (48-feat) | 19.28 | 88.27 | 0.9792 |
| Persistence | 19.43 | 91.80 | 0.9776 |
| lag_1 | 26.30 | 102.47 | 0.9720 |

---

## 12. Movement Performance
Evaluating the model exclusively on the **3,883 rows** where the price *actually changed*. 

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **lag_1** | **85.37** | 197.33 | 0.8992 |
| Tuned 48-feature XGB | 96.52 | **193.11** | **0.9034** |
| Step 5 Base | 99.83 | 201.92 | 0.8944 |
| Persistence | 102.43 | 210.74 | 0.8850 |

*Finding: The `lag_1` heuristic achieves a lower movement-only MAE, while the Tuned XGBoost model achieves superior movement-only RMSE (handling extreme shifts better) and R².*

---

## 13. Zero-Change Performance
Evaluating the model exclusively on the **16,582 rows** where the price *stayed exactly the same*.

| Model | MAE | RMSE |
|---|---|---|
| **Persistence** | **0.00** | **0.00** |
| Step 5 Base | 0.42 | 8.28 |
| Tuned 48-feature XGB | 0.81 | 16.48 |

*Finding: The overall dataset is heavily dominated by zero-change days, anchoring the strength of the Persistence Baseline.*

---

## 14. Horizon Performance
Model accuracy based on the distance (in days) to the next available valid observation.

| Horizon | Rows | MAE | RMSE |
|---|---|---|---|
| **1 day** | 17,419 | 15.74 | 69.09 |
| **2 days** | 1,739 | 26.01 | 118.68 |
| **3 days** | 721 | 48.41 | 151.84 |
| **4 days** | 314 | 50.14 | 244.55 |
| **5 days** | 142 | 52.03 | 99.01 |
| **6 days** | 74 | 111.94 | 232.71 |
| **7 days** | 56 | 46.32 | 123.52 |

---

## 15. Volatility Definition
Volatility classifies the intensity of price fluctuations over a short-term horizon.
- **Target Definition**: `max(price) - min(price)` across the next 5 consecutive valid observations within the same Market + Variety group.
- **Classes**:
  - `LOW`: `<= 0.5`
  - `MEDIUM`: `> 0.5` and `<= 50.0`
  - `HIGH`: `> 50.0`

---

## 16. Volatility Model Experiments
Extensive ML approaches were tested to predict Volatility:
- XGBoost (balanced weights vs no weights vs mild weights)
- Probability decision adjustment
- Time-aware dynamic thresholds
- Random Forest & Logistic Regression

**Key Finding**: The original balanced XGBoost strongly overpredicted HIGH volatility due to a major temporal shift in class distribution (inflation & market dynamics increased historical variance). Redesigning the XGBoost model fixed the overprediction bug, but the final, honest evaluation proved that ML could not beat a simple Persistence baseline.

---

## 17. Final Production Volatility Method
The final production Volatility method is the **Persistence Baseline**. 
*(We do NOT use XGBoost for production Volatility).*

The inference pipeline maps the historical range of the *past* 5 observations directly into the future volatility class.

**Final Holdout Performance:**
- **Persistence (Production)**: Accuracy = 0.663 | Macro F1 = 0.593 | Balanced Accuracy = 0.593
- **XGBoost v2 (Discarded)**: Accuracy = 0.647 | Macro F1 = 0.522 | Balanced Accuracy = 0.504

---

## 18. Volatility Limitations
- The class distribution shifted substantially over time (inflation pushed more rows into `HIGH` under static thresholds).
- Dynamic thresholding was tested but ultimately reduced predictive performance out-of-sample.
- `MEDIUM` class recall remains highly limited.
- The Volatility prediction is currently heuristic/persistence-based as complex modeling failed to add predictive lift over past variance.

---

## 19. Inference Pipeline
File: `src/inference.py` (Class: `RiceInferencePipeline`)
- Point-in-time safe architecture.
- Loads the frozen Price model and configuration.
- Generates all 48 features inline strictly from history.
- Predicts price (XGBoost) and evaluates volatility (Persistence).
- Completely memory-only; does not modify training state or save dirty CSVs to disk.

---

## 20. Training/Inference Parity
Step 10.5 aggressively tested pipeline correctness:
- **5 unique historical windows** were compared between the original training script and `inference.py`.
- All 48 features matched perfectly within `< 1e-5`.
- End-to-end prediction outputs matched within `< 1e-2`.
- Future-row poisoning tests strictly failed to affect the targeted date's output.
- **2/2 Parity Tests Passed.** (Parity prevents "Train-Serve Skew", a major cause of production ML failures).

---

## 21. FastAPI Service
File: `src/api.py`

### `GET /health`
Returns API status.
### `POST /predict`
Generates predictions using raw history payloads.

**Request Schema:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "history": [
    {
      "reported_date": "2023-09-01",
      "modal_price": 2000.0,
      "arrivals": 100.0
    }
  ]
}
```

**Response Schema:**
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "predicted_modal_price": 3121.78,
  "predicted_volatility_range": 60.0,
  "predicted_volatility_class": "HIGH"
}
```

---

## 22. API Error Handling
- **`400 Bad Request`**: Raised if history is empty or insufficient (e.g., `< 30` observations), preventing the `lag_30` feature from collapsing.
- **`422 Unprocessable Entity`**: Pydantic typing validation failures.
- **`503 Service Unavailable`**: Raised if the frozen artifacts fail to load on application startup.

---

## 23. Project Structure
```text
member2-price-volatility/
├── artifacts/
│   ├── rice_feature_config.json
│   ├── rice_price_final_config.json
│   ├── rice_price_model.pkl
│   ├── rice_volatility_final_config_v2.json
│   └── rice_volatility_model_v2.pkl
├── data/
├── src/
│   ├── features/
│   ├── inference.py
│   └── api.py
├── tests/
│   ├── test_api.py
│   ├── test_inference.py
│   ├── test_leakage.py
│   ├── test_pipeline.py
│   └── test_training_inference_parity.py
├── example_inference.py
├── api_readme.md
└── README.md
```

---

## 24. Testing
Extensive automated testing confirms pipeline health.

| Test Suite | Result | Command |
|---|---|---|
| API Tests | 6/6 | `pytest tests/test_api.py -v` |
| Parity Tests | 2/2 | `pytest tests/test_training_inference_parity.py -v` |
| Inference Tests | 5/5 | `pytest tests/test_inference.py -v` |
| Leakage Tests | 10/10 | `pytest tests/test_leakage.py -v` |
| Pipeline Tests | 2/2 | `pytest tests/test_pipeline.py -v -s` |

---

## 25. Leakage Prevention
Leakage prevention is the highest priority:
- Strict chronological Train/Validation/Holdout split.
- `shift(1)` enforcement on all lags and rolling statistics.
- Current `Modal Price` is intentionally excluded from the baseline `PRICE_FEATURES`.
- Extreme point-in-time filtering guarantees future dates mathematically cannot pollute inference.
- **10/10 Leakage tests passed.**

---

## 26. How to Run

**1. Environment Setup:**
(Refer to the root repository for `requirements.txt` / virtual environment setups).

**2. Run Tests:**
```bash
python -m pytest tests/test_api.py -v
python -m pytest tests/test_training_inference_parity.py -v
python -m pytest tests/test_inference.py -v
python -m pytest tests/test_leakage.py -v
python -m pytest tests/test_pipeline.py -v -s
```

**3. Run Inference Example:**
```bash
python example_inference.py
```

**4. Start FastAPI:**
```bash
uvicorn src.api:app --reload
```

---

## 27. API Usage

**Check API Health:**
```bash
curl -X GET "http://127.0.0.1:8000/health"
```

**Interactive Docs:**
Navigate to `http://127.0.0.1:8000/docs` in your browser.

---

## 28. Model Artifacts
Production Artifacts (Frozen & untouched):
- `artifacts/rice_price_model.pkl`
- `artifacts/rice_price_final_config.json`

*(Note: `rice_volatility_model_v2.pkl` is retained strictly as an experimental artifact. It is **not** the production predictor).*

---

## 29. Limitations
- **Zero-Change Dominance**: Mandi prices frequently stagnate, keeping the Persistence Baseline artificially strong.
- **Class Shift**: Agricultural volatility expanded greatly in recent years due to macroeconomic forces; static thresholds face distribution shift.
- **Diminishing Horizon Quality**: Forecast horizons > 3 days have drastically reduced sample sizes in the historical dataset.
- **Regional Bias**: This implementation is currently specialized explicitly for West Bengal Rice varieties.

---

## 30. Model Interpretation
This machine learning model learns predictive patterns from historical lag features, momentum, and arrival trends. It **does not establish causal relationships** and should be used strictly as a statistical forecasting tool to augment market intelligence.

---

## 31. Example Output
```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2023-10-31",
  "predicted_modal_price": 3121.78,
  "predicted_volatility_range": 60.0,
  "predicted_volatility_class": "HIGH"
}
```

---

## 32. Final Results Summary

| Component | Final Method | Main Result |
|---|---|---|
| **Price** | Tuned XGBoost | MAE 18.97, RMSE 85.42, R² 0.9806 |
| **Volatility** | Persistence | Accuracy 66.31%, Macro F1 0.593 |
| **Testing** | Pytest | 25 tests passed across 5 suites |

---

## 33. Responsibility
**Member 2 — Price ML Responsibilities:**
- Mandi price forecasting
- Volatility estimation
- Price/arrival feature engineering
- Baseline modeling
- Price model development
- Volatility evaluation
- Time-aware validation
- Leakage prevention
- Model export
- Inference/API integration

---

## 34. Technology Stack
- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **XGBoost**
- **FastAPI**
- **Pydantic**
- **Pytest**
