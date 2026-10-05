# Member 1 Rice Yield Model (WITH MANDI) Results

## 1. Feature List
- **Categorical:** ['District', 'Season']
- **Numerical Lags (Yield/Env):** 9 features
- **Numerical Lags (Mandi):** 16 features
*(Strictly pre-planting historical features. Zero current-season mandi data).*

## 2. Validation Metrics (Hyperparameter / Model Selection)
| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | 217.8 | 369.9 | 0.584 | 0.105 |
| Historical Mean | 441.8 | 510.7 | 0.207 | 0.177 |
| Ridge | 240.0 | 362.8 | 0.597 | 0.110 |
| Random Forest | 250.0 | 362.7 | 0.598 | 0.111 |
| XGBoost | 262.3 | 400.4 | 0.509 | 0.119 |

**Selected Model:** Ridge

## 3. Final Test Metrics (Out-of-Sample: 2017-2019)
| Model | MAE (kg/ha) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Persistence (Lag1) | 249.5 | 357.6 | 0.612 | 0.098 |
| **Ridge (ML)** | **270.2** | **354.1** | **0.619** | **0.103** |

> ⚠️ **Warning:** The ML model failed to beat the best baseline. Adding mandi data did not provide enough forward-looking signal to overcome the strong autocorrelation of yield.
