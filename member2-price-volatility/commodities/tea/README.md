# AgroVision-ML — Member 2: Tea Commodity Pipeline (Reference Baseline)

Engineering specification for the **Tea commodity pipeline**, maintained primarily as a lightweight reference baseline and multi-commodity smoke-test suite within AgroVision-ML.

---

## 1. Overview & Market Scope

Tea (*Camellia sinensis*) is a premier commercial plantation crop in Northern West Bengal (Dooars, Terai, and Darjeeling). Market trading occurs via auction centers and regional mandis.

- **Market Focus:** 13 North Bengal wholesale auction mandis.
- **Varieties Handled:** 4 commercial grades and regional categories.
- **Dataset Size:** 6,432 historical records.
- **Role in Platform:** Serves as a fast, reproducible baseline dataset for rapid integration checks, multi-commodity unit testing, and bridge compatibility verification.

---

## 2. Commodity Artifacts & File Structure

```text
member2-price-volatility/
├── artifacts/
│   └── tea/
│       ├── tea_price_model.pkl          ← Trained XGBoost price regressor
│       ├── tea_volatility_model.pkl     ← Trained XGBoost volatility classifier
│       ├── tea_feature_config.json      ← Feature configuration
│       ├── tea_volatility_config.json   ← Volatility configuration
│       └── tea_group_mappings.json      ← Canonical strings for 13 mandis
│
├── data/
│   ├── raw/
│   │   └── tea/
│   │       └── tea.csv                  ← Source Agmarknet CSV (6,432 rows)
│   │
│   └── processed/
│       └── tea/
│           ├── tea_cleaned.csv
│           ├── tea_timeseries.csv
│           ├── tea_features_final.csv
│           ├── tea_train.csv
│           └── tea_test.csv
```

---

## 3. How to Train and Predict

```bash
# Train Tea pipeline
python member2-price-volatility/run.py --commodity tea
```

```python
import pandas as pd
from src.models.predict import predict

tea_data = pd.read_csv("member2-price-volatility/data/processed/tea/tea_test.csv").iloc[:5]
res = predict(tea_data, commodity="tea")
print("Tea prediction:", res)
```
