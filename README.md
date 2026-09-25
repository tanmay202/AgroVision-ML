# AgroVision-ML

> **Agricultural commodity intelligence platform** — ML pipelines for tea mandi arrival forecasting, price prediction, and volatility classification.

Built as a 3-member team project using Indian Agmarknet data.

---

## Team Responsibilities

| Member | Role | Module | Output |
|--------|------|--------|--------|
| **Member 1** | Yield/Arrival ML | `member1-yield-model/` | `yield_model.pkl` |
| **Member 2** | Price & Volatility ML | `member2-price-volatility/` | `{commodity}_price_model.pkl`, `{commodity}_volatility_model.pkl` |
| **Member 3** | Engineering & API | FastAPI service (separate repo) | REST API, Dashboard |

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data
```bash
python scripts/download_data.py --commodity tea
```
This saves to both member data directories automatically.

### 3. Run Both Pipelines
```bash
# Full run (both members, tea data)
python run_all.py

# Fast mode (skip XGBoost tuning)
python run_all.py --no-tune

# Individual members
python run_all.py --member 1
python run_all.py --member 2
```

### 4. Run Members Individually
```bash
# Member 1
cd member1-yield-model
python main.py --no-tune

# Member 2
cd member2-price-volatility
python run.py --commodity tea
```

---

## Project Structure

```
AgroVision-ML/
├── run_all.py                       ← Unified orchestrator (run both pipelines)
├── requirements.txt                 ← All dependencies (merged from both members)
├── complete_guide.md                ← Full technical guide
├── README.md                        ← This file
│
├── scripts/
│   └── download_data.py             ← Data.gov.in API download script
│
├── shared/
│   └── predict_bridge.py            ← Member 3 integration API
│
├── member1-yield-model/             ← MEMBER 1: Arrival forecasting
│   ├── main.py                      ← Entry point
│   ├── data/raw/                    ← Place tea_cleaned.csv here
│   ├── models/yield_model.pkl       ← Trained model (auto-generated)
│   ├── outputs/                     ← Plots & reports (auto-generated)
│   └── src/                         ← Pipeline source code
│
└── member2-price-volatility/        ← MEMBER 2: Price + volatility
    ├── run.py                        ← Entry point
    ├── data/raw/                    ← Place tea.csv here
    ├── artifacts/                   ← Trained models (auto-generated)
    └── src/                         ← Pipeline source code
```

---

## For Member 3 (FastAPI Integration)

```python
from shared.predict_bridge import AgroVisionPredictor
import pandas as pd

predictor = AgroVisionPredictor()

# Full prediction (both models)
data = pd.DataFrame([{ "lag_1": 240, "lag_7": 235, "rolling_mean_7": 242, ... }])
result = predictor.predict_all(data, commodity="tea")
print(result)
# {
#   "yield_arrivals_tonnes": 52.3,
#   "predicted_price_rs_quintal": 248.50,
#   "volatility": "LOW",
#   "commodity": "tea"
# }
```

See [`complete_guide.md`](complete_guide.md) for the full technical reference.
