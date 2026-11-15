# AgroVision-ML

> **Agricultural commodity intelligence platform** — ML pipelines for West Bengal rice crop yield prediction (Member 1) and mandi price forecasting & volatility classification (Member 2).

Built as a 3-member team project using Indian Agmarknet and NASA POWER agricultural data.

---

## Team Responsibilities

| Member | Role | Module | Primary Outputs & Artifacts |
|--------|------|--------|-----------------------------|
| **Member 1** | Crop Yield ML (tonnes/ha, kg/ha) | `member1-yield-model/` | `m1_improved_delta_ridge.pkl` |
| **Member 2** | Price & Volatility ML | `member2-price-volatility/`<br>`deliverables/member2-ml/` | **Multi-Commodity Support (Rice, Wheat, Potato, Tea):**<br>• [🌾 Rice](member2-price-volatility/commodities/rice/README.md): 48-feat XGBoost (`rice_price_model.pkl`), 14D & 30D horizons, V2 normalized volatility (`rice_volatility_normalized_model.pkl`), V3 shadow<br>• [🌾 Wheat](member2-price-volatility/commodities/wheat/README.md): 20-feat XGBoost (`wheat_price_model.pkl`), balanced classifier (`wheat_volatility_model.pkl`)<br>• [🥔 Potato](member2-price-volatility/commodities/potato/README.md): 20-feat XGBoost (`potato_price_model.pkl`), balanced classifier (`potato_volatility_model.pkl`)<br>• [🍵 Tea](member2-price-volatility/commodities/tea/README.md): Baseline models<br>• **Deliverable Package:** `RicePricePredictor` (pure Python) |
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

### 4. Run & Test Members Individually
```bash
# ─── Member 1 ───
cd member1-yield-model
python main.py --no-tune
pytest tests/ -v

# ─── Member 2 (Multi-Commodity Price & Volatility) ───
cd member2-price-volatility
python run.py --commodity rice    # Train Rice pipeline
python run.py --commodity wheat   # Train Wheat pipeline
python run.py --commodity potato  # Train Potato pipeline
python run.py --commodity all     # Train ALL commodities
pytest tests/ -v  # 166 tests passing across 15 test files

# ─── Member 2 Standalone Deliverable Package ───
pytest deliverables/member2-ml/tests/ -v  # 11 tests passing
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
│   ├── paths.py                     ← Centralized dynamic path manager
│   └── predict_bridge.py            ← Unified prediction bridge for Member 3
│
├── deliverables/
│   └── member2-ml/                  ← Standalone pure-Python ML deliverable for Member 3
│       ├── artifacts/               ← Bit-for-bit models, configs, residuals & mappings
│       ├── src/predictor.py         ← Standalone RicePricePredictor class
│       └── tests/test_predictor.py  ← Standalone regression test suite (11 tests)
│
├── member1-yield-model/             ← MEMBER 1: Rice Crop Yield Model
│   ├── scripts/                     ← Ingest, aggregate, train & inference scripts
│   ├── models/                      ← Candidate models (m1_improved_delta_ridge.pkl)
│   ├── data/                        ← Raw and processed district-seasonal datasets
│   └── tests/                       ← Comprehensive regression test suite (139 tests)
│
└── member2-price-volatility/        ← MEMBER 2: Multi-Commodity Price & Volatility Engine
    ├── run.py                       ← Entry point CLI (`--commodity rice|wheat|potato|all`)
    ├── commodities/                 ← Separated commodity guides (Rice, Wheat, Potato, Tea)
    │   ├── rice/README.md           ← Complete Rice architecture & operational guide
    │   ├── wheat/README.md          ← Complete Wheat architecture & operational guide
    │   ├── potato/README.md         ← Complete Potato architecture & operational guide
    │   └── tea/README.md            ← Complete Tea reference baseline guide
    ├── artifacts/                   ← Model weights, configs, residuals & mappings
    │   ├── rice/                    ← Separated Rice models (48-feat price, 14d/30d, V2 norm)
    │   ├── wheat/                   ← Separated Wheat models (20-feat price, balanced vol)
    │   ├── potato/                  ← Separated Potato models (20-feat price, balanced vol)
    │   ├── tea/                     ← Separated Tea baseline models
    │   └── v3/                      ← Prospective shadow candidate models (14d/7d)
    ├── data/
    │   ├── raw/                     ← Source raw CSVs (rice/, wheat/, potato/, tea/)
    │   └── processed/               ← Cleaned features & splits (rice/, wheat/, potato/, tea/)
    ├── src/                         ← Pipeline source code & shadow evaluation engines
    │   ├── api.py                   ← FastAPI service with shadow observability
    │   ├── config.py                ← Dynamic multi-commodity path manager
    │   ├── data_connector.py        ← Agmarknet API connector & drift detector
    │   ├── inference.py             ← Point-in-time RiceInferencePipeline
    │   ├── pipeline.py              ← Unified multi-commodity training pipeline
    │   ├── shadow_evaluation.py     ← Price horizon shadow evaluation engine
    │   ├── volatility_shadow.py     ← Volatility V3 prospective shadow pipeline
    │   └── models/predict.py        ← Multi-commodity prediction API (Rice, Wheat, Potato)
    ├── tests/                       ← Full automated test suite (146 tests)
    └── OPERATIONAL_SHADOW_MONITORING.md ← Runbook for prospective shadow pipelines
```

---

## Semantic Contract: Yield vs. Production vs. Arrivals

AgroVision strictly distinguishes between three different agricultural concepts:

1. **Crop Yield (`predicted_yield_tonnes_per_ha`, `predicted_yield_kg_per_ha`):**
   Crop productivity intensity predicted by Member 1 (tonnes/ha or kg/ha). Typical West Bengal rice range is 1.5–4.0 tonnes/ha (500–6000 kg/ha agronomic bounds).
2. **Crop Production (`estimated_production_tonnes`):**
   Estimated total harvest in tonnes. Calculated optionally when caller provides harvested/planted area in hectares:
   $$\text{Production (tonnes)} = \text{Yield (tonnes/ha)} \times \text{Area (ha)}$$
3. **Mandi Market Arrivals:**
   Quantity of produce physically traded at an APMC market wholesale yard on a given day (tonnes). This is a dynamic Agmarknet market observation consumed by Member 2, **never** output by Member 1's crop productivity model.

> **Deprecated Key Notice:** The legacy key `yield_arrivals_tonnes` conflated crop yield with mandi arrivals. It is deprecated and emits a `DeprecationWarning` upon access. It is scheduled for removal in AgroVision v3.0.

---

## For Member 3 (FastAPI Integration)

```python
from shared.predict_bridge import AgroVisionPredictor
import pandas as pd

predictor = AgroVisionPredictor()

# ─── 1. Member 1: Rice Yield Prediction ──────────────────────────────
yield_input = {
    "District": "Hooghly",
    "Season": "Winter",
    "Yield_lag1": 2850.0,
    "Area_ha": 150.0  # Optional: enables estimated_production_tonnes
}
yield_res = predictor.predict_yield(yield_input)
print(yield_res)
# {
#   "predicted_yield_kg_per_ha": 2813.5,
#   "predicted_yield_tonnes_per_ha": 2.8135,
#   "estimated_production_tonnes": 422.025,
#   "baseline_yield_kg_ha": 2850.0,
#   "warnings": []
# }

# ─── 2. Combined Prediction (predict_all) ───────────────────────────
data_history = pd.DataFrame([...])  # 35+ days of Agmarknet price history
result = predictor.predict_all(data_history, commodity="rice", yield_input=yield_input)
print(result["crop_yield"])    # Isolated Member 1 crop productivity
print(result["market_price"])  # Isolated Member 2 mandi price + volatility

# ─── 3. Multi-Commodity Price Forecasting ────────────────────────────
# Supports commodity="wheat", "potato", or "rice"
wheat_res = predictor.predict_price(data_history, commodity="wheat")
potato_res = predictor.predict_price(data_history, commodity="potato")
print("Wheat Price:", wheat_res["predicted_price_rs_quintal"], wheat_res["volatility"])
print("Potato Price:", potato_res["predicted_price_rs_quintal"], potato_res["volatility"])
```

See [`complete_guide.md`](complete_guide.md) for the full technical reference.
