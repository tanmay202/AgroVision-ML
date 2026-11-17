# M2 Audit Report — Member 2 Price Models & Residuals → TimescaleDB / FastAPI Integration

**Audit Date:** 2026-11-01  
**Auditor:** Automated code-level inspection (read-only)  
**Scope:** Verify the `deliverables/member2-ml/` handoff package for wiring into Member 3's TimescaleDB + FastAPI backend  

---

## Section 1: Canonical Handoff & Integration Contract

### 1.1 Is `deliverables/member2-ml/` the canonical handoff package?

**Yes.** Both [`deliverables/member2-ml/README.md`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/README.md) and [`deliverables/member2-ml/integration_contract.md`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/integration_contract.md) explicitly declare this as the certified, production-ready deliverable:

> **Package Name:** `member2-ml`  
> **Status:** Certified & Ready for Production Integration  
> **Target Consumer:** Member 3 (Backend & Application Integration)

The package is a **self-contained, pure-Python module** with no FastAPI/uvicorn dependencies. It is intended to **supersede** the development workspace at `member2-price-volatility/` for integration purposes. The older `member2-price-volatility/` retains the experiment scripts, training pipelines, raw data, and historical notebooks, but **`deliverables/member2-ml/` is the only directory Member 3 should import from**.

Key differences:

| Aspect | `deliverables/member2-ml/` | `member2-price-volatility/` |
|---|---|---|
| Entry class | `RicePricePredictor` | `RiceInferencePipeline` |
| Role | Consumer-facing deliverable | Development workspace |
| Horizons | Unified `predict(horizon=...)` API | Feature-flagged via env vars |
| Dependencies | Pure ML (no web framework) | Includes FastAPI, uvicorn, api.py |
| Artifacts | Frozen + checksummed in contract | Same `.pkl` files, duplicated |

### 1.2 Documented Consumption Pipeline

Per `integration_contract.md` §3–§4, Member 3's consumption flow is:

1. **Place artifacts** in an accessible directory (default `deliverables/member2-ml/artifacts/`, or set `AGROVISION_ARTIFACTS_DIR`).
2. **Import and instantiate** `RicePricePredictor`:
   ```python
   from src.predictor import RicePricePredictor, InsufficientHistoryError
   predictor = RicePricePredictor(artifacts_dir="path/to/artifacts")
   ```
3. **Supply ≥ 30 genuine historical observations** as `pd.DataFrame` or `List[Dict]`.
4. **Call `predictor.predict(history, horizon=...)`** with `horizon="short"`, `14`, or `30`.
5. **Consume the returned dict** containing `predicted_modal_price`, `predicted_price_change`, `current_modal_price`, and metadata.
6. **Wrap in FastAPI**, mapping exceptions to HTTP codes per §5 of the contract.
7. **Construct prediction intervals** from residual JSON files per §7.

---

## Section 2: Inference Entrypoint & Signatures

### 2.1 Primary Class: `RicePricePredictor`

**File:** [`deliverables/member2-ml/src/predictor.py`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/src/predictor.py)

```python
class RicePricePredictor:
    """
    Production-ready standalone rice price and volatility predictor.

    Parameters:
    -----------
    artifacts_dir : str or Path, optional
        Path to directory containing model `.pkl` and config `.json` files.
        If None, automatically searches:
        1. Environment variable `AGROVISION_ARTIFACTS_DIR`
        2. `deliverables/member2-ml/artifacts`
        3. `member2-price-volatility/artifacts`
        4. `artifacts` in repository root
    """

    def __init__(self, artifacts_dir: Optional[Union[str, Path]] = None):
```

### 2.2 Unified Dispatch: `predict()`

```python
def predict(
    self,
    history: Union[pd.DataFrame, List[Dict]],
    horizon: Union[int, str] = "short",
    market: Optional[str] = None,
    variety: Optional[str] = None,
    prediction_date: Optional[Union[str, pd.Timestamp]] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified prediction entry point across all three horizons.

    Parameters:
    -----------
    history : DataFrame or list of dicts
        Historical observations.
    horizon : int or str, default 'short'
        - 'short', 1, or '1-7': Short-term production forecast.
        - 14: 14-day calendar horizon forecast.
        - 30: 30-day calendar horizon forecast.
    """
```

**Dispatch logic (L426–L453):**
- `horizon in ("short", "1-7", 1, "1")` → calls `predict_short_term(...)`
- `horizon in (14, "14", "14d")` → calls `predict_horizon(history, horizon_days=14, ...)`
- `horizon in (30, "30", "30d")` → calls `predict_horizon(history, horizon_days=30, ...)`
- Otherwise → raises `ValueError`

### 2.3 `predict_short_term()` Signature

```python
def predict_short_term(
    self,
    history: Union[pd.DataFrame, List[Dict]],
    market: Optional[str] = None,
    variety: Optional[str] = None,
    prediction_date: Optional[Union[str, pd.Timestamp]] = None,
    include_volatility: bool = True,
    volatility_version: str = "v2_normalized",
) -> Dict[str, Any]:
```

### 2.4 `predict_horizon()` Signature

```python
def predict_horizon(
    self,
    history: Union[pd.DataFrame, List[Dict]],
    horizon_days: int,
    market: Optional[str] = None,
    variety: Optional[str] = None,
    prediction_date: Optional[Union[str, pd.Timestamp]] = None,
) -> Dict[str, Any]:
```

Only `horizon_days=14` or `horizon_days=30` accepted. Other values raise `ValueError`.

### 2.5 Input Structure

**Format:** `pandas.DataFrame` or `List[Dict[str, Any]]`

**Required columns** (with aliases auto-resolved via [`config.py`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/src/config.py)):

| Canonical Column | Accepted Aliases | Data Type | Notes |
|---|---|---|---|
| `Reported Date` | `reported_date`, `date` | `str` → `datetime` | Format `YYYY-MM-DD` |
| `Modal Price (Rs./Quintal)` | `modal_price`, `price` | `float` | Rs. per quintal |
| `Arrivals (Tonnes)` | `arrivals`, `arrival` | `float` | Tonnes |
| `Market Name` | `market`, `market_name`, `mandi` | `str` | e.g. `"Burdwan"` |
| `Variety` | `variety` | `str` | e.g. `"Common"` |

**Minimum rows:** 30 valid daily observations at or before the forecast origin date. Fewer raises `InsufficientHistoryError`.

### 2.6 Exact Return Structure

**All models predict a price _change_ (delta), not an absolute price.** The absolute price is reconstructed internally:

```python
# predictor.py L263-265 (short-term) and L379-381 (horizon)
predicted_change = float(model.predict(X)[0])
current_price = float(feat_df.iloc[-1][PRICE_COLUMN])
predicted_price = current_price + predicted_change
```

**Return dict keys (short-term example):**

```json
{
  "market": "Burdwan",
  "variety": "Common",
  "prediction_date": "2024-02-04",
  "forecast_origin_date": "2024-02-04",
  "horizon_days": 1,
  "requested_target_date": "2024-02-05",
  "current_modal_price": 2670.0,
  "predicted_modal_price": 2692.50,
  "predicted_price_change": 22.50,
  "target_date_convention": "Next valid observation within 1–7 calendar days",
  "model_name": "rice-price-xgb",
  "model_version": "1.0.0",
  "release_status": "production",
  "limitations": "Production baseline model for short-term 1-7 day rice price forecasting in West Bengal mandis.",
  "predicted_volatility_range": 45.0,
  "predicted_volatility_class": "MEDIUM",
  "volatility_version": "v2_normalized",
  "normalized_volatility_class": "MEDIUM",
  "normalized_volatility_probabilities": {"LOW": 0.125, "MEDIUM": 0.742, "HIGH": 0.133},
  "historical_pct_range_5": 1.68,
  "thresholds": {"low_medium_threshold": 0.014286, "medium_high_threshold": 1.960784}
}
```

**Return dict keys (14d/30d horizon example):** Same core keys plus:

```json
{
  "horizon_metadata": {
    "max_tolerance_days": 3,
    "train_period": ["2002-05-28", "2021-01-31"],
    "holdout_benchmark": {
      "holdout_rows": 18921,
      "model_mae": 59.87,
      "persistence_mae": 60.16,
      "model_rmse": 141.01,
      "persistence_rmse": 146.29
    }
  }
}
```

> **Key detail:** The raw XGBoost output is a **predicted price _change_ (delta in Rs./Quintal)**. The `predicted_modal_price` in the response dict is `current_modal_price + predicted_price_change`.

### 2.7 Comparison with Legacy `member2-price-volatility/src/inference.py`

The legacy file ([`member2-price-volatility/src/inference.py`](file:///home/aditya/projects/agrovision-ml/member2-price-volatility/src/inference.py)) contains the older `RiceInferencePipeline` class with the same fundamental inference logic but:
- Uses env-var feature flags (`is_horizon_forecasting_enabled`, `is_horizon_14d_enabled`, `is_horizon_30d_enabled`) for horizon gating
- Includes FastAPI-coupled patterns (API-oriented response structures)
- References the same 48-feature price models and v2_normalized volatility model

The deliverable `RicePricePredictor` is a cleaner, framework-agnostic refactoring of this class.

---

## Section 3: Exact Pinned Dependencies

### 3.1 `deliverables/member2-ml/requirements.txt`

**File:** [`deliverables/member2-ml/requirements.txt`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/requirements.txt)

```
# AgroVision-ML Member 2 ML Deliverables Requirements
# Core ML inference and validation dependencies (No web framework required)
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=1.7.0
joblib>=1.3.0
pytest>=7.0.0
```

### 3.2 Root `requirements.txt`

**File:** [`requirements.txt`](file:///home/aditya/projects/agrovision-ml/requirements.txt)

```
# Core data science
pandas>=2.0.0
numpy>=1.24.0

# Machine learning
scikit-learn>=1.3.0
xgboost>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
joblib>=1.3.0
openpyxl>=3.1.0       # For Excel file support (Member 1)
requests>=2.28.0      # For data download script (Member 2)

# Testing
pytest>=7.0.0
```

### 3.3 Dependency Comparison & Warning

| Package | `deliverables/member2-ml/` | Root `requirements.txt` | Conflict? |
|---|---|---|---|
| `numpy` | `>=1.24.0` | `>=1.24.0` | ✅ Match |
| `pandas` | `>=2.0.0` | `>=2.0.0` | ✅ Match |
| `scikit-learn` | `>=1.3.0` | `>=1.3.0` | ✅ Match |
| **`xgboost`** | **`>=1.7.0`** | **`>=2.0.0`** | ⚠️ **Floor mismatch** |
| `joblib` | `>=1.3.0` | `>=1.3.0` | ✅ Match |
| `pytest` | `>=7.0.0` | `>=7.0.0` | ✅ Match |
| `matplotlib` | Not required | `>=3.7.0` | N/A (viz only) |
| `seaborn` | Not required | `>=0.12.0` | N/A (viz only) |
| `requests` | Not required | `>=2.28.0` | N/A (data download) |

> **⚠️ XGBoost floor discrepancy:** The deliverable allows `xgboost>=1.7.0`, while the root requires `>=2.0.0`. The serialized `.pkl` models were trained with a specific version; ensure the deployed environment uses the same major version to guarantee `joblib.load()` compatibility. **No versions are strictly pinned (exact `==`) — all use `>=` minimum floors.**

---

## Section 4: Holdout Error Residuals & Sign Conventions

### 4.1 File Locations

| File | Path | Holdout Period | Samples |
|---|---|---|---|
| 7d residuals | [`deliverables/member2-ml/artifacts/rice_holdout_residuals_7d.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/rice_holdout_residuals_7d.json) | 2023-02-01 → 2024-02-01 | 20,098 |
| 14d residuals | [`deliverables/member2-ml/artifacts/rice_holdout_residuals_14d.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/rice_holdout_residuals_14d.json) | 2023-02-01 → 2024-02-01 | 18,921 |
| 30d residuals | [`deliverables/member2-ml/artifacts/rice_holdout_residuals_30d.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/rice_holdout_residuals_30d.json) | 2023-02-01 → 2024-02-01 | 17,860 |
| Short-term (1–7d) | [`deliverables/member2-ml/artifacts/rice_holdout_residuals.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/rice_holdout_residuals.json) | 2023-02-01 → 2024-02-01 | 20,465 |

### 4.2 Top-Level JSON Structure

All four files share the same schema. Top-level keys (from `rice_holdout_residuals_7d.json`):

```
["crop", "horizon_days", "model_version", "model_artifact",
 "model_artifact_sha256", "target_definition", "holdout_period",
 "n_holdout_samples", "metrics", "error_quantiles", "residuals"]
```

- `crop`: `"rice"`
- `horizon_days`: `7`, `14`, or `30`
- `model_version`: e.g. `"1.0.0-methodA"`, `"1.0.0-production-14d"`, `"1.0.0-production-30d"`
- `holdout_period`: `{"start": "2023-02-01", "end": "2024-02-01"}`
- `metrics`: dict of `mae`, `rmse`, `r2`, `mape`, `directional_accuracy`, `naive_baseline_mae`, etc.
- `error_quantiles`: dict of `p05`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`
- `residuals`: flat `List[float]` of individual residual values (one per holdout sample)

### 4.3 First 3 Records Per Horizon

**7d:** `[-0.06, -146.74, -3.21]`  
**14d:** `[144.07, 142.22, 45.26]`  
**30d:** `[149.12, 9.76, -31.97]`

### 4.4 Error Quantile Summaries

| Quantile | 7d | 14d | 30d |
|---|---|---|---|
| p05 | −99.93 | −104.31 | −195.53 |
| p10 | −49.82 | −98.12 | −97.02 |
| p25 | −0.09 | −2.73 | −3.74 |
| **p50** | **0.07** | **0.02** | **5.70** |
| p75 | 0.58 | 47.58 | 100.61 |
| p90 | 99.76 | 100.22 | 201.06 |
| p95 | 100.70 | 199.32 | 297.67 |

### 4.5 Sign Convention: `residual = actual − predicted`

**Verified by source code inspection.** The residual generation script at [`member2-price-volatility/experiments/run_direct_horizon_experiments.py`](file:///home/aditya/projects/agrovision-ml/member2-price-volatility/experiments/run_direct_horizon_experiments.py) states on line 15:

```
- Computes residual = actual - predicted.
```

And the computation on line 288:

```python
unrounded_res = true_price_holdout - pred_price_holdout
```

**This means:**
- **Positive residual** → the model **undershot** (actual was higher than predicted)
- **Negative residual** → the model **overshot** (actual was lower than predicted)
- **Prediction interval construction** (from `integration_contract.md` §7.2):
  ```python
  lower = predicted_modal_price + quantiles["p05"]   # p05 is negative → lower bound
  upper = predicted_modal_price + quantiles["p95"]    # p95 is positive → upper bound
  ```

This additive sign convention is consistent: adding a negative p05 to the prediction lowers the bound (covering cases where the model overshot), and adding a positive p95 raises it (covering cases where the model undershot).

---

## Section 5: Group Mappings & West Bengal Rice Identifiers

### 5.1 File Location

**File:** [`deliverables/member2-ml/artifacts/rice_group_mappings.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/rice_group_mappings.json)

### 5.2 Exact JSON Schema

Top-level keys:

```json
{
  "crop": "rice",
  "source_dataset": "member2-price-volatility/data/processed/rice_cleaned.csv",
  "training_date_range": {"start": "2001-03-16", "end": "2021-01-31"},
  "total_training_records": 257872,
  "columns": {
    "State Name": ["West Bengal"],
    "District Name": ["Bankura", "Birbhum", "Burdwan", ...],
    "Market Name": ["Alipurduar", "Asansol", ...],
    "Variety": ["1009 Kar", "AP Raw Rice PLR Soma", ...],
    "Group": ["Cereals"]
  },
  "missing_value_counts": {
    "State Name": 0, "District Name": 0,
    "Market Name": 0, "Variety": 0, "Group": 0
  },
  "observed_combinations_count": 297,
  "observed_combinations": [...]
}
```

> **Note:** This schema differs slightly from the one shown in `integration_contract.md` §8.1. The contract shows keys `train_date_max`, `total_training_rows`, and `observed_combinations` as `[["West Bengal", ...], ...]` (list of lists). The actual file uses `training_date_range`, `total_training_records`, `observed_combinations_count`, and `observed_combinations` as **list of dicts**. Member 3 should use the actual file format.

**Categorical counts:**
- States: 1 (`West Bengal`)
- Districts: 19
- Markets: 87
- Varieties: 34
- Groups: 1 (`Cereals`)
- Observed (Market, Variety) combinations: 297

### 5.3 Sample West Bengal Entries

Each entry in `observed_combinations` is a dict:

```json
[
  {
    "State Name": "West Bengal",
    "District Name": "Bankura",
    "Market Name": "Bankura Sadar",
    "Variety": "Common",
    "Group": "Cereals"
  },
  {
    "State Name": "West Bengal",
    "District Name": "Bankura",
    "Market Name": "Bankura Sadar",
    "Variety": "Fine",
    "Group": "Cereals"
  },
  {
    "State Name": "West Bengal",
    "District Name": "Bankura",
    "Market Name": "Bishnupur(Bankura)",
    "Variety": "Alur Sanna",
    "Group": "Cereals"
  }
]
```

**Key observations for backend ingestion:**
- All entries have `State Name` = `"West Bengal"` and `Group` = `"Cereals"`.
- Market names may contain parentheses: e.g. `"Bishnupur(Bankura)"`.
- Variety strings include non-obvious names like `"1009 Kar"`, `"AP Raw Rice PLR Soma"`, `"Alur Sanna"`, `"Amire Full Basmati"` — exact character-for-character matching is mandatory.

---

## Section 6: Volatility Models & Class Outputs

### 6.1 Baseline Volatility Config (`rice_volatility_config.json`)

**File:** [`deliverables/member2-ml/artifacts/rice_volatility_config.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/rice_volatility_config.json)

```json
{
    "threshold_method": "zero-aware split (57% zeros -> LOW, non-zeros split at median -> MEDIUM/HIGH)",
    "volatility_definition": "price range (max - min) of next 5 observations",
    "volatility_window": 5,
    "low_medium_threshold": 0.5,
    "medium_high_threshold": 50.0,
    "classes": ["LOW", "MEDIUM", "HIGH"]
}
```

This is the **v1 baseline** model. The `RicePricePredictor` uses raw absolute thresholds (Rs. 0.5 and Rs. 50.0) on the 5-day rolling price range.

### 6.2 Normalized Volatility Config (`rice_volatility_normalized_config.json`)

**File:** [`deliverables/member2-ml/artifacts/rice_volatility_normalized_config.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/rice_volatility_normalized_config.json)

```json
{
  "commodity": "rice",
  "volatility_version": "v2_normalized",
  "target_name": "future_pct_range_class",
  "volatility_definition": "percentage range of next 5 observations: (max - min) / price_t * 100",
  "volatility_window": 5,
  "classes": ["LOW", "MEDIUM", "HIGH"],
  "thresholds": {
    "low_medium_threshold": 0.014286,
    "medium_high_threshold": 1.960784,
    "derivation": "derived strictly from training set (<= 0.014286% is flat/LOW, non-zero median 1.960784% splits MEDIUM/HIGH)"
  },
  "features": [
    "past_5_range", "past_5_pct_range", "past_10_range", "past_10_pct_range",
    "past_20_range", "past_20_pct_range", "rolling_std_5", "rolling_std_7",
    "rolling_std_14", "rolling_std_30", "rolling_std_pct_5", "rolling_std_pct_14",
    "pct_change_1", "pct_change_2", "pct_change_5", "abs_pct_change_1",
    "abs_pct_change_5", "ewma_abs_diff_5", "ewma_abs_diff_14", "ewma_pct_vol_5",
    "ewma_pct_vol_14", "vol_trend_5_10", "vol_pct_trend_5_10", "vol_std_ratio_5_14",
    "nonzero_moves_5", "nonzero_moves_10", "streak_zero", "max_daily_pct_jump_5",
    "price_level", "price_to_mean_7", "month", "day_of_week", "week_of_year",
    "arrival_lag_1", "arrival_rolling_mean_7"
  ],
  "feature_count": 35,
  "model_architecture": {
    "type": "XGBClassifier",
    "objective": "multi:softprob",
    "weight_strategy": "balanced",
    "n_estimators": 250,
    "max_depth": 4,
    "learning_rate": 0.05
  }
}
```

### 6.3 Target Class Labels and Mapping

| Class Index | Label | Percentage Range Condition |
|---|---|---|
| 0 | `LOW` | ≤ 0.014286% (effectively flat / zero change) |
| 1 | `MEDIUM` | > 0.014286% and ≤ 1.960784% |
| 2 | `HIGH` | > 1.960784% |

These thresholds are percentage-based (price-normalized), unlike the v1 baseline which uses absolute Rs. thresholds.

### 6.4 Does the model output `predict_proba` or hard labels?

**The model outputs class probabilities via `predict_proba()`.** From [`predictor.py`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/src/predictor.py) L295–L301:

```python
proba = self._normalized_vol_model.predict_proba(X_norm)[0]
class_idx = int(np.argmax(proba))
v2_class = self._normalized_vol_classes[class_idx]
proba_dict = {
    cls: round(float(proba[i]), 4)
    for i, cls in enumerate(self._normalized_vol_classes)
}
```

The response includes **both** the hard label (argmax) and the full probability distribution:

```json
{
  "normalized_volatility_class": "MEDIUM",
  "normalized_volatility_probabilities": {
    "LOW": 0.1250,
    "MEDIUM": 0.7420,
    "HIGH": 0.1330
  }
}
```

The XGBClassifier architecture uses `"objective": "multi:softprob"`, which is specifically designed to output probability distributions over all classes.

### 6.5 Holdout Performance (v2_normalized)

From the config's `retrospective_holdout_summary`:
- Accuracy: 68.25%
- Macro F1: 0.6247
- Balanced Accuracy: 63.20%
- HIGH-class Recall: 55.62%
- McNemar test: χ² = 18.61, p = 1.6 × 10⁻⁵ (statistically significant improvement over baseline)

---

## Section 7: UI Prototype Inventory

### 7.1 File Location

**File:** [`artifacts_backup/checkpoint_ui_integration/ui/index.html`](file:///home/aditya/projects/agrovision-ml/artifacts_backup/checkpoint_ui_integration/ui/index.html)

### 7.2 Is This the Reference Dashboard Layout?

**Yes.** This is a fully self-contained, single-file HTML/CSS/JS prototype implementing the complete farmer-facing AgroVision dashboard. It serves as the **reference layout and styling guide** for the production frontend. It includes:

- **Sticky gradient header** with AgroVision branding, language toggle (English/Bengali), and settings
- **Hero section** with crop, mandi, and variety selectors
- **Stat grid** showing Current Modal Price, Predicted Price, Expected Yield, Market Volatility
- **Advice card** with actionable recommendation (Hold/Sell/Wait) and tone-based styling
- **Horizon selector** (3-button bar: Short-Term 1–7d / 14-Day Opt-In / 30-Day Experimental)
- **Forecast panel** with loading/error/results states, target window policy notice, SVG price chart, volatility badge, and model governance metadata
- **Tab navigation** (Predictions, Overview, Market, Weather, Tools)
- **Bottom nav bar** (5 icons: Predict, Home, Market, Weather, More)
- **Modal sheets** for: Why?, Profit Calculator, What-if, Storage Tips, Alerts, Help, Settings
- **Voice output** (Web Speech API) and **share** (Web Share API / clipboard fallback)
- **Bilingual support** (English / Bengali toggle via data-i18n attributes)

### 7.3 External Dependencies

**No external CSS or JS files are linked locally.** The prototype is entirely self-contained in the single `index.html` file:

- **CSS:** All styles are embedded in a single `<style>` block (~257 lines) within `<head>`
- **JavaScript:** All logic is embedded in a single `<script>` block (~420 lines) before `</body>`
- **External resources:** Only Google Fonts CDN (`fonts.googleapis.com`) for `Zilla Slab` display font
- **Favicon:** Inline data URI SVG (🌾 emoji)

### 7.4 Complete File Inventory of `artifacts_backup/checkpoint_ui_integration/`

```
artifacts_backup/checkpoint_ui_integration/
├── test_ui_api_integration.py      # API integration test script (8,638 bytes)
└── ui/
    └── index.html                  # Self-contained prototype (64,572 bytes)
```

Only two files exist. No separate CSS, JS, image, or font assets are stored alongside the HTML file.

### 7.5 API Contract Expected by the UI

The UI's JavaScript `fetchForecast()` function expects two FastAPI endpoints:

| Endpoint | Method | Payload | Used For |
|---|---|---|---|
| `{baseUrl}/predict` | `POST` | `{market, variety, prediction_date, history, volatility_version}` | Short-term (1–7d) |
| `{baseUrl}/predict-horizon` | `POST` | `{market, variety, prediction_date, horizon_days, history}` | 14d and 30d |
| `{baseUrl}/history` | `GET` | `?market=...&variety=...&limit=35` | Loading historical observations |

The base URL defaults to `http://127.0.0.1:8000` and is configurable via the Settings modal.

---

## Appendix: Model Handoff Manifest

A machine-readable manifest exists at [`deliverables/member2-ml/artifacts/model_handoff_manifest.json`](file:///home/aditya/projects/agrovision-ml/deliverables/member2-ml/artifacts/model_handoff_manifest.json) containing SHA-256 checksums, holdout metrics, and release tiers for every artifact. This should be loaded at application startup for integrity verification.

**Models registered in manifest:**

| Model ID | Artifact | Features | Horizon | Status |
|---|---|---|---|---|
| `rice-price-short-term` | `rice_price_model.pkl` | 48 | 1–7d | `production` |
| `rice-price-14d-production` | `rice_price_model_14d.pkl` | 48 | 14d | `opt-in` |
| `rice-price-30d-production` | `rice_price_model_30d.pkl` | 48 | 30d | `experimental` |
| `rice-price-7d-methodA` | `experiments/direct_horizon_models/rice_price_model_7d_methodA.pkl` | 48 | 7d | `experimental_research` |
| `rice-volatility-normalized` | `rice_volatility_normalized_model.pkl` | 35 | 5d | `production` |

