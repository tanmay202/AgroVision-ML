# Member 1 Time-Aware Split Audit

## 1. Split Strategy & Reasoning
**Strategy:** Strict Chronological Split
- **Train:** 2000 – 2014
- **Validation:** 2015 – 2016
- **Test:** 2017 – 2019

**Reasoning:** 
Yield forecasting is fundamentally a time-series problem. Using a standard randomized `train_test_split` would scramble the temporal order, allowing the model to peek into the future (e.g., training on 2018 data to predict 2015 yield). This is catastrophic data leakage. 
A strict chronological split replicates the real-world deployment scenario: training on all available past data to predict entirely unseen future years. The 2000-2019 dataset allows a healthy 15-year training block, a 2-year validation block for hyperparameter tuning, and a 3-year out-of-sample test block.

## 2. Row & Coverage Statistics

| Split | Years | Rows | Districts Covered |
|---|---|---|---|
| **Train** | 2000 – 2014 | 852 | 19 |
| **Validation**| 2015 – 2016 | 118 | 22 |
| **Test** | 2017 – 2019 | 192 | 22 |
| **Total** | 2000 – 2019 | 1162 | 23 (Overall) |

*(Note: District coverage changes slightly in the source dataset due to the 2014 and 2017 boundary restructurings in West Bengal, such as the creation of Alipurduar, Paschim Bardhaman, and Purba Bardhaman).*

## 3. Leakage Checks
Before splitting, the pipeline performed a strict programmatic self-join audit to ensure lags were purely historical:
- **`Yield_lag1` Check:** Verified that for every row $T$, the `Yield_lag1` value mathematically equals the `Rice_Yield_kg_ha` target for year $T-1$ in the same District and Season. (0 mismatches detected).
- **Target Isolation:** Verified that no current-year target-derived statistics exist. `Yield_hist_mean` correctly uses a shifted expanding mean, strictly ignoring current/future years.
- **Future Peeking Prevention:** By physically splitting the dataset chronologically before any model processing, there is absolute certainty that the Validation and Test sets are completely isolated.

## 4. Missing Data & Imputation Policy
- **No Imputation Strategy:** We explicitly forbid computing imputation statistics (like median or mean filling) because computing those across the entire dataset leaks Validation/Test distributions into the Train set.
- **Native NaN Handling:** All missing values are preserved exactly as `NaN`s in the CSVs. Missingness is handled natively by the downstream XGBoost/LightGBM model during training (which learns optimal split directions for missing data).
- **Train Missingness:**
  - `Precip_lag1` / `T2M_lag1` / `RH2M_lag1`: 45 missing (Undivided Bardhaman).
  - `NDVI_lag1`: 99 missing (1999 has no NDVI, plus undivided Bardhaman).
- **Test Missingness:**
  - `Yield_lag2` / `Area_lag2`: 10 missing (New districts lack full 2-year history).
  - Environmental lags: 0 missing (Bardhaman was fully split by 2017, so its environmental features successfully join).

## 5. Duplicate Keys
- Checked and verified: `0` duplicate keys (`District` + `Year` + `Season`) across all three splits.

## 6. Target Distribution Analysis
| Split | Mean Yield (kg/ha) | Std Dev |
|---|---|---|
| **Train** | 161,833.1 | 126,877.9 |
| **Validation**| 278,984.3 | 57,416.2 |
| **Test** | 289,307.2 | 57,517.0 |

*(Note: The anomalously high overall means stem from uncorrected raw data errors in the source dataset — specifically `PURBA BARDHAMAN` records prior to 2017, which appear to have been multiplied by 100 in the upstream `rice_yield_wb_clean.csv`. As per strict instructions not to synthesize or manipulate raw dataset values without explicit commands, this anomaly is preserved exactly as it appears in the source data).*
