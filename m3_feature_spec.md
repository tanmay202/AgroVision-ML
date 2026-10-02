# M3 Integration Specification: Rice Feature Engineering

This document provides the **exact** transformation logic required to reproduce the frozen 48-dimensional feature vector for the Rice Price XGBoost model. It is extracted directly from the verified production implementation (`src/inference.py`).

---

## A. Complete 48-Feature Transformation Table

All grouping operations (denoted as `grouped_price` or `grouped_arr` or `grouped`) are executed over:
`['State Name', 'District Name', 'Market Name', 'Variety', 'Group']`.

| Feature | Source Column | Exact Transformation | Shift Used? | Window | Grouping | Output Meaning |
|---|---|---|---|---|---|---|
| **Base Features** |
| `lag_1` | `Modal Price` | `grouped_price.shift(1)` | Yes (1) | N/A | Market+Variety | Previous day's price |
| `lag_7` | `Modal Price` | `grouped_price.shift(7)` | Yes (7) | N/A | Market+Variety | Price 7 valid days ago |
| `lag_14` | `Modal Price` | `grouped_price.shift(14)` | Yes (14) | N/A | Market+Variety | Price 14 valid days ago |
| `lag_30` | `Modal Price` | `grouped_price.shift(30)` | Yes (30) | N/A | Market+Variety | Price 30 valid days ago |
| `rolling_mean_7` | `Modal Price` | `grouped_price.shift(1).rolling(7).mean()` | Yes (1) | 7 | Market+Variety | 7-day average of PAST prices |
| `rolling_mean_14` | `Modal Price` | `grouped_price.shift(1).rolling(14).mean()` | Yes (1) | 14 | Market+Variety | 14-day average of PAST prices |
| `rolling_mean_30` | `Modal Price` | `grouped_price.shift(1).rolling(30).mean()` | Yes (1) | 30 | Market+Variety | 30-day average of PAST prices |
| `rolling_std_7` | `Modal Price` | `grouped_price.shift(1).rolling(7).std().fillna(0)` | Yes (1) | 7 | Market+Variety | 7-day std deviation of PAST prices |
| `rolling_std_14`| `Modal Price` | `grouped_price.shift(1).rolling(14).std().fillna(0)`| Yes (1) | 14 | Market+Variety | 14-day std deviation of PAST prices |
| `year` | `Reported Date` | `dt.year` | No | N/A | None | Integer year |
| `month` | `Reported Date` | `dt.month` | No | N/A | None | Integer month (1-12) |
| `day` | `Reported Date` | `dt.day` | No | N/A | None | Integer day of month (1-31) |
| `day_of_week` | `Reported Date`| `dt.dayofweek` | No | N/A | None | Integer day of week (0-6) |
| `week_of_year` | `Reported Date`| `dt.isocalendar().week.astype(int)` | No | N/A | None | Integer ISO week |
| `arrival_lag_1` | `Arrivals` | `grouped_arr.shift(1)` | Yes (1) | N/A | Market+Variety | Previous day's arrivals |
| `arrival_lag_7` | `Arrivals` | `grouped_arr.shift(7)` | Yes (7) | N/A | Market+Variety | Arrivals 7 valid days ago |
| `arrival_rolling_mean_7` | `Arrivals` | `grouped_arr.shift(1).rolling(7).mean()` | Yes (1) | 7 | Market+Variety | 7-day average of PAST arrivals |
| `arrival_rolling_mean_14`| `Arrivals` | `grouped_arr.shift(1).rolling(14).mean()`| Yes (1) | 14 | Market+Variety | 14-day average of PAST arrivals |
| `arrival_pct_change` | `Arrivals` | `grouped_arr.pct_change(fill_method=None) * 100.0` | No | 1 (diff) | Market+Variety | Daily % change in arrivals |
| `price_pct_change` | `Modal Price` | `grouped_price.pct_change(fill_method=None) * 100.0` | No | 1 (diff) | Market+Variety | Daily % change in price |
| **Momentum Features** |
| `price_change_1` | `Modal Price` | `grouped_price.diff(1)` | No | 1 | Market+Variety | Absolute Rs change (Today - Yesterday) |
| `price_change_7` | `Modal Price` | `grouped_price.diff(7)` | No | 7 | Market+Variety | Absolute Rs change (Today - 7 days ago) |
| `price_change_14`| `Modal Price` | `grouped_price.diff(14)` | No | 14 | Market+Variety | Absolute Rs change (Today - 14 days ago) |
| `accel_1_7` | `price_change_*` | `price_change_1 - price_change_7` | No | N/A | None | Velocity shift (1-day vs 7-day) |
| `accel_1_14`| `price_change_*` | `price_change_1 - price_change_14`| No | N/A | None | Velocity shift (1-day vs 14-day) |
| **Trend Features** |
| `trend_slope_7` | `Modal Price` | `sum(w7[i] * grouped_price.shift(6-i))` (see Section B) | No | 7 | Market+Variety | Linear regression slope over 7 days |
| `trend_slope_14`| `Modal Price` | `sum(w14[i] * grouped_price.shift(13-i))` | No | 14 | Market+Variety | Linear regression slope over 14 days |
| `trend_slope_30`| `Modal Price` | `sum(w30[i] * grouped_price.shift(29-i))` | No | 30 | Market+Variety | Linear regression slope over 30 days |
| `ewma_7` | `Modal Price` | `grouped_price.ewm(span=7, adjust=False).mean()` | No | 7 (span)| Market+Variety | Exp. Weighted Moving Avg |
| `ewma_14` | `Modal Price` | `grouped_price.ewm(span=14, adjust=False).mean()` | No | 14 (span)| Market+Variety | Exp. Weighted Moving Avg |
| `ewma_30` | `Modal Price` | `grouped_price.ewm(span=30, adjust=False).mean()` | No | 30 (span)| Market+Variety | Exp. Weighted Moving Avg |
| `rolling_median_7` | `Modal Price` | `grouped_price.rolling(7).median()` | No | 7 | Market+Variety | 7-day median including today |
| `rolling_median_14`| `Modal Price` | `grouped_price.rolling(14).median()` | No | 14 | Market+Variety | 14-day median including today |
| `rolling_median_30`| `Modal Price` | `grouped_price.rolling(30).median()` | No | 30 | Market+Variety | 30-day median including today |
| `gap_mean_7` | `Modal Price` | `price - rolling_mean_7` | N/A | N/A | None | Gap between current price and PAST 7d avg |
| `gap_mean_14`| `Modal Price` | `price - rolling_mean_14` | N/A | N/A | None | Gap between current price and PAST 14d avg |
| `zscore_7` | `Modal Price` | `(price - rolling_mean_7) / (rolling_std_7 + 1e-5)` | N/A | N/A | None | Current price Z-score vs PAST 7d distribution |
| `zscore_14`| `Modal Price` | `(price - rolling_mean_14) / (rolling_std_14 + 1e-5)`| N/A | N/A | None | Current price Z-score vs PAST 14d distribution |
| **History Features** |
| `move_count_7` | `price_change_1` | `(price_change_1 != 0).rolling(7).sum()` | No | 7 | Market+Variety | Count of non-zero moves in last 7 days |
| `move_count_14`| `price_change_1` | `(price_change_1 != 0).rolling(14).sum()` | No | 14 | Market+Variety | Count of non-zero moves in last 14 days |
| `move_count_30`| `price_change_1` | `(price_change_1 != 0).rolling(30).sum()` | No | 30 | Market+Variety | Count of non-zero moves in last 30 days |
| `pos_count_7` | `price_change_1` | `(price_change_1 > 0).rolling(7).sum()` | No | 7 | Market+Variety | Count of positive moves in last 7 days |
| `neg_count_7` | `price_change_1` | `(price_change_1 < 0).rolling(7).sum()` | No | 7 | Market+Variety | Count of negative moves in last 7 days |
| `streak_zero` | `price_change_1` | Consecutive days where `price_change_1 == 0` | No | N/A | Market+Variety | Days since last price change |
| `streak_pos` | `price_change_1` | Consecutive days where `price_change_1 > 0` | No | N/A | Market+Variety | Days in current uptrend |
| `streak_neg` | `price_change_1` | Consecutive days where `price_change_1 < 0` | No | N/A | Market+Variety | Days in current downtrend |
| `last_non_zero_move`| `price_change_1` | `replace(0, NaN).ffill().fillna(0)` | No | N/A | Market+Variety | The value of the last non-zero price change |
| `last_move_abs` | `last_non_zero_mo` | `abs(last_non_zero_move)` | No | N/A | None | Absolute value of the last non-zero price change |

---

## B. Complex Formulas & Logic Breakdown

**1. Trend Slope Extraction (`trend_slope_7`, `14`, `30`)**
Calculates the linear regression slope over the window ending on the current day.
```python
def get_slope_weights(window):
    x = np.arange(window) - (window - 1) / 2.0
    return x / np.sum(x**2)

# For window = 7:
w7 = get_slope_weights(7)
df["trend_slope_7"] = sum(w7[i] * grouped_price.shift(6 - i) for i in range(7))
```

**2. Streak Calculations (`streak_zero`, `streak_pos`, `streak_neg`)**
Resets the count to 0 when the condition breaks, increments otherwise.
```python
def compute_streak(df, condition_series):
    group_keys = [df[c] for c in GROUP_COLUMNS]
    block_id = (~condition_series).groupby(group_keys).cumsum()
    streak = condition_series.groupby(group_keys + [block_id]).cumsum()
    return streak

is_move = (df["price_change_1"] != 0).astype(int)
df["streak_zero"] = compute_streak(df, is_move == 0)
```

---

## C. M3 Implementation Specification

M3 must process data using these strict rules to identically recreate the feature vector:

1. **Required Input Columns**:
   `['State Name', 'District Name', 'Market Name', 'Variety', 'Group', 'Arrivals (Tonnes)', 'Modal Price (Rs./Quintal)', 'Reported Date']`
2. **Required Sorting**:
   Filter dataset to the target Market+Variety, cast `Reported Date` to datetime, and sort chronologically: `df.sort_values("Reported Date")`. Keep only the `last` observed value if exact date duplicates exist.
3. **Required Grouping**:
   All Pandas window/shift functions *must* be grouped by the full key: `['State Name', 'District Name', 'Market Name', 'Variety', 'Group']`.
4. **Chronological Restrictions**:
   **Critical**: For a prediction on date `T`, filter the dataframe to `< T` (or `<= T` if `T` represents "today's known data"). *Never* allow rows `> T` in the feature generation dataframe.
5. **Missing-Value Rules**:
   - Empty numeric inputs should be `ffill().bfill()` prior to feature generation.
   - Percentage changes (`pct_change`) explicitly use `fill_method=None`.
   - `rolling_std` outputs `NaN` when variance is 0 (e.g. only 1 unique value in window); explicitly `.fillna(0)`.
   - Z-score uses `+ 1e-5` to prevent `ZeroDivisionError`.
6. **Data Types**:
   The final generated row must be coerced to numeric via `pd.to_numeric(errors="coerce")` before passing to XGBoost.

---

## D. Final 48-Feature Ordered Vector

The model expects exactly 48 columns in this precise frozen sequence:

```python
[
  "lag_1", "lag_7", "lag_14", "lag_30", 
  "rolling_mean_7", "rolling_mean_14", "rolling_mean_30", 
  "rolling_std_7", "rolling_std_14", 
  "year", "month", "day", "day_of_week", "week_of_year", 
  "arrival_lag_1", "arrival_lag_7", 
  "arrival_rolling_mean_7", "arrival_rolling_mean_14", 
  "arrival_pct_change", "price_pct_change", 
  "price_change_1", "price_change_7", "price_change_14", 
  "accel_1_7", "accel_1_14", 
  "trend_slope_7", "trend_slope_14", "trend_slope_30", 
  "ewma_7", "ewma_14", "ewma_30", 
  "rolling_median_7", "rolling_median_14", "rolling_median_30", 
  "gap_mean_7", "gap_mean_14", 
  "zscore_7", "zscore_14", 
  "move_count_7", "move_count_14", "move_count_30", 
  "pos_count_7", "neg_count_7", 
  "streak_zero", "streak_pos", "streak_neg", 
  "last_non_zero_move", "last_move_abs"
]
```

---

## E. Leakage Safety Notes
Every feature strictly uses only information available at prediction time.
- **Future Prices**: Never accessed.
- **Future Targets**: Never accessed.
- **Current Target/Future Target Exclusion**: The model target is `future_price_change`. The feature generation safely uses the known "current" `Modal Price` (the anchor) for `pct_change`, `ewma`, `trend_slope`, and `zscore` calculations, which does *not* leak the future target.
- **Past Rolling**: `rolling_mean_X` and `rolling_std_X` enforce a `.shift(1)` to calculate historical baseline distributions prior to the current day, allowing features like `gap_mean` and `zscore` to safely compare "today's price vs history".

---

## F. Validation Report

| Component | Status | Note |
|---|---|---|
| **Feature formulas** | **PASS** | Extracted exactly from `src/inference.py` inline definitions. |
| **Feature order** | **PASS** | Matches `artifacts/rice_price_final_config.json`. |
| **Training/inference consistency** | **PASS** | Verified via `test_training_inference_parity.py` (2/2 Passed). |
| **Leakage safety** | **PASS** | Verified via `test_leakage.py` (10/10 Passed). |

**Files Inspected (Untouched):**
- `src/inference.py`
- `tests/test_training_inference_parity.py`
- `artifacts/rice_price_final_config.json`
