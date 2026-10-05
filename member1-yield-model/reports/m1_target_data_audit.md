# Member 1 Target Data Audit & Correction

## 1. The Anomaly
During the time-split validation, the target variable (`Rice_Yield_kg_ha`) exhibited physically impossible mean values:
- **Train Mean:** ≈ 161,833 kg/ha
- **Validation Mean:** ≈ 278,984 kg/ha
- **Test Mean:** ≈ 289,307 kg/ha

In reality, standard rice yields in West Bengal range between 1,500 and 4,000 kg/ha.

## 2. Root Cause Discovery
An audit of the raw dataset (`data/raw/rice.csv`) revealed a systemic data entry corruption affecting the `Production` column:
- For **975 rows** (representing the vast majority of the dataset, particularly all rows from 2015-2019, and the majority of districts pre-2015), the `Production` value was exactly **100 times higher** than it should be.
- For example, in 1997, `BARDHAMAN` recorded a production of 91,590 tonnes. A duplicate entry under the name `PURBA BARDHAMAN` for the exact same season recorded 9,159,000 (exactly 100x).
- Because `Rice_Yield = Production / Area`, this factor-of-100 error passed directly into the yield calculation, resulting in values exceeding 300,000 kg/ha.

## 3. The Correction Rule
Instead of blindly deleting rows, a strictly deterministic, evidence-based correction was applied:
- **Condition:** If `Rice_Yield_kg_ha > 10000` (which is >10 tonnes/ha, biologically impossible for a district-level average and perfectly isolating the x100 anomaly gap).
- **Action:** Divide `Production` by 100.
- **Recalculation:** Re-derive `Rice_Yield_t_ha` and `Rice_Yield_kg_ha` from the corrected `Production` and the original `Area`.

*The raw dataset (`rice.csv`) was preserved unchanged. The correction was applied to `rice_yield_wb_clean.csv` via `scripts/fix_rice_yield_units.py`.*

## 4. Affected Rows
- **Total Rows Corrected:** 975 rows.
- **Districts Affected:** 22 out of 23 districts (including 100% of rows for 15 districts, and all rows statewide from 2015-2019).

## 5. Before & After Statistics (All Data)

**Before Correction:**
- **Mean:** 187,913 kg/ha
- **Median:** 223,576 kg/ha
- **Max:** 447,939 kg/ha

**After Correction:**
- **Mean:** 2,551 kg/ha
- **Median:** 2,596 kg/ha
- **Min:** 285 kg/ha
- **Max:** 4,479 kg/ha
*(These values perfectly align with standard agronomic realities in India).*

## 6. Pipeline Rebuild & Target Statistics
All downstream files were rebuilt using the corrected clean data. The new, correct target distributions across the time splits are:

| Split | Years | Mean Yield (kg/ha) | Std Dev |
|---|---|---|---|
| **Train** | 2000 – 2014 | 2,497.9 | 616.3 |
| **Validation** | 2015 – 2016 | 2,789.8 | 574.2 |
| **Test** | 2017 – 2019 | 2,893.1 | 575.2 |

*(The slight upward trend from Train to Test reflects realistic historical agricultural improvements).*

## 7. Leakage & Duplicate Validation
- **Leakage:** The time-split script (`create_time_split.py`) was re-run. It confirmed `0` mismatches in the `Yield_lag1` self-join check, proving that the corrected target values correctly propagated into the historical lag features without peeking into the future.
- **Duplicates:** The merge script (`fix_m1_merge.py`) confirmed `0` duplicate keys (`District` + `Year` + `Season`) were introduced.
