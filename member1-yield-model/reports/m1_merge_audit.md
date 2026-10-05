# Member 1 Merge Audit & Fix Report

## 1. Problem Found
The original join between the Rice Yield, NASA POWER Weather, and MODIS NDVI datasets on `District` + `Year` + `Season` failed for a large number of rows.
- **Previous Weather missing:** 546 rows
- **Previous NDVI missing:** 645 rows

This massive data loss occurred despite all three datasets containing roughly the same geographical and temporal coverage.

## 2. Root Cause
The root cause of the merge failure was inconsistent district naming conventions across the three datasets:
1. **Capitalization:** The Yield dataset used UPPERCASE names (e.g., `ALIPURDUAR`), while Weather and NDVI used Title Case (e.g., `Alipurduar`).
2. **Spelling Variations:**
   - Yield: `24 PARAGANAS NORTH` vs Weather/NDVI: `North 24 Parganas`
   - Yield: `24 PARAGANAS SOUTH` vs Weather/NDVI: `South 24 Parganas`
   - Yield: `DINAJPUR DAKSHIN` vs Weather/NDVI: `Dakshin Dinajpur`
   - Yield: `DINAJPUR UTTAR` vs Weather/NDVI: `Uttar Dinajpur`
   - Yield: `MEDINIPUR WEST` vs Weather/NDVI: `Paschim Medinipur`
   - Yield: `MEDINIPUR EAST` vs Weather/NDVI: `Purba Medinipur`
   - NDVI: `Alipur Duar` vs Weather: `Alipurduar`
3. **Historical Boundaries:** The Yield dataset contained 54 rows for undivided `BARDHAMAN` (1997-2014). However, the NASA POWER and MODIS NDVI datasets were generated using modern district shapefiles, meaning they only contained data for the post-2017 split districts: `Paschim Bardhaman` and `Purba Bardhaman`. 

## 3. Exact Fix
A Python script (`scripts/fix_m1_merge.py`) was created to apply a strict, explicit mapping dictionary to standardise the Yield district names to match the Weather/NDVI conventions.
- NDVI's `Alipur Duar` was corrected to `Alipurduar`.
- `BARDHAMAN` was deliberately kept as `Bardhaman` and NOT mapped to `Purba Bardhaman` or `Paschim Bardhaman` to strictly prevent synthesizing data or introducing duplicate keys, as undivided Bardhaman physically represents a different geographic area than the modern split districts.

## 4. Before/After Row Counts
- **Original Yield Rows:** 1333
- **Final Merged Rows:** 1333
- **Row Multiplication:** None. The 1:1 merge constraint was successfully validated.

## 5. Missing-Value Results
By standardising the names, missing data dropped significantly:
- **Missing Weather Rows:** 54 (Down from 546)
- **Missing NDVI Rows (Year >= 2000):** 45 (Down from ~474 unexpected missing)
- **Legitimate NDVI Missing (1997-1999):** 171 rows

*Note: All 54 unmatched Weather rows and 45 unmatched NDVI rows (>= 2000) belong exclusively to the undivided `Bardhaman` district.*

## 6. Leakage/Duplication Validation
- **Row Duplication:** Validated 0 duplicate keys across `District` + `Year` + `Season` before and after the merge.
- **Temporal Leakage:** Verified that no future weather or NDVI data was joined to historical yield data. The join strictly enforced matching `Year` and `Season`.
- **Target Leakage:** The target variable (`Rice_Yield_kg_ha`) remains strictly isolated. No target-derived features were synthesized.

## 7. Remaining Limitations
- **Undivided Bardhaman (1997-2014):** 54 rows for `Bardhaman` lack environmental features (Weather/NDVI) because the source environmental datasets only contain records for the post-2017 split districts. Since interpolation or back-casting environmental data violates strict data integrity rules, these values remain legitimately `NaN`. Machine learning models down the pipeline will need to natively handle these NaNs (e.g., via XGBoost's `missing` parameter) or impute them during the feature engineering phase.
- **NDVI Data (1997-1999):** 171 rows lack NDVI data because MODIS satellite data collection began in 2000. These remain legitimately `NaN`.
