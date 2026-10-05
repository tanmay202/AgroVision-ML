# M1 In-Season Feature Audit

## Cutoff Definitions
- **Autumn:** May 1 to **July 31**
- **Winter:** Jun 1 to **Sep 30**
- **Summer:** Nov 1 (prev year) to **Feb 28**

## Weather Features Constructed
| Feature | Definition |
|---|---|
| `InSeason_Precip_Cum` | Cumulative rainfall from season start to cutoff |
| `InSeason_Precip_Mean` | Mean daily rainfall from season start to cutoff |
| `InSeason_Precip_Max` | Maximum single-day rainfall from season start to cutoff |
| `InSeason_T2M_Mean` | Mean 2m temperature from season start to cutoff |
| `InSeason_T2M_Min` | Absolute minimum temperature from season start to cutoff |
| `InSeason_T2M_Max` | Absolute maximum temperature from season start to cutoff |
| `InSeason_T2M_Range` | Temperature range (max - min) from season start to cutoff |
| `InSeason_RH2M_Mean` | Mean relative humidity from season start to cutoff |
| `InSeason_Solar_Mean` | Mean solar radiation from season start to cutoff |
| `InSeason_Wind_Mean` | Mean wind speed from season start to cutoff |
| `InSeason_Weather_Days` | Number of daily observations from season start to cutoff |

## NDVI
Full-season NDVI aggregates cannot be used (would include post-cutoff observations).
Only `NDVI_lag1` (previous year) is included.

## Leakage Verification
For every row in the dataset, the maximum weather observation date was verified
programmatically to be ≤ the forecast cutoff date. Zero violations detected.

## Missingness
In-season weather features may be missing for districts where the NASA POWER
coverage starts after 2000 or where historical district boundaries changed (e.g., undivided Bardhaman).
Missing values are handled via median imputation (fitted on training data only) for Ridge/RF,
or via native NaN handling for XGBoost.
