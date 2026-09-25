**# AgroVision-ML — Complete Project Guide**

**## Table of Contents**

1\. [Project Overview]\(#project-overview)

2\. [Architecture & Structure]\(#architecture--structure)

3\. [Bugs Found & Fixed]\(#bugs-found--fixed)

4\. [Known Limitations (Not Bugs)]\(#known-limitations-not-bugs)

5\. [How to Add a Different data.csv]\(#how-to-add-a-different-datacsv)

6\. [Member 1 — Detailed Guide]\(#member-1--yield-ml-detailed-guide)

7\. [Member 2 — Detailed Guide]\(#member-2--price-ml-detailed-guide)

8\. [How Both Members Connect (Integration)]\(#how-both-members-connect-integration)

9\. [Learning Roadmap: From Scratch to End]\(#learning-roadmap-from-scratch-to-end)

10\. [Quick Reference Commands]\(#quick-reference-commands)

**---**

**## Project Overview**

AgroVision-ML is a **\*\*machine learning pipeline for Indian agricultural commodity (tea) market analysis\*\***. It has two independent ML components built by two team members:

\| Component | Member | Purpose | Model Type |

\|-----------|--------|---------|------------|

\| **\*\*Yield ML\*\*** | Member 1 (\`memeber1/\`) | Predict tea **\*\*arrivals\*\*** (tonnes) | XGBoost Regression |

\| **\*\*Price ML\*\*** | Member 2 (\`src/\`) | Predict tea **\*\*prices\*\*** & **\*\*volatility\*\*** | XGBoost Regression + Classification |

Both use the same **\*\*Agmarknet mandi data\*\*** (tea prices/arrivals from Indian markets).

**---**

**## Architecture & Structure**

\`\`\`

AgroVision-ML/

│

├── memeber1/                     ← MEMBER 1: Yield/Arrival forecasting

│   ├── main.py                   ← Entry point (run this)

│   ├── inspect_data.py           ← Quick data inspection utility

│   ├── requirements.txt

│   ├── data/

│   │   ├── raw/                  ← PUT YOUR CSV HERE (tea_cleaned.csv)

│   │   ├── processed/            ← Auto-generated cleaned data

│   │   └── features/             ← Auto-generated feature matrix

│   ├── models/                   ← Saved .pkl model files

│   ├── outputs/                  ← Evaluation plots & reports

│   └── src/

│       ├── config.py             ← ⚙️ ALL SETTINGS HERE

│       ├── data_loader.py        ← Step 1: Load CSV

│       ├── data_cleaner.py       ← Step 2: Clean data

│       ├── feature_engineer.py   ← Step 3: Create ML features

│       ├── model_trainer.py      ← Step 4: Train models

│       ├── model_evaluator.py    ← Step 5: Evaluate & report

│       └── predict.py            ← Inference API for Member 3

│

├── src/                          ← MEMBER 2: Price forecasting + Volatility

│   ├── config.py                 ← ⚙️ ALL SETTINGS HERE

│   ├── data/

│   │   ├── check_raw_data.py     ← Inspect raw CSV

│   │   ├── clean_data.py         ← Step 1: Clean raw data

│   │   ├── inspect_data.py       ← EDA utility

│   │   └── prepare_timeseries.py ← Step 2: Create time-series target

│   ├── features/

│   │   ├── create_lag_features.py       ← Step 3a: Price lags

│   │   ├── create_rolling_features.py   ← Step 3b: Rolling stats

│   │   ├── create_date_features.py      ← Step 3c: Date features

│   │   ├── create_arrival_features.py   ← Step 3d: Arrival features

│   │   ├── create_pct_change_features.py← Step 3e: % change features

│   │   ├── create_volatility_target.py  ← Step 4: Volatility labels

│   │   ├── create_volatility_dataset.py ← Step 5: Volatility dataset

│   │   └── create_volatility_test.py    ← Step 6: Volatility test set

│   ├── models/

│   │   ├── baseline.py                  ← Naive baseline model

│   │   ├── random_forest_price.py       ← RF price model

│   │   ├── xgboost_price.py            ← XGBoost price model

│   │   ├── xgboost_volatility.py        ← XGBoost volatility model

│   │   ├── volatility_classifier.py     ← Volatility classifier

│   │   ├── save_final_models.py         ← Export final .pkl models

│   │   ├── predict.py                   ← Inference API for Member 3

│   │   └── test_prediction.py           ← Test predictions

│   ├── evaluation/

│   │   ├── time_split.py                ← Train/test split

│   │   ├── error_analysis.py            ← Error analysis

│   │   ├── check_price_anomalies.py     ← Anomaly detection

│   │   └── final_validation.py          ← Leakage & validation checks

│   └── utils/

│

├── data/

│   ├── raw/                      ← Member 2's raw data (tea.csv)

│   └── processed/                ← Member 2's processed outputs

│

├── artifacts/                    ← Saved models & configs

│   ├── price_model.pkl

│   ├── volatility_model.pkl

│   ├── feature_config.json

│   └── volatility_config.json

│

└── requirements.txt

\`\`\`

**---**

**## Bugs Found & Fixed**

**### 🔴 CRITICAL Bugs (Fixed)**

**#### 1. Model Prediction Crash — \`memeber1/src/predict.py\`**

**\*\*Problem:\*\*** \`model_trainer.py\` saves the model as a **\*\*dictionary\*\*** (\`{"model": estimator, "feature_names": [...], "use_log_target": True}\`), but \`predict.py\` called \`model.predict()\` directly on the dict — causing \`AttributeError\`.

**\*\*Fix Applied:\*\*** Updated \`predict.py\` to extract \`model_info["model"]\` before calling \`.predict()\`. Added backward compatibility for raw model objects.

**---**

**#### 2. Missing Log Inverse Transform — \`memeber1/src/predict.py\`**

**\*\*Problem:\*\*** The model is trained on \`np.log1p(y)\` (log-transformed arrivals), but predictions were returned **\*\*without\*\*** calling \`np.expm1()\` to convert back. All predictions were returning **\*\*log values\*\*** (\~2.5) instead of actual tonnes (\~12.0).

**\*\*Fix Applied:\*\*** Added \`np.expm1(prediction)\` when \`use_log_target=True\`.

**---**

**#### 3. Evaluation on Log Scale — \`memeber1/src/model_evaluator.py\`**

**\*\*Problem:\*\*** RMSE, MAE, R2 were computed on log-scale values, giving **\*\*falsely optimistic\*\*** error metrics (e.g., RMSE=0.3 instead of the real RMSE=15.0 in tonnes). Plots also showed log-scale values.

**\*\*Fix Applied:\*\*** Added inverse transformation before computing all metrics and generating all plots. Also added "Gradient Boosting" to the comparison list (it was missing).

**---**

**#### 4. Infinity Data Loss — \`src/features/create_pct_change_features.py\`**

**\*\*Problem:\*\*** When previous price/arrival was 0, division created \`inf\`. These were replaced with \`pd.NA\`, causing downstream \`dropna()\` to silently delete valid data rows.

**\*\*Fix Applied:\*\*** Changed infinity replacement from \`pd.NA\` to \`0.0\` (zero percent change is the correct interpretation).

**---**

**#### 5. Hardcoded Data Path — \`memeber1/inspect_data.py\`**

**\*\*Problem:\*\*** Hardcoded \`'data/raw/tea_cleaned.csv'\` instead of using config.

**\*\*Fix Applied:\*\*** Now imports \`RAW_DATA_FILE\` from config.

**---**

**#### 6. Date Format Lock-in — \`src/data/clean_data.py\`**

**\*\*Problem:\*\*** Hardcoded \`format="%d %b %Y"\` means any dataset using \`YYYY-MM-DD\` or other formats would have **\*\*100% of dates coerced to NaT\*\***, destroying the entire dataset.

**\*\*Fix Applied:\*\*** Added fallback — if >50% of dates fail explicit parsing, retries with automatic format detection.

**---**

**#### 7. README Mismatch — \`memeber1/README.md\`**

**\*\*Problem:\*\*** Instructed users to place \`crop_production.csv\`, \`weather_data.csv\`, \`ndvi_data.csv\`, \`soil_data.csv\` — but the code only uses tea mandi data. Completely misleading.

**\*\*Fix Applied:\*\*** Rewrote README to accurately describe the tea mandi pipeline.

**---**

**### 🟡 Known Limitations (Not Bugs)**

These are design limitations you should be aware of but don't crash the code:

\| # | Issue | Where | Impact |

\|---|-------|-------|--------|

\| 1 | **\*\*Forecast horizon mixing\*\*** | \`src/data/prepare_timeseries.py\` | \`shift(-1)\` predicts "next record" regardless if it's 1 day or 6 months away. The model mixes different horizons. |

\| 2 | **\*\*Aggressive NaN dropping\*\*** | \`src/models/xgboost_price.py\` | Drops rows with NaN from 30-day lags/rolling, losing the first 30 records per group. XGBoost handles NaN natively. |

\| 3 | **\*\*Hardcoded anomaly threshold\*\*** | \`src/evaluation/check_price_anomalies.py\` | \`threshold=100000\` is arbitrary. Different commodities need different thresholds. |

\| 4 | **\*\*Inconsistent groupby\*\*** | Multiple \`src/\` files | Some scripts use \`config.GROUP_COLUMNS\`, others hardcode \`["Market Name", "Variety"]\`. |

\| 5 | **\*\*Volatility labels bypass config\*\*** | \`src/features/create_volatility_target.py\` | Hardcodes \`"LOW"\`, \`"MEDIUM"\`, \`"HIGH"\` instead of using \`config.VOLATILITY_CLASSES\`. |

\| 6 | **\*\*Single-row std edge case\*\*** | \`memeber1/src/data_cleaner.py\` | Groups with 1 row produce \`NaN\` std — caught but edge-case risky. |

**---**

**## How to Add a Different data.csv**

\> [!IMPORTANT]

\> Both members expect **\*\*Agmarknet-format\*\*** tea mandi data. If your new CSV has the same columns, just drop it in. If it has different columns, follow the steps below.

**### Step-by-Step for Any New CSV**

**#### Phase 1: Check Your CSV Structure**

\`\`\`python

import pandas as pd

df = pd.read_csv("your_new_data.csv")

print(df.columns.tolist())

print(df.dtypes)

print(df.head())

\`\`\`

**#### Phase 2: Map Your Columns**

Your CSV needs these types of data (column names can differ):

\| Required Data | Example Column Name |

\|---------------|-------------------|

\| Date | \`Reported Date\`, \`Date\`, \`date\` |

\| Price (primary) | \`Modal Price (Rs./Quintal)\`, \`Price\` |

\| Price (min) | \`Min Price (Rs./Quintal)\`, \`Low\` |

\| Price (max) | \`Max Price (Rs./Quintal)\`, \`High\` |

\| Volume/Arrivals | \`Arrivals (Tonnes)\`, \`Quantity\` |

\| Market identifier | \`Market Name\`, \`Mandi\` |

\| Variety/Category | \`Variety\`, \`Commodity\` |

**#### Phase 3: Update Configs**

**\*\*For Member 1\*\*** — Edit \`memeber1/src/config.py\`:

\`\`\`python

\# 1. Change the raw data filename

RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "your_new_data.csv")

\# 2. Map YOUR column names

COLUMNS = {

    "state": "Your State Column",      # or remove if not present

    "district": "Your District Column",

    "market": "Your Market Column",

    "variety": "Your Variety Column",

    "group": "Your Group Column",

    "arrivals": "Your Arrivals Column",

    "min_price": "Your Min Price Column",

    "max_price": "Your Max Price Column",

    "modal_price": "Your Modal Price Column",

    "date": "Your Date Column",

}

\# 3. Set what you want to predict

TARGET_COLUMN = "Your Arrivals Column"

\`\`\`

**\*\*For Member 2\*\*** — Edit \`src/config.py\`:

\`\`\`python

\# 1. Change commodity name (drives file paths)

COMMODITY = "your_commodity"

\# 2. Change column names

DATE_COLUMN = "Your Date Column"

PRICE_COLUMN = "Your Modal Price Column"

MIN_PRICE_COLUMN = "Your Min Price Column"

MAX_PRICE_COLUMN = "Your Max Price Column"

ARRIVAL_COLUMN = "Your Arrivals Column"

\# 3. Update GROUP_COLUMNS

GROUP_COLUMNS = ["Your Market Column", "Your Variety Column"]

\`\`\`

**#### Phase 4: Place Your Data**

\`\`\`

\# For Member 1:

memeber1/data/raw/your_new_data.csv

\# For Member 2:

data/raw/your_commodity.csv

\`\`\`

**#### Phase 5: Run Pipelines**

\`\`\`bash

\# Member 1

cd memeber1

python main.py --no-tune    # Fast first run

\# Member 2 (run scripts in order)

cd ..   # back to project root

python src/data/clean_data.py

python src/data/prepare_timeseries.py

python src/features/create_lag_features.py

python src/features/create_rolling_features.py

python src/features/create_date_features.py

python src/features/create_arrival_features.py

python src/features/create_pct_change_features.py

python src/evaluation/time_split.py

python src/models/xgboost_price.py

python src/models/save_final_models.py

\`\`\`

\> [!WARNING]

\> If your date column uses a format other than \`dd Mon YYYY\` (e.g., \`2024-01-15\`), the fallback auto-detection in \`clean_data.py\` will handle it. For Member 1, \`data_cleaner.py\` already uses auto-detection. But always verify dates parsed correctly before training.

**---**

**## Member 1 — Yield ML Detailed Guide**

**### What It Does**

Predicts **\*\*tea arrivals (tonnes)\*\*** at Indian mandis using historical price and arrival data.

**### Pipeline Flow**

\`\`\`

CSV → Load → Clean → Feature Engineering → Train 4 Models → Evaluate Best → Save .pkl

\`\`\`

**### Models Trained (in order)**

1\. **\*\*Linear Regression\*\*** — Baseline

2\. **\*\*Random Forest\*\*** — Ensemble

3\. **\*\*Gradient Boosting\*\*** — Sklearn's GBR

4\. **\*\*XGBoost\*\*** — With GridSearchCV tuning

Best model (by R² score) is automatically selected and saved.

**### Features Created**

\| Category | Features |

\|----------|----------|

\| **\*\*Lag\*\*** | \`Arrival_Lag_1/3/7\`, \`Price_Lag_1/3/7\` |

\| **\*\*Rolling\*\*** | \`Arrival_RollMean_7/14/30\`, \`Price_RollMean_7/14/30\`, \`Price_Volatility_14\`, \`Arrival_Volatility_14\` |

\| **\*\*Price Derived\*\*** | \`Price_Spread\`, \`Price_Ratio\`, \`Log_Arrivals\`, \`Log_Modal_Price\`, \`Price_Change\`, \`Price_Pct_Change\`, \`Arrival_Change\` |

\| **\*\*Temporal\*\*** | \`Year\`, \`Month\`, \`Day\`, \`DayOfWeek\`, \`WeekOfYear\`, \`Quarter\`, \`Month_Sin/Cos\`, \`DayOfWeek_Sin/Cos\`, \`Year_Trend\`, \`Is_Weekend\` |

\| **\*\*Encoded\*\*** | \`State_Name_Encoded\`, \`District_Name_Encoded\`, \`Market_Name_Encoded\`, \`Variety_Encoded\` |

**### Key Design Decisions**

\- **\*\*Log-transformed target\*\***: \`np.log1p(Arrivals)\` — because arrival distribution is heavily right-skewed

\- **\*\*Time-based split\*\***: Last 20% chronologically (not random) — realistic for time-series

\- **\*\*Inverse transform\*\***: Predictions are converted back via \`np.expm1()\` before output

**### Outputs**

\| File | Description |

\|------|-------------|

\| \`models/yield_model.pkl\` | Saved best model (dict with model + metadata) |

\| \`outputs/actual_vs_predicted.png\` | Scatter plot |

\| \`outputs/residual_analysis.png\` | Residual distribution |

\| \`outputs/feature_importance.png\` | Top 15 feature importances |

\| \`outputs/evaluation_report.csv\` | RMSE, MAE, R², MAPE |

\| \`outputs/model_comparison.csv\` | All 4 models compared |

**### Member 1 TODO for Integration**

\- [ ] Retrain with new dataset if CSV changes

\- [ ] Verify \`models/yield_model.pkl\` exists after training

\- [ ] Hand \`src/predict.py\` to Member 3 for FastAPI integration

\- [ ] Update \`COLUMNS\` dict in \`config.py\` if column names change

**---**

**## Member 2 — Price ML Detailed Guide**

**### What It Does**

1\. **\*\*Predicts future modal price\*\*** (₹/Quintal) — XGBoost Regression

2\. **\*\*Classifies price volatility\*\*** (LOW/MEDIUM/HIGH) — XGBoost Classification

**### Pipeline Flow (Run Scripts in Order!)**

\> [!IMPORTANT]

\> Unlike Member 1's single \`main.py\`, Member 2 requires running scripts **\*\*sequentially\*\***. Each script reads the previous script's output.

\`\`\`

Step 1:  python src/data/clean_data.py

Step 2:  python src/data/prepare_timeseries.py

Step 3a: python src/features/create_lag_features.py

Step 3b: python src/features/create_rolling_features.py

Step 3c: python src/features/create_date_features.py

Step 3d: python src/features/create_arrival_features.py

Step 3e: python src/features/create_pct_change_features.py

Step 4:  python src/evaluation/time_split.py

Step 5:  python src/models/baseline.py

Step 6:  python src/models/random_forest_price.py

Step 7:  python src/models/xgboost_price.py

Step 8:  python src/features/create_volatility_target.py

Step 9:  python src/features/create_volatility_dataset.py

Step 10: python src/features/create_volatility_test.py

Step 11: python src/models/volatility_classifier.py

Step 12: python src/models/save_final_models.py

Step 13: python src/evaluation/final_validation.py  (optional)

\`\`\`

**### Features Used (Price Model)**

\| Feature | Description |

\|---------|-------------|

\| \`lag_1/7/14/30\` | Historical price at N records ago |

\| \`rolling_mean_7/14/30\` | Rolling average price |

\| \`rolling_std_7/14\` | Rolling price standard deviation |

\| \`year/month/day/day_of_week/week_of_year\` | Date components |

\| \`arrival_lag_1/7\` | Historical arrivals |

\| \`arrival_change\` | Arrival delta |

\| \`arrival_rolling_mean_7/14\` | Arrival rolling averages |

\| \`price_pct_change\` | Price % change (excluded from volatility model!) |

\| \`arrival_pct_change\` | Arrival % change |

**### Outputs**

\| File | Description |

\|------|-------------|

\| \`artifacts/price_model.pkl\` | XGBoost price regressor |

\| \`artifacts/volatility_model.pkl\` | XGBoost volatility classifier |

\| \`artifacts/feature_config.json\` | Feature list & metadata |

\| \`artifacts/volatility_config.json\` | Volatility thresholds |

**### Member 2 TODO for Integration**

\- [ ] Run all 12+ scripts in order after data change

\- [ ] Verify \`artifacts/\` contains all 4 files

\- [ ] Hand \`src/models/predict.py\` to Member 3 for FastAPI integration

\- [ ] Update \`src/config.py\` column names if CSV changes

**---**

**## How Both Members Connect (Integration)**

**### For Member 3 (Engineering / FastAPI)**

Member 3 needs to import from **\*\*two separate predict modules\*\***:

\`\`\`python

\# Member 1's prediction (arrivals/yield)

from memeber1.src.predict import load_model, predict_yield

\# Member 2's prediction (price + volatility)

from src.models.predict import predict

\# --- Member 1 Usage ---

model_info = load_model()  # Returns dict

prediction = predict_yield(model_info, {

    "Modal Price (Rs./Quintal)": 250,

    "Min Price (Rs./Quintal)": 200,

    "Max Price (Rs./Quintal)": 300,

    "Price_Lag_1": 240,

    "Price_RollMean_7": 245,

    # ... all features from model_info["feature_names"]

})

print(f"Predicted Arrivals: {prediction} tonnes")

\# --- Member 2 Usage ---

import pandas as pd

data = pd.DataFrame([{

    "Arrivals (Tonnes)": 50,

    "Modal Price (Rs./Quintal)": 250,

    "lag_1": 240,

    "lag_7": 235,

    # ... all PRICE_FEATURES + VOLATILITY_FEATURES

}])

result = predict(data)

print(f"Predicted Price: ₹{result['predicted_price']}")

print(f"Volatility: {result['volatility']}")

\`\`\`

**### Integration Architecture**

\`\`\`

┌─────────────────────────────────────────────────────┐

│                    FastAPI Service                    │

│                   (Member 3 builds)                  │

├─────────────────────┬───────────────────────────────┤

│                     │                               │

│  memeber1/src/      │   src/models/                 │

│  predict.py         │   predict.py                  │

│                     │                               │

│  → Arrival forecast │   → Price forecast            │

│    (tonnes)         │   → Volatility (L/M/H)        │

│                     │                               │

│  yield_model.pkl    │   price_model.pkl              │

│                     │   volatility_model.pkl         │

└─────────────────────┴───────────────────────────────┘

\`\`\`

**---**

**## Learning Roadmap: From Scratch to End**

If you're building this project from scratch, here's **\*\*exactly what to learn\*\*** in order:

**### Phase 1: Python Foundations (1-2 weeks)**

\- [ ] Python basics (variables, loops, functions, classes)

\- [ ] File I/O (reading/writing CSV, JSON)

\- [ ] Virtual environments (\`pip install\`, \`requirements.txt\`)

\- [ ] Command line / terminal basics

\- [ ] Git basics (\`git init\`, \`commit\`, \`push\`, \`pull\`, branching)

**\*\*Resources\*\***: Python.org tutorial, freeCodeCamp Python

**---**

**### Phase 2: Data Science Libraries (2-3 weeks)**

\- [ ] **\*\*NumPy\*\***: Arrays, mathematical operations, \`np.log1p()\`, \`np.expm1()\`

\- [ ] **\*\*Pandas\*\***: DataFrames, \`read_csv\`, \`groupby\`, \`shift()\`, \`rolling()\`, \`merge\`

\- [ ] **\*\*Matplotlib/Seaborn\*\***: Scatter plots, histograms, bar charts

\- [ ] Data cleaning: handling missing values, outliers, duplicates

\- [ ] Feature engineering concepts: lags, rolling windows, percentage changes

**\*\*Resources\*\***: Kaggle's Pandas course, "Python for Data Analysis" by Wes McKinney

**---**

**### Phase 3: Machine Learning (3-4 weeks)**

\- [ ] **\*\*Scikit-learn basics\*\***: \`fit()\`, \`predict()\`, \`score()\`, train/test split

\- [ ] **\*\*Regression models\*\***: Linear Regression, Random Forest, Gradient Boosting

\- [ ] **\*\*Classification models\*\***: Logistic Regression, Random Forest Classifier

\- [ ] **\*\*Evaluation metrics\*\***: RMSE, MAE, R², Accuracy, F1, Confusion Matrix

\- [ ] **\*\*Cross-validation\*\***: K-Fold, GridSearchCV for hyperparameter tuning

\- [ ] **\*\*XGBoost\*\***: Installation, \`XGBRegressor\`, \`XGBClassifier\`, parameter tuning

\- [ ] **\*\*Time-series concepts\*\***: Why not random split, chronological splitting, look-ahead bias

**\*\*Resources\*\***: Scikit-learn docs, Andrew Ng's ML course, Kaggle's ML course

**---**

**### Phase 4: ML Pipeline Design (1-2 weeks)**

\- [ ] Project structure: \`src/\`, \`data/\`, \`models/\`, \`config.py\`

\- [ ] Pipeline pattern: Load → Clean → Feature Engineering → Train → Evaluate

\- [ ] Configuration management: centralizing paths, column names, hyperparameters

\- [ ] Model serialization: \`joblib.dump()\` / \`joblib.load()\`

\- [ ] Target transformations: \`log1p\` for skewed data, remembering to \`expm1\`

\- [ ] Feature importance analysis

\- [ ] Error analysis and debugging ML models

**\*\*Resources\*\***: Scikit-learn Pipeline docs, "Hands-On ML" by Aurélien Géron

**---**

**### Phase 5: Domain Knowledge (1 week)**

\- [ ] Understanding **\*\*Agmarknet data\*\***: What mandi price data looks like

\- [ ] Agricultural market concepts: arrivals, modal/min/max prices, seasonality

\- [ ] **\*\*Volatility classification\*\***: Price stability as LOW/MEDIUM/HIGH

\- [ ] Time-series forecasting for agricultural commodities

\- [ ] Indian agricultural market dynamics

**\*\*Resources\*\***: Agmarknet website, ICAR publications

**---**

**### Phase 6: Engineering & Deployment (2-3 weeks)**

\- [ ] **\*\*FastAPI\*\***: Creating REST APIs, request/response models, endpoints

\- [ ] **\*\*Model serving\*\***: Loading \`.pkl\` files, making predictions via API

\- [ ] **\*\*Docker\*\***: Containerizing the ML service

\- [ ] **\*\*Frontend\*\***: Basic dashboard (React/Streamlit) for visualizations

\- [ ] **\*\*CI/CD\*\***: Automated testing, deployment pipelines

**\*\*Resources\*\***: FastAPI docs, Streamlit docs, Docker getting started

**---**

**### Key Concepts This Project Teaches**

\| Concept | Where Used | Why It Matters |

\|---------|-----------|----------------|

\| Log transformation | \`model_trainer.py\` | Handles skewed target distributions |

\| Inverse transform | \`predict.py\`, \`model_evaluator.py\` | Returns predictions in original scale |

\| Time-based split | \`model_trainer.py\`, \`time_split.py\` | Prevents future data leaking into training |

\| Lag features | \`feature_engineer.py\`, \`create_lag_features.py\` | Captures temporal dependencies |

\| Rolling statistics | \`feature_engineer.py\`, \`create_rolling_features.py\` | Captures trends and volatility |

\| Cyclical encoding | \`feature_engineer.py\` | Makes month/day-of-week features circular |

\| Label encoding | \`feature_engineer.py\` | Converts categorical strings to numbers |

\| GridSearchCV | \`model_trainer.py\` | Automated hyperparameter tuning |

\| Model serialization | \`model_trainer.py\`, \`save_final_models.py\` | Saving models for production use |

\| Feature importance | \`model_evaluator.py\` | Understanding what drives predictions |

**---**

**## Quick Reference Commands**

**### Setup**

\`\`\`bash

\# Install all dependencies

pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

\# Or use requirements files

pip install -r requirements.txt                  # Main project

pip install -r memeber1/requirements.txt         # Member 1

\`\`\`

**### Member 1 — Run Pipeline**

\`\`\`bash

cd memeber1

python main.py                # Full pipeline (with tuning, \~5-10 min)

python main.py --no-tune      # Fast mode (\~1-2 min)

python main.py --predict-only # Just load model

\`\`\`

**### Member 2 — Run Pipeline (in order!)**

\`\`\`bash

\# From project root (AgroVision-ML/)

python src/data/clean_data.py

python src/data/prepare_timeseries.py

python src/features/create_lag_features.py

python src/features/create_rolling_features.py

python src/features/create_date_features.py

python src/features/create_arrival_features.py

python src/features/create_pct_change_features.py

python src/evaluation/time_split.py

python src/models/xgboost_price.py

python src/features/create_volatility_target.py

python src/features/create_volatility_dataset.py

python src/features/create_volatility_test.py

python src/models/volatility_classifier.py

python src/models/save_final_models.py

\`\`\`

**### Verify Everything Works**

\`\`\`bash

\# Check Member 1 outputs exist

dir memeber1\models\yield_model.pkl

dir memeber1\outputs\\

\# Check Member 2 outputs exist

dir artifacts\price_model.pkl

dir artifacts\volatility_model.pkl

\# Test Member 2 predictions

python src/models/test_prediction.py

\`\`\`

**---**

\> [!TIP]

\> **\*\*When adding new data\*\***: Always run \`inspect_data.py\` (or \`check_raw_data.py\`) first to verify your CSV columns match what the config expects. Fix column mappings in \`config.py\` BEFORE running the pipeline.

\> [!CAUTION]

\> **\*\*Never commit \`.pkl\` model files or large CSV data to Git.\*\*** The \`.gitignore\` already handles this, but double-check before pushing. Model files can be 1-3 MB each.