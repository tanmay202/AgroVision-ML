"""
AgroVision — Centralized Configuration

Single source of truth for feature lists, file paths,
column names, and constants.

To add a new commodity, change COMMODITY below.
All paths will update automatically.
"""

from pathlib import Path


# ============================================================
# COMMODITY CONFIGURATION
# Change this to switch commodities (e.g., "onion", "potato")
# ============================================================

COMMODITY = "tea"


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / f"{COMMODITY}.csv"

CLEANED_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / f"{COMMODITY}_cleaned.csv"
)

TIMESERIES_PATH = (
    PROJECT_ROOT / "data" / "processed" / f"{COMMODITY}_timeseries.csv"
)

LAG_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / f"{COMMODITY}_lag_features.csv"
)

ROLLING_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / f"{COMMODITY}_rolling_features.csv"
)

DATE_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / f"{COMMODITY}_date_features.csv"
)

ARRIVAL_FEATURES_PATH = (
    PROJECT_ROOT / "data" / "processed" / f"{COMMODITY}_arrival_features.csv"
)

FEATURES_FINAL_PATH = (
    PROJECT_ROOT / "data" / "processed" / f"{COMMODITY}_features_final.csv"
)

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

VOLATILITY_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "volatility_train.csv"
)

VOLATILITY_DATASET_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "volatility_dataset_train.csv"
)

VOLATILITY_DATASET_TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "volatility_dataset_test.csv"
)

ERROR_ANALYSIS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "price_error_analysis.csv"
)


# ============================================================
# ARTIFACT PATHS
# ============================================================

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PRICE_MODEL_PATH = ARTIFACTS_DIR / "price_model.pkl"
VOLATILITY_MODEL_PATH = ARTIFACTS_DIR / "volatility_model.pkl"
FEATURE_CONFIG_PATH = ARTIFACTS_DIR / "feature_config.json"
VOLATILITY_CONFIG_PATH = ARTIFACTS_DIR / "volatility_config.json"


# ============================================================
# COLUMN NAMES
# ============================================================

DATE_COLUMN = "Reported Date"
PRICE_COLUMN = "Modal Price (Rs./Quintal)"
MIN_PRICE_COLUMN = "Min Price (Rs./Quintal)"
MAX_PRICE_COLUMN = "Max Price (Rs./Quintal)"
ARRIVAL_COLUMN = "Arrivals (Tonnes)"

PRICE_TARGET = "future_modal_price"
VOLATILITY_TARGET = "volatility"


# ============================================================
# GROUP COLUMNS (define one time series)
# ============================================================

GROUP_COLUMNS = ["Market Name", "Variety"]


# ============================================================
# NUMERIC COLUMNS (for data validation)
# ============================================================

NUMERIC_COLUMNS = [
    ARRIVAL_COLUMN,
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    PRICE_COLUMN,
]


# ============================================================
# PRICE COLUMNS (for anomaly checks)
# ============================================================

PRICE_COLUMNS = [
    MIN_PRICE_COLUMN,
    MAX_PRICE_COLUMN,
    PRICE_COLUMN,
]


# ============================================================
# PRICE MODEL FEATURES
# Used for price forecasting (XGBoost regressor).
# Includes price_pct_change and arrival_pct_change.
# ============================================================

PRICE_FEATURES = [
    "Arrivals (Tonnes)",
    "Min Price (Rs./Quintal)",
    "Max Price (Rs./Quintal)",
    "Modal Price (Rs./Quintal)",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    "year",
    "day",
    "month",
    "day_of_week",
    "week_of_year",
    "arrival_lag_1",
    "arrival_lag_7",
    "arrival_change",
    "arrival_rolling_mean_7",
    "arrival_rolling_mean_14",
    "price_pct_change",
    "arrival_pct_change",
]


# ============================================================
# VOLATILITY MODEL FEATURES
#
# IMPORTANT: price_pct_change is intentionally EXCLUDED.
# Volatility is derived from price_pct_change, so including
# it would cause direct target leakage.
# ============================================================

VOLATILITY_FEATURES = [
    "Arrivals (Tonnes)",
    "Min Price (Rs./Quintal)",
    "Max Price (Rs./Quintal)",
    "Modal Price (Rs./Quintal)",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    "year",
    "day",
    "month",
    "day_of_week",
    "week_of_year",
    "arrival_lag_1",
    "arrival_lag_7",
    "arrival_change",
    "arrival_rolling_mean_7",
    "arrival_rolling_mean_14",
    "arrival_pct_change",
]


# ============================================================
# VOLATILITY LABEL ENCODING
# ============================================================

VOLATILITY_LABEL_MAP = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
}

VOLATILITY_REVERSE_LABEL_MAP = {
    0: "LOW",
    1: "MEDIUM",
    2: "HIGH",
}

VOLATILITY_CLASSES = ["LOW", "MEDIUM", "HIGH"]
