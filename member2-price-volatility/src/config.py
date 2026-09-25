"""
AgroVision — Centralized Configuration

Single source of truth for feature lists, file paths,
column names, and constants.

Multi-Commodity Support:
    - Call set_commodity("onion") or use CLI --commodity onion
    - All paths update automatically with commodity prefix
    - Models and data are saved per-commodity
"""

import os
from pathlib import Path


# ============================================================
# COMMODITY CONFIGURATION
#
# Default commodity. Override with set_commodity() or CLI args.
# ============================================================

_COMMODITY = os.environ.get("AGROVISION_COMMODITY", "tea")


def get_commodity():
    """Return the currently active commodity name."""
    return _COMMODITY


def set_commodity(name):
    """
    Switch the active commodity. All paths update automatically.

    Parameters
    ----------
    name : str
        Commodity name (e.g., "tea", "onion", "potato").
        Must match the raw CSV filename: data/raw/{name}.csv
    """
    global _COMMODITY
    _COMMODITY = name.strip().lower()


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def _commodity():
    """Shortcut for current commodity name."""
    return _COMMODITY


def raw_data_path():
    return RAW_DATA_DIR / f"{_commodity()}.csv"


def cleaned_data_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_cleaned.csv"


def timeseries_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_timeseries.csv"


def lag_features_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_lag_features.csv"


def rolling_features_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_rolling_features.csv"


def date_features_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_date_features.csv"


def arrival_features_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_arrival_features.csv"


def features_final_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_features_final.csv"


def train_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_train.csv"


def test_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_test.csv"


def volatility_train_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_volatility_train.csv"


def volatility_dataset_train_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_volatility_dataset_train.csv"


def volatility_dataset_test_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_volatility_dataset_test.csv"


def error_analysis_path():
    return PROCESSED_DATA_DIR / f"{_commodity()}_price_error_analysis.csv"


# --- Artifact paths (commodity-specific models) ---

def price_model_path():
    return ARTIFACTS_DIR / f"{_commodity()}_price_model.pkl"


def volatility_model_path():
    return ARTIFACTS_DIR / f"{_commodity()}_volatility_model.pkl"


def feature_config_path():
    return ARTIFACTS_DIR / f"{_commodity()}_feature_config.json"


def volatility_config_path():
    return ARTIFACTS_DIR / f"{_commodity()}_volatility_config.json"


# ============================================================
# BACKWARD-COMPATIBLE STATIC PATHS
#
# These are properties that return the path for the current
# commodity. Existing code that reads TRAIN_PATH etc. will
# still work, but now they resolve dynamically.
# ============================================================

class _DynamicPath:
    """Descriptor that calls a path function each time it's accessed."""
    def __init__(self, func):
        self.func = func
    def __fspath__(self):
        return str(self.func())
    def __str__(self):
        return str(self.func())
    def __repr__(self):
        return str(self.func())
    @property
    def parent(self):
        return self.func().parent
    def exists(self):
        return self.func().exists()
    def __truediv__(self, other):
        return self.func() / other
    def __eq__(self, other):
        return str(self.func()) == str(other)


# For simplicity and backward compat, expose as module-level constants
# that are re-evaluated. Scripts that import at module level will get
# the path for whatever commodity was set at import time.
# For dynamic use, call the functions above.

RAW_DATA_PATH = _DynamicPath(raw_data_path)
CLEANED_DATA_PATH = _DynamicPath(cleaned_data_path)
TIMESERIES_PATH = _DynamicPath(timeseries_path)
LAG_FEATURES_PATH = _DynamicPath(lag_features_path)
ROLLING_FEATURES_PATH = _DynamicPath(rolling_features_path)
DATE_FEATURES_PATH = _DynamicPath(date_features_path)
ARRIVAL_FEATURES_PATH = _DynamicPath(arrival_features_path)
FEATURES_FINAL_PATH = _DynamicPath(features_final_path)
TRAIN_PATH = _DynamicPath(train_path)
TEST_PATH = _DynamicPath(test_path)
VOLATILITY_TRAIN_PATH = _DynamicPath(volatility_train_path)
VOLATILITY_DATASET_TRAIN_PATH = _DynamicPath(volatility_dataset_train_path)
VOLATILITY_DATASET_TEST_PATH = _DynamicPath(volatility_dataset_test_path)
ERROR_ANALYSIS_PATH = _DynamicPath(error_analysis_path)
PRICE_MODEL_PATH = _DynamicPath(price_model_path)
VOLATILITY_MODEL_PATH = _DynamicPath(volatility_model_path)
FEATURE_CONFIG_PATH = _DynamicPath(feature_config_path)
VOLATILITY_CONFIG_PATH = _DynamicPath(volatility_config_path)


# ============================================================
# COLUMN NAMES
# ============================================================

DATE_COLUMN = "Reported Date"
PRICE_COLUMN = "Modal Price (Rs./Quintal)"
MIN_PRICE_COLUMN = "Min Price (Rs./Quintal)"
MAX_PRICE_COLUMN = "Max Price (Rs./Quintal)"
ARRIVAL_COLUMN = "Arrivals (Tonnes)"

PRICE_TARGET = "future_modal_price"
PRICE_CHANGE_TARGET = "future_price_pct_change"
PRICE_LOG_RETURN_TARGET = "future_log_return"
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
# FORECAST HORIZON
#
# Maximum number of days ahead we consider a valid target.
# Rows where the next observation is further away are dropped
# to keep the prediction horizon consistent.
# ============================================================

FORECAST_HORIZON_MAX_DAYS = 7


# IMPORTANT:
# Current modal price IS included because it is known at prediction time.
# It is not future information.
#
# The model target is future_modal_price / future_price_pct_change /
# future_log_return depending on the experiment.
#
# Min Price and Max Price current-row values remain excluded.
# arrival_change is also excluded because it uses the current-row arrival.

PRICE_FEATURES = [
    PRICE_COLUMN, 
    # Historical price lags (safe — from past observations)
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    # Rolling statistics (safe — computed from shifted past values)
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    # Date features (safe — known at prediction time)
    "year",
    "day",
    "month",
    "day_of_week",
    "week_of_year",
    # Arrival features (lags only — safe)
    "arrival_lag_1",
    "arrival_lag_7",
    "arrival_rolling_mean_7",
    "arrival_rolling_mean_14",
    # Percentage changes (safe — computed from previous row)
    "price_pct_change",
    "arrival_pct_change",
]


# ============================================================
# VOLATILITY MODEL FEATURES
#
# IMPORTANT: price_pct_change is intentionally EXCLUDED.
# Volatility is derived from price_pct_change, so including
# it would cause direct target leakage.
#
# Same leakage fixes as PRICE_FEATURES apply here.
# ============================================================

VOLATILITY_FEATURES = [
    # Historical price lags
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_30",
    # Rolling statistics
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_30",
    "rolling_std_7",
    "rolling_std_14",
    # Date features
    "year",
    "day",
    "month",
    "day_of_week",
    "week_of_year",
    # Arrival features (lags only)
    "arrival_lag_1",
    "arrival_lag_7",
    "arrival_rolling_mean_7",
    "arrival_rolling_mean_14",
    # Arrival percentage change
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
