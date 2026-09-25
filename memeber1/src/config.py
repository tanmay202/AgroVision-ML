"""
AgroVision - Configuration
Central configuration for paths, constants, and feature definitions.
Adapted for Tea Mandi Price/Arrival data.
"""

import os

# ------------------------------------------
# Project root (assumes this file is at src/config.py)
# ------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------
# Data paths
# ------------------------------------------
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "data", "features")

# Raw data file - your downloaded CSV
RAW_DATA_FILE = os.path.join(RAW_DATA_DIR, "tea_cleaned.csv")

# Processed outputs
CLEANED_DATASET_FILE = os.path.join(PROCESSED_DATA_DIR, "cleaned_dataset.csv")
FEATURE_MATRIX_FILE = os.path.join(FEATURES_DIR, "feature_matrix.csv")

# ------------------------------------------
# Model paths
# ------------------------------------------
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
YIELD_MODEL_FILE = os.path.join(MODELS_DIR, "yield_model.pkl")

# ------------------------------------------
# Output paths
# ------------------------------------------
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

# ------------------------------------------
# Column names in your CSV
# ------------------------------------------
COLUMNS = {
    "state": "State Name",
    "district": "District Name",
    "market": "Market Name",
    "variety": "Variety",
    "group": "Group",
    "arrivals": "Arrivals (Tonnes)",
    "min_price": "Min Price (Rs./Quintal)",
    "max_price": "Max Price (Rs./Quintal)",
    "modal_price": "Modal Price (Rs./Quintal)",
    "date": "Reported Date",
}

# ------------------------------------------
# Target variable for prediction
# ------------------------------------------
# We predict "Arrivals (Tonnes)" as our yield/arrival forecast
TARGET_COLUMN = "Arrivals (Tonnes)"

# ------------------------------------------
# Model training settings
# ------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# XGBoost hyperparameter search space (fast version)
XGBOOST_PARAM_GRID_FAST = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.05, 0.1],
}

# Full search space (use for final tuning)
XGBOOST_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}
