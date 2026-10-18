"""
Configuration and constants for AgroVision-ML Member 2 (Price Prediction and Volatility ML).
"""

from pathlib import Path

# Standard Column Names
DATE_COLUMN = "Reported Date"
PRICE_COLUMN = "Modal Price (Rs./Quintal)"
ARRIVAL_COLUMN = "Arrivals (Tonnes)"
GROUP_COLUMNS = ["Market Name", "Variety"]

# Alternative column aliases for flexible ingestion
COLUMN_ALIASES = {
    "date": DATE_COLUMN,
    "reported_date": DATE_COLUMN,
    "reported date": DATE_COLUMN,
    "price": PRICE_COLUMN,
    "modal_price": PRICE_COLUMN,
    "modal price": PRICE_COLUMN,
    "modal_price (rs./quintal)": PRICE_COLUMN,
    "arrivals": ARRIVAL_COLUMN,
    "arrival": ARRIVAL_COLUMN,
    "arrivals (tonnes)": ARRIVAL_COLUMN,
    "market": GROUP_COLUMNS[0],
    "market_name": GROUP_COLUMNS[0],
    "market name": GROUP_COLUMNS[0],
    "mandi": GROUP_COLUMNS[0],
    "variety": GROUP_COLUMNS[1],
}

# Horizons and Target Conventions
SUPPORTED_HORIZONS = [1, 14, 30]
DEFAULT_MAX_TOLERANCE_DAYS = 3

TARGET_CONVENTIONS = {
    1: "Next valid observation within 1–7 calendar days",
    14: "First observation in [t + 14, t + 17] calendar days within Market + Variety",
    30: "First observation in [t + 30, t + 33] calendar days within Market + Variety",
}

RELEASE_STATUSES = {
    1: "production",
    14: "opt-in",
    30: "experimental",
}

MODEL_NAMES = {
    1: "rice-price-xgb",
    14: "rice-price-xgb-14d",
    30: "rice-price-xgb-30d",
}

MODEL_LIMITATIONS = {
    1: "Production baseline model for short-term 1-7 day rice price forecasting in West Bengal mandis.",
    14: (
        "Opt-in release. Evaluated on 18,921 holdout observations (MAE: Rs.59.87 vs Rs.60.16 persistence, "
        "RMSE: Rs.141.01 vs Rs.146.29). Performance is validated across major rice varieties (Fine, Common, Super Fine); "
        "accuracy may vary in low-liquidity mandis."
    ),
    30: (
        "Experimental release. Evaluated on 17,860 holdout observations. Demonstrates variance reduction on moving markets "
        "(movement MAE: Rs.161.45 vs Rs.164.91 persistence, RMSE: Rs.257.07 vs Rs.263.30), but exhibits parity on flat markets "
        "(overall MAE: Rs.99.95 vs Rs.99.50). Interpret with caution."
    ),
}
