"""
AgroVision — Prediction API

Loads trained models and generates predictions.
Supports multi-commodity: specify commodity to load the right model.

Usage (for Member 3 / FastAPI):
    from src.models.predict import predict
    result = predict(data_df, commodity="tea")
    # result = {"predicted_price": 250.50, "volatility": "LOW"}
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from config import (
    PRICE_FEATURES,
    VOLATILITY_FEATURES,
    VOLATILITY_REVERSE_LABEL_MAP,
    set_commodity,
    get_commodity,
    price_model_path,
    volatility_model_path,
)


# ============================================================
# Model cache — loaded once per commodity, reused on every call.
# ============================================================

_model_cache = {}


def _get_price_model(commodity=None):
    """Load the saved XGBoost price model (cached per commodity)."""
    commodity = commodity or get_commodity()
    cache_key = f"{commodity}_price"
    if cache_key not in _model_cache:
        set_commodity(commodity)
        path = price_model_path()
        if not path.exists():
            raise FileNotFoundError(f"Price model not found: {path}")
        _model_cache[cache_key] = joblib.load(path)
    return _model_cache[cache_key]


def _get_volatility_model(commodity=None):
    """Load the saved XGBoost volatility model (cached per commodity)."""
    commodity = commodity or get_commodity()
    cache_key = f"{commodity}_volatility"
    if cache_key not in _model_cache:
        set_commodity(commodity)
        path = volatility_model_path()
        if not path.exists():
            raise FileNotFoundError(f"Volatility model not found: {path}")
        _model_cache[cache_key] = joblib.load(path)
    return _model_cache[cache_key]


def predict(data, commodity=None):
    """
    Generate price and volatility predictions.

    Parameters
    ----------
    data : pandas.DataFrame
        One or more rows containing all required features.
    commodity : str, optional
        Commodity name (e.g., "tea"). Defaults to current commodity.

    Returns
    -------
    dict
        {
            "predicted_price": float,
            "volatility": str  ("LOW", "MEDIUM", or "HIGH"),
            "commodity": str,
        }
    """
    commodity = commodity or get_commodity()

    # Validate features
    available_price = [f for f in PRICE_FEATURES if f in data.columns]
    missing_price = set(PRICE_FEATURES) - set(available_price)
    if missing_price:
        raise ValueError(
            f"Missing price features: {', '.join(missing_price)}"
        )

    available_vol = [f for f in VOLATILITY_FEATURES if f in data.columns]
    missing_vol = set(VOLATILITY_FEATURES) - set(available_vol)
    if missing_vol:
        raise ValueError(
            f"Missing volatility features: {', '.join(missing_vol)}"
        )

    # Price prediction
    price_model = _get_price_model(commodity)
    X_price = data[PRICE_FEATURES].copy()
    predicted_price = float(price_model.predict(X_price)[0])

    # Volatility prediction
    vol_model = _get_volatility_model(commodity)
    X_vol = data[VOLATILITY_FEATURES].copy()
    predicted_class = vol_model.predict(X_vol)[0]
    volatility = VOLATILITY_REVERSE_LABEL_MAP[int(predicted_class)]

    return {
        "predicted_price": round(predicted_price, 2),
        "volatility": volatility,
        "commodity": commodity,
    }


def main():
    print("=" * 60)
    print("PREDICTION PIPELINE")
    print("=" * 60)
    print(f"Commodity: {get_commodity()}")
    print(f"\nPrice model features ({len(PRICE_FEATURES)}):")
    for f in PRICE_FEATURES:
        print(f"  - {f}")
    print(f"\nVolatility model features ({len(VOLATILITY_FEATURES)}):")
    for f in VOLATILITY_FEATURES:
        print(f"  - {f}")


if __name__ == "__main__":
    main()