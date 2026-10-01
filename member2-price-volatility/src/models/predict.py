"""Public commodity-specific prediction API."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from config import (VOLATILITY_REVERSE_LABEL_MAP, feature_config_path,
                    price_model_path, set_commodity, volatility_model_path)

_cache = {}


def _artifacts(commodity):
    """Load exactly the selected commodity's artifacts; never fall back."""
    commodity = str(commodity).strip().lower()
    if not commodity:
        raise ValueError("A commodity is required for prediction.")
    if commodity not in _cache:
        set_commodity(commodity)
        paths = (feature_config_path(), price_model_path(), volatility_model_path())
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing artifacts for commodity "
                                f"'{commodity}': {', '.join(missing)}")
        with open(paths[0], encoding="utf-8") as handle:
            configuration = json.load(handle)
        if configuration.get("commodity") != commodity:
            raise ValueError(f"Artifact config commodity does not match '{commodity}'.")
        _cache[commodity] = (configuration, joblib.load(paths[1]), joblib.load(paths[2]))
    return _cache[commodity]


def _frame(data):
    if isinstance(data, pd.Series):
        return data.to_frame().T
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Prediction data is empty.")
        return data
    raise TypeError("Prediction data must be a pandas DataFrame or Series.")


def predict(data, commodity="rice"):
    """Return an actual future modal price and a volatility class.

    The price model predicts a change internally.  This API reconstructs the
    actual Rs./Quintal price by adding it to the supplied current modal price.
    """
    frame = _frame(data)
    commodity = str(commodity).strip().lower()
    config, price_model, volatility_model = _artifacts(commodity)
    price_config = config["price_model"]
    price_features = price_config["features"]
    volatility_features = config["volatility_model"]["features"]
    reconstruction_column = price_config["reconstruction_column"]
    required = price_features + volatility_features + [reconstruction_column]
    missing = list(dict.fromkeys(c for c in required if c not in frame.columns))
    if missing:
        raise ValueError("Missing required prediction columns: " + ", ".join(missing))

    predicted_change = float(price_model.predict(frame[price_features])[0])
    current_price = pd.to_numeric(frame.iloc[0][reconstruction_column], errors="coerce")
    if pd.isna(current_price) or current_price <= 0:
        raise ValueError(f"'{reconstruction_column}' must be a positive numeric current price.")
    predicted_price = float(current_price + predicted_change)
    if predicted_price <= 0:
        raise ValueError("Reconstructed predicted price is not positive.")

    predicted_code = int(volatility_model.predict(frame[volatility_features])[0])
    if predicted_code not in VOLATILITY_REVERSE_LABEL_MAP:
        raise ValueError(f"Unknown volatility class code: {predicted_code}")
    return {"predicted_price": round(predicted_price, 2),
            "volatility": VOLATILITY_REVERSE_LABEL_MAP[predicted_code],
            "commodity": commodity}
