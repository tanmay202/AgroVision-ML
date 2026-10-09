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


def _artifacts(commodity, artifacts_dir=None):
    """Load exactly the selected commodity's artifacts; never fall back."""
    commodity = str(commodity).strip().lower()
    if not commodity:
        raise ValueError("A commodity is required for prediction.")
    cache_key = (commodity, str(artifacts_dir) if artifacts_dir else None)
    if cache_key not in _cache:
        set_commodity(commodity)
        if artifacts_dir:
            base_dir = Path(artifacts_dir)
            paths = (
                base_dir / f"{commodity}_feature_config.json",
                base_dir / f"{commodity}_price_model.pkl",
                base_dir / f"{commodity}_volatility_model.pkl"
            )
        else:
            paths = (feature_config_path(), price_model_path(), volatility_model_path())
        missing = [str(path) for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing artifacts for commodity "
                                f"'{commodity}': {', '.join(missing)}")
        with open(paths[0], encoding="utf-8") as handle:
            configuration = json.load(handle)
        if configuration.get("commodity") != commodity:
            raise ValueError(f"Artifact config commodity does not match '{commodity}'.")

        # Load price model (handles dict wrappers from frozen production models)
        p_obj = joblib.load(paths[1])
        if isinstance(p_obj, dict) and "model" in p_obj:
            price_model = p_obj["model"]
        else:
            price_model = p_obj

        # If config expects 20 features but loaded model has 48 features, check for legacy model
        exp_features = configuration.get("price_model", {}).get("features", [])
        if commodity == "rice" and len(exp_features) == 20 and getattr(price_model, "n_features_in_", None) == 48:
            legacy_path = Path(paths[1]).parent / "legacy" / "rice_price_model_20feat.pkl"
            if legacy_path.exists():
                p_legacy = joblib.load(legacy_path)
                price_model = p_legacy.get("model", p_legacy) if isinstance(p_legacy, dict) else p_legacy

        # Load volatility model (handles dict wrappers)
        v_obj = joblib.load(paths[2])
        if isinstance(v_obj, dict) and "model" in v_obj:
            volatility_model = v_obj["model"]
        else:
            volatility_model = v_obj

        _cache[cache_key] = (configuration, price_model, volatility_model)
    return _cache[cache_key]


def _frame(data):
    if isinstance(data, pd.Series):
        return data.to_frame().T
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Prediction data is empty.")
        return data
    raise TypeError("Prediction data must be a pandas DataFrame or Series.")


def predict(data, commodity="rice", artifacts_dir=None, **kwargs):
    """Return an actual future modal price and a volatility class.

    The price model predicts a change internally.  This API reconstructs the
    actual Rs./Quintal price by adding it to the supplied current modal price.
    """
    frame = _frame(data)
    commodity = str(commodity).strip().lower()

    # Route Rice through production RiceInferencePipeline when history or 48-feature format is present
    if commodity == "rice":
        has_history = len(frame) >= 30 and any(c in frame.columns for c in ["Modal Price (Rs./Quintal)", "modal_price", "price"])
        has_48_feats = "price_change_1" in frame.columns and "trend_slope_7" in frame.columns
        if has_history or has_48_feats:
            try:
                from inference import RiceInferencePipeline
                pipeline = RiceInferencePipeline(artifacts_dir=artifacts_dir)
                res = pipeline.predict(frame, **kwargs)
                return {
                    "predicted_price": res["predicted_modal_price"],
                    "volatility": res["predicted_volatility_class"],
                    "commodity": "rice",
                    "predicted_volatility_range": res.get("predicted_volatility_range", 0.0),
                }
            except Exception:
                pass

    config, price_model, volatility_model = _artifacts(commodity, artifacts_dir=artifacts_dir)
    price_config = config["price_model"]
    price_features = price_config["features"]
    volatility_features = config["volatility_model"]["features"]
    reconstruction_column = price_config.get("reconstruction_column", "Modal Price (Rs./Quintal)")
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
