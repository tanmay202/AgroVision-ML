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
            sub_paths = (
                base_dir / commodity / f"{commodity}_feature_config.json",
                base_dir / commodity / f"{commodity}_price_model.pkl",
                base_dir / commodity / f"{commodity}_volatility_model.pkl"
            )
            flat_paths = (
                base_dir / f"{commodity}_feature_config.json",
                base_dir / f"{commodity}_price_model.pkl",
                base_dir / f"{commodity}_volatility_model.pkl"
            )
            if sub_paths[0].exists() and sub_paths[1].exists():
                paths = sub_paths
            else:
                paths = flat_paths
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
            legacy_candidates = [
                Path(paths[1]).parent / "legacy" / "rice_price_model_20feat.pkl",
                get_artifacts_dir() / "rice" / "legacy" / "rice_price_model_20feat.pkl",
                get_artifacts_dir() / "legacy" / "rice_price_model_20feat.pkl",
            ]
            for legacy_path in legacy_candidates:
                if legacy_path.exists():
                    p_legacy = joblib.load(legacy_path)
                    price_model = p_legacy.get("model", p_legacy) if isinstance(p_legacy, dict) else p_legacy
                    break

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


def _extract_20_features_from_history(df):
    """Compute 20 price and volatility features from historical time series observations."""
    col_map = {}
    for c in df.columns:
        clow = c.lower().replace(" ", "_")
        if "modal" in clow or clow in ("price", "modal_price"):
            col_map[c] = "Modal Price (Rs./Quintal)"
        elif "arrival" in clow:
            col_map[c] = "Arrivals (Tonnes)"
        elif "date" in clow:
            col_map[c] = "Reported Date"
    work = df.rename(columns=col_map).copy()
    if "Arrivals (Tonnes)" not in work.columns:
        work["Arrivals (Tonnes)"] = 0.0
    if "Reported Date" not in work.columns:
        work["Reported Date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(work)).strftime("%Y-%m-%d")

    work["Reported Date"] = pd.to_datetime(work["Reported Date"], errors="coerce")
    work = work.sort_values("Reported Date").reset_index(drop=True)
    work["Modal Price (Rs./Quintal)"] = pd.to_numeric(work["Modal Price (Rs./Quintal)"], errors="coerce")
    work["Arrivals (Tonnes)"] = pd.to_numeric(work["Arrivals (Tonnes)"], errors="coerce").fillna(0.0)

    p = work["Modal Price (Rs./Quintal)"]
    arr = work["Arrivals (Tonnes)"]
    dt = work["Reported Date"]
    last_idx = len(work) - 1

    price_slice_7 = p.iloc[max(0, last_idx - 7):last_idx] if last_idx > 0 else p.iloc[0:1]
    price_slice_14 = p.iloc[max(0, last_idx - 14):last_idx] if last_idx > 0 else p.iloc[0:1]
    price_slice_30 = p.iloc[max(0, last_idx - 30):last_idx] if last_idx > 0 else p.iloc[0:1]

    arr_slice_7 = arr.iloc[max(0, last_idx - 7):last_idx] if last_idx > 0 else arr.iloc[0:1]
    arr_slice_14 = arr.iloc[max(0, last_idx - 14):last_idx] if last_idx > 0 else arr.iloc[0:1]

    row_feat = {
        "Modal Price (Rs./Quintal)": float(p.iloc[last_idx]),
        "lag_1": float(p.iloc[last_idx - 1]) if len(p) >= 2 else float(p.iloc[last_idx]),
        "lag_7": float(p.iloc[last_idx - 7]) if len(p) >= 8 else float(p.iloc[0]),
        "lag_14": float(p.iloc[last_idx - 14]) if len(p) >= 15 else float(p.iloc[0]),
        "lag_30": float(p.iloc[last_idx - 30]) if len(p) >= 31 else float(p.iloc[0]),
        "rolling_mean_7": float(price_slice_7.mean()),
        "rolling_mean_14": float(price_slice_14.mean()),
        "rolling_mean_30": float(price_slice_30.mean()),
        "rolling_std_7": float(price_slice_7.std(ddof=0)) if len(price_slice_7) > 1 else 0.0,
        "rolling_std_14": float(price_slice_14.std(ddof=0)) if len(price_slice_14) > 1 else 0.0,
        "year": int(dt.iloc[last_idx].year),
        "month": int(dt.iloc[last_idx].month),
        "day": int(dt.iloc[last_idx].day),
        "day_of_week": int(dt.iloc[last_idx].dayofweek),
        "week_of_year": int(dt.iloc[last_idx].isocalendar().week),
        "arrival_lag_1": float(arr.iloc[last_idx - 1]) if len(arr) >= 2 else float(arr.iloc[last_idx]),
        "arrival_lag_7": float(arr.iloc[last_idx - 7]) if len(arr) >= 8 else float(arr.iloc[0]),
        "arrival_rolling_mean_7": float(arr_slice_7.mean()),
        "arrival_rolling_mean_14": float(arr_slice_14.mean()),
        "price_pct_change": float((p.iloc[last_idx] - p.iloc[last_idx - 1]) / p.iloc[last_idx - 1] * 100) if (len(p) >= 2 and p.iloc[last_idx - 1] > 0) else 0.0,
        "arrival_pct_change": float((arr.iloc[last_idx] - arr.iloc[last_idx - 1]) / arr.iloc[last_idx - 1] * 100) if (len(arr) >= 2 and arr.iloc[last_idx - 1] > 0) else 0.0,
        "price_spread": 0.0,
        "price_position_in_range": 0.5,
    }
    return pd.DataFrame([row_feat])


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

    if missing and len(frame) >= 1 and any(c.lower().replace(" ", "_") in ("modal_price", "price", "modal_price_(rs./quintal)") for c in frame.columns):
        frame = _extract_20_features_from_history(frame)
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
