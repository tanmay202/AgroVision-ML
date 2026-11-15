"""Public commodity-specific prediction API with Multi-Commodity V1/V2 Runtime."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd

from config import (
    VOLATILITY_REVERSE_LABEL_MAP,
    feature_config_path,
    price_model_path,
    set_commodity,
    volatility_model_path,
    get_artifacts_dir,
)

_cache = {}


def _artifacts(commodity, artifacts_dir=None, version="v2", horizon_days=1):
    """Load exactly the selected commodity's artifacts; never fall back."""
    commodity = str(commodity).strip().lower()
    if not commodity:
        raise ValueError("A commodity is required for prediction.")

    cache_key = (commodity, str(artifacts_dir) if artifacts_dir else None, version, horizon_days)
    if cache_key not in _cache:
        set_commodity(commodity)
        base_dir = Path(artifacts_dir) if artifacts_dir else get_artifacts_dir()

        # Check for V2 artifacts if requested and available
        is_v2 = (version == "v2") and (commodity in ["wheat", "potato"])
        v2_dir = base_dir / commodity / "v2"

        if is_v2 and v2_dir.exists():
            cfg_path = v2_dir / f"{commodity}_feature_config.json"
            if not cfg_path.exists():
                cfg_path = v2_dir / f"{commodity}_price_model_v2_config.json"
            if horizon_days == 14:
                p_path = v2_dir / f"{commodity}_price_14d_model.pkl"
            elif horizon_days == 30:
                p_path = v2_dir / f"{commodity}_price_30d_model.pkl"
            else:
                p_path = v2_dir / f"{commodity}_price_model_v2.pkl"
                if not p_path.exists():
                    p_path = v2_dir / f"{commodity}_price_model.pkl"
            v_path = v2_dir / f"{commodity}_volatility_model_v2.pkl"
            if not v_path.exists():
                v_path = v2_dir / f"{commodity}_volatility_model.pkl"
            res_path = v2_dir / f"{commodity}_residual_distributions.json"
        else:
            # V1 or fallback path
            cfg_path = base_dir / commodity / f"{commodity}_feature_config.json"
            if not cfg_path.exists():
                cfg_path = base_dir / f"{commodity}_feature_config.json"
            p_path = base_dir / commodity / f"{commodity}_price_model.pkl"
            if not p_path.exists():
                p_path = base_dir / f"{commodity}_price_model.pkl"
            v_path = base_dir / commodity / f"{commodity}_volatility_model.pkl"
            if not v_path.exists():
                v_path = base_dir / f"{commodity}_volatility_model.pkl"
            res_path = None

        if not cfg_path.exists() or not p_path.exists() or not v_path.exists():
            # Fall back to config helpers if paths don't exist
            cfg_path = feature_config_path()
            p_path = price_model_path()
            v_path = volatility_model_path()

        if not cfg_path.exists() or not p_path.exists() or not v_path.exists():
            missing = [str(p) for p in [cfg_path, p_path, v_path] if not p.exists()]
            raise FileNotFoundError(f"Missing artifacts for commodity '{commodity}': {', '.join(missing)}")

        with open(cfg_path, encoding="utf-8") as handle:
            configuration = json.load(handle)

        if configuration.get("commodity") != commodity:
            raise ValueError(f"Artifact config commodity does not match '{commodity}'.")

        # Load price model
        p_obj = joblib.load(p_path)
        deadband_eps = 0.0
        if isinstance(p_obj, dict) and "model" in p_obj:
            price_model = p_obj["model"]
            deadband_eps = float(p_obj.get("deadband_eps", 0.0))
        else:
            price_model = p_obj

        # Load volatility model
        v_obj = joblib.load(v_path)
        if isinstance(v_obj, dict) and "model" in v_obj:
            volatility_model = v_obj["model"]
        else:
            volatility_model = v_obj

        # Load residual distribution if present
        residuals_info = None
        if res_path and res_path.exists():
            with open(res_path, encoding="utf-8") as f:
                residuals_info = json.load(f)

        _cache[cache_key] = (configuration, price_model, volatility_model, deadband_eps, residuals_info)

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
    """Compute 20 price and volatility features from historical time series observations (V1)."""
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


def _extract_v2_features_from_history(df, feature_names):
    """Compute V2 candidate features from history for Wheat and Potato."""
    from features_v2 import generate_candidate_price_features
    col_map = {}
    for c in df.columns:
        clow = c.lower().replace(" ", "_")
        if "modal" in clow or clow in ("price", "modal_price"):
            col_map[c] = "Modal Price (Rs./Quintal)"
        elif "arrival" in clow:
            col_map[c] = "Arrivals (Tonnes)"
        elif "date" in clow:
            col_map[c] = "Reported Date"
        elif "market" in clow:
            col_map[c] = "Market Name"
        elif "variety" in clow:
            col_map[c] = "Variety"

    work = df.rename(columns=col_map).copy()
    p_col = "Modal Price (Rs./Quintal)"
    if p_col not in work.columns:
        raise ValueError(f"Missing required price column '{p_col}' in history.")

    work[p_col] = pd.to_numeric(work[p_col], errors="coerce")
    if "Arrivals (Tonnes)" not in work.columns:
        work["Arrivals (Tonnes)"] = 0.0
    else:
        work["Arrivals (Tonnes)"] = pd.to_numeric(work["Arrivals (Tonnes)"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "Reported Date" not in work.columns:
        work["Reported Date"] = pd.date_range(end=pd.Timestamp.now(), periods=len(work)).strftime("%Y-%m-%d")
    if "Market Name" not in work.columns:
        work["Market Name"] = "DefaultMarket"
    if "Variety" not in work.columns:
        work["Variety"] = "DefaultVariety"

    if "Min Price (Rs./Quintal)" not in work.columns:
        work["Min Price (Rs./Quintal)"] = work[p_col]
    else:
        work["Min Price (Rs./Quintal)"] = pd.to_numeric(work["Min Price (Rs./Quintal)"], errors="coerce").fillna(work[p_col])

    if "Max Price (Rs./Quintal)" not in work.columns:
        work["Max Price (Rs./Quintal)"] = work[p_col]
    else:
        work["Max Price (Rs./Quintal)"] = pd.to_numeric(work["Max Price (Rs./Quintal)"], errors="coerce").fillna(work[p_col])

    feats = generate_candidate_price_features(work)

    # Extra volatility-specific features
    feats["rolling_range_7"] = feats["rolling_max_7"] - feats["rolling_min_7"]
    feats["rolling_pct_range_7"] = (feats["rolling_range_7"] / (feats[p_col] + 1e-4)) * 100.0
    feats["rolling_pct_std_7"] = (feats["rolling_std_7"] / (feats[p_col] + 1e-4)) * 100.0
    feats["rolling_pct_std_14"] = (feats["rolling_std_14"] / (feats[p_col] + 1e-4)) * 100.0
    feats["rolling_pct_std_30"] = (feats["rolling_std_30"] / (feats[p_col] + 1e-4)) * 100.0
    feats["price_to_rolling_mean_30"] = feats[p_col] / (feats["rolling_mean_30"] + 1e-4)

    last_row = feats.iloc[[-1]].copy().reset_index(drop=True)
    # Ensure all feature_names exist
    for f in feature_names:
        if f not in last_row.columns:
            last_row[f] = np.nan

    return last_row


def predict(
    data=None,
    commodity="rice",
    artifacts_dir=None,
    version=None,
    horizon_days=None,
    return_intervals=False,
    history=None,
    horizon="short",
    **kwargs
):
    """Return an actual future modal price and a volatility class.

    The price model predicts a change internally. This API reconstructs the
    actual Rs./Quintal price by adding it to the supplied current modal price.
    Supports multi-commodity V1/V2 routing, direct horizons (14d, 30d), and empirical intervals.
    Accepts history as either 'data' or 'history', and horizon as 'short', 14, 30.
    """
    if data is None and history is not None:
        data = history
    if data is None:
        raise ValueError("Prediction data/history is required.")

    frame = _frame(data)
    commodity = str(commodity).strip().lower()

    supported_commodities = {"rice", "wheat", "potato", "tea"}
    if commodity not in supported_commodities:
        raise ValueError(
            f"Unsupported commodity: '{commodity}'. Supported commodities are: {', '.join(sorted(supported_commodities))}."
        )

    # Resolve horizon / horizon_days
    if horizon_days is None:
        if horizon is not None:
            h_str = str(horizon).strip().lower()
            if h_str in ("short", "1-7", "1", "1d", "short_term"):
                horizon_days = 1
            elif h_str in ("14", "14d"):
                horizon_days = 14
            elif h_str in ("30", "30d"):
                horizon_days = 30
            elif isinstance(horizon, (int, float)):
                horizon_days = int(horizon)
            else:
                raise ValueError(f"Unsupported horizon '{horizon}'. Valid options: 'short', 14, 30.")
        else:
            horizon_days = 1

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

    # Determine Version (defaults to 'v2' for wheat and potato when available, 'v1' otherwise)
    if version is None:
        if commodity in ["wheat", "potato"]:
            base = Path(artifacts_dir) if artifacts_dir else get_artifacts_dir()
            v2_exists = (base / commodity / "v2" / f"{commodity}_price_model.pkl").exists()
            version = "v2" if v2_exists else "v1"
        else:
            version = "v1"

    config, price_model, volatility_model, deadband_eps, res_dist = _artifacts(
        commodity, artifacts_dir=artifacts_dir, version=version, horizon_days=horizon_days
    )
    price_config = config["price_model"]
    price_features = price_config["features"]
    volatility_features = config["volatility_model"]["features"]
    reconstruction_column = price_config.get("reconstruction_column", "Modal Price (Rs./Quintal)")
    required = list(dict.fromkeys(price_features + volatility_features + [reconstruction_column]))
    missing = [c for c in required if c not in frame.columns]

    if missing:
        # Check if history is provided
        has_price_col = any(c.lower().replace(" ", "_") in ("modal_price", "price", "modal_price_(rs./quintal)") for c in frame.columns)
        if has_price_col and len(frame) >= 1:
            if version == "v2" and commodity in ["wheat", "potato"]:
                frame = _extract_v2_features_from_history(frame, required)
            else:
                frame = _extract_20_features_from_history(frame)
            missing = [c for c in required if c not in frame.columns]

    if missing:
        raise ValueError("Missing required prediction columns: " + ", ".join(missing))

    current_price = pd.to_numeric(frame.iloc[0][reconstruction_column], errors="coerce")
    if pd.isna(current_price) or current_price <= 0:
        raise ValueError(f"'{reconstruction_column}' must be a positive numeric current price.")

    # Predict price change
    predicted_change = float(price_model.predict(frame[price_features])[0])
    if deadband_eps > 0 and abs(predicted_change) < deadband_eps:
        predicted_change = 0.0

    predicted_price = float(current_price + predicted_change)
    if predicted_price <= 0:
        raise ValueError("Reconstructed predicted price is not positive.")

    # Predict volatility
    vol_classes = ["LOW", "MEDIUM", "HIGH"]
    vol_probs = None
    if hasattr(volatility_model, "predict_proba"):
        probs = volatility_model.predict_proba(frame[volatility_features])[0]
        class_idx = int(np.argmax(probs))
        volatility_str = vol_classes[class_idx]
        vol_probs = {cls: round(float(probs[i]), 4) for i, cls in enumerate(vol_classes)}
    else:
        predicted_code = int(volatility_model.predict(frame[volatility_features])[0])
        if predicted_code in VOLATILITY_REVERSE_LABEL_MAP:
            volatility_str = VOLATILITY_REVERSE_LABEL_MAP[predicted_code]
        elif 0 <= predicted_code < len(vol_classes):
            volatility_str = vol_classes[predicted_code]
        else:
            raise ValueError(f"Unknown volatility class code: {predicted_code}")

    out = {
        "predicted_price": round(predicted_price, 2),
        "volatility": volatility_str,
        "commodity": commodity,
        "version": version,
    }
    if horizon_days > 1:
        out["horizon_days"] = horizon_days
    if vol_probs is not None:
        out["volatility_probabilities"] = vol_probs

    # Empirical residual prediction intervals
    if (return_intervals or horizon_days > 1) and res_dist is not None:
        h_key = "short_term" if horizon_days <= 7 else (f"{horizon_days}d" if f"{horizon_days}d" in res_dist else "short_term")
        if h_key in res_dist:
            q = res_dist[h_key].get("quantiles", {})
            if "p10" in q and "p90" in q:
                out["prediction_interval"] = {
                    "p10": round(predicted_price + q["p10"], 2),
                    "p90": round(predicted_price + q["p90"], 2),
                    "coverage": "80% empirical holdout"
                }

    return out
