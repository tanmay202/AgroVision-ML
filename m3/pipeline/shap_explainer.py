"""
AgroVision Request-Time Multi-Crop TreeSHAP Explainer
Computes exact Shapley feature attributions (in ₹/quintal) on demand using
XGBoost's C++ TreeSHAP (pred_contribs=True) and CatBoost's native TreeSHAP
(get_feature_importance(type='ShapValues')) across Rice, Potato, and Wheat.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from psycopg.rows import dict_row

LOCAL_ROOT = Path(__file__).resolve().parent.parent
SIBLING_ROOT = Path.home() / "projects/agrovision-ml"
REPO_ROOT = LOCAL_ROOT if (LOCAL_ROOT / "deliverables/member2-ml").exists() else SIBLING_ROOT
PKG_DIR = REPO_ROOT / "deliverables/member2-ml"
M2_SRC_DIR = REPO_ROOT / "member2-price-volatility/src"
for p in (PKG_DIR, M2_SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from src.feature_engineering import generate_48_price_features, normalize_input_data
from src.predictor import RicePricePredictor
from models.predict import _artifacts as m2_v2_artifacts, _extract_v2_features_from_history

from pipeline.m2_adapter import DB_CONFIG, lookup_series_meta_from_db

HORIZON_MAP: dict[str, int] = {
    "short": 1,
    "14d": 14,
    "30d": 30,
}

FEATURE_LABELS: dict[str, tuple[str, str]] = {
    "modal_price": ("Current Modal Price", "₹/q"),
    "min_price": ("Daily Floor Price", "₹/q"),
    "max_price": ("Daily Ceiling Price", "₹/q"),
    "arrivals": ("Daily Mandi Arrivals", "MT"),
    "lag_1": ("Prior-Session Price (Lag 1)", "₹/q"),
    "lag_7": ("7-Day Lagged Price", "₹/q"),
    "lag_14": ("14-Day Lagged Price", "₹/q"),
    "lag_30": ("30-Day Lagged Price", "₹/q"),
    "rolling_mean_7": ("7-Day Rolling Price Avg", "₹/q"),
    "rolling_mean_14": ("14-Day Rolling Price Avg", "₹/q"),
    "rolling_mean_30": ("30-Day Rolling Price Avg", "₹/q"),
    "rolling_std_7": ("7-Day Price Volatility (Std)", "₹/q"),
    "rolling_std_14": ("14-Day Price Volatility (Std)", "₹/q"),
    "rolling_std_30": ("30-Day Price Volatility (Std)", "₹/q"),
    "price_pct_change": ("Daily Price Momentum", "%"),
    "price_diff_1": ("1-Session Price Change", "₹/q"),
    "price_diff_7": ("7-Day Price Change", "₹/q"),
    "price_spread": ("Intraday Min-Max Spread", "₹/q"),
    "price_position_in_range": ("Position in Daily Price Band", "ratio"),
    "arrival_lag_1": ("Prior-Session Arrivals", "MT"),
    "arrival_lag_7": ("7-Day Lagged Arrivals", "MT"),
    "arrival_change": ("Daily Arrival Change", "MT"),
    "arrival_pct_change": ("Arrival Supply Surge/Drop", "%"),
    "arrival_rolling_mean_7": ("7-Day Avg Arrivals", "MT"),
    "arrival_rolling_mean_14": ("14-Day Avg Arrivals", "MT"),
    "month": ("Calendar Month (Seasonality)", "month"),
    "week_of_year": ("Week of Year", "week"),
    "day_of_week": ("Market Day of Week", "day"),
    "day": ("Day of Month", "day"),
    "year": ("Calendar Year (Macro Trend)", "year"),
    "accel_1_7": ("Price Acceleration (1d vs 7d)", "₹/q"),
    "ewma_7": ("7-Day Exponential Moving Avg", "₹/q"),
    "ewma_14": ("14-Day Exponential Moving Avg", "₹/q"),
    "ewma_30": ("30-Day Exponential Moving Avg", "₹/q"),
    "last_move_abs": ("Last Session Absolute Price Swing", "₹/q"),
    "price_change_1": ("1-Session Price Change", "₹/q"),
    "price_change_7": ("7-Day Net Price Change", "₹/q"),
    "price_change_14": ("14-Day Net Price Change", "₹/q"),
}

_RICE_PREDICTOR_SINGLETON: RicePricePredictor | None = None


def _get_rice_predictor() -> RicePricePredictor:
    global _RICE_PREDICTOR_SINGLETON
    if _RICE_PREDICTOR_SINGLETON is None:
        _RICE_PREDICTOR_SINGLETON = RicePricePredictor(artifacts_dir=PKG_DIR / "artifacts")
    return _RICE_PREDICTOR_SINGLETON


def _humanize_feature(raw_name: str) -> tuple[str, str]:
    if raw_name in FEATURE_LABELS:
        return FEATURE_LABELS[raw_name]
    low = raw_name.lower()
    if "price" in low or "lag_" in low or "rolling_mean" in low or "rolling_std" in low or "ema" in low:
        unit = "₹/q"
    elif "arrival" in low:
        unit = "MT"
    elif "pct" in low or "return" in low or "ratio" in low:
        unit = "%"
    else:
        unit = "val"
    pretty = raw_name.replace("_", " ").strip().title()
    return pretty, unit


def _unwrap_estimator(estimator: Any) -> Any:
    if hasattr(estimator, "named_steps"):
        return list(estimator.named_steps.values())[-1]
    return estimator


def _compute_tree_contribs(estimator: Any, X_row: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    core = _unwrap_estimator(estimator)
    if isinstance(core, CatBoostRegressor) or "CatBoost" in type(core).__name__:
        pool = Pool(X_row[feature_names], feature_names=feature_names)
        shap_matrix = core.get_feature_importance(pool, type="ShapValues")
        return np.asarray(shap_matrix[0], dtype=np.float64)

    if isinstance(core, xgb.Booster):
        booster = core
    elif hasattr(core, "get_booster"):
        booster = core.get_booster()
    else:
        raise TypeError(f"Unsupported estimator type for TreeSHAP: {type(core)}")

    dmat = xgb.DMatrix(X_row[feature_names].to_numpy(dtype=np.float32), feature_names=feature_names)
    return np.asarray(booster.predict(dmat, pred_contribs=True)[0], dtype=np.float64)


def explain_forecast(
    mandi_id: str,
    crop_id: str,
    horizon_label: str = "short",
    top_k: int = 8,
) -> dict[str, Any]:
    """
    Computes exact TreeSHAP feature contributions for any (mandi_id, crop_id, horizon_label)
    across Rice, Potato, and Wheat.
    """
    h_norm = horizon_label.lower().strip()
    if h_norm not in HORIZON_MAP:
        raise ValueError(f"Invalid horizon '{horizon_label}'. Expected one of {list(HORIZON_MAP.keys())}.")
    horizon_days = HORIZON_MAP[h_norm]

    meta = lookup_series_meta_from_db(mandi_id, crop_id)
    comm = meta["commodity"]

    with psycopg.connect(**DB_CONFIG, row_factory=dict_row) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                (time AT TIME ZONE 'Asia/Kolkata')::date AS reported_date,
                modal_price_inr,
                arrivals_mt
            FROM features.market_features
            WHERE mandi_id = %s AND crop_id = %s
            ORDER BY time ASC
            LIMIT 400
            """,
            (mandi_id, crop_id),
        )
        rows = cur.fetchall()

    if len(rows) < 30:
        raise ValueError(f"Insufficient history ({len(rows)} rows) for {mandi_id}/{crop_id}")

    records = [
        {
            "State Name": meta["State Name"],
            "District Name": meta["District Name"],
            "Market Name": meta["Market Name"],
            "Variety": meta["Variety"],
            "Group": meta["Group"],
            "Reported Date": str(r["reported_date"]),
            "Modal Price (Rs./Quintal)": float(r["modal_price_inr"]),
            "Arrivals (Tonnes)": float(r["arrivals_mt"] or 0.0),
        }
        for r in rows
    ]

    if comm == "rice":
        predictor = _get_rice_predictor()
        clean_df, _, _, resolved_date = normalize_input_data(
            records,
            market=meta["Market Name"],
            variety=meta["Variety"],
            min_observations=30,
        )
        feat_df = generate_48_price_features(clean_df)
        feature_names = predictor.get_required_features()
        X_row = feat_df.iloc[[-1]][feature_names].copy().apply(pd.to_numeric, errors="coerce")
        if horizon_days == 1:
            estimator = predictor.short_term_model
            model_version = predictor.short_term_config.get("model_version", "xgb_short_v1")
        else:
            estimator, cfg = predictor._load_horizon_model(horizon_days)
            model_version = cfg.get("model_version", f"xgb_{horizon_days}d_v1")
        current_price = float(feat_df.iloc[-1]["Modal Price (Rs./Quintal)"])
    else:
        raw_df = pd.DataFrame(records)
        resolved_date = str(raw_df.iloc[-1]["Reported Date"])
        current_price = float(raw_df.iloc[-1]["Modal Price (Rs./Quintal)"])
        v2_art_dir = REPO_ROOT / "member2-price-volatility/artifacts"
        cfg, estimator, _, _, _ = m2_v2_artifacts(
            commodity=comm, artifacts_dir=v2_art_dir, version="v2", horizon_days=horizon_days
        )
        feature_names = list((cfg.get("price_model") or {}).get("features") or cfg.get("price_features") or cfg.get("features") or [])
        feat_df = _extract_v2_features_from_history(raw_df, feature_names)
        for col in feature_names:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        X_row = feat_df.iloc[[-1]][feature_names].copy().apply(pd.to_numeric, errors="coerce").fillna(0.0)
        model_version = f"{comm}-price-v2-{h_norm}"

    contribs = _compute_tree_contribs(estimator, X_row, feature_names)
    shap_vals = contribs[:-1]
    base_change = float(contribs[-1])
    predicted_change = float(np.sum(shap_vals) + base_change)
    predicted_price = current_price + predicted_change

    ranked_idx = np.argsort(np.abs(shap_vals))[::-1]
    top_features: list[dict[str, Any]] = []
    for idx in ranked_idx[:top_k]:
        fname = feature_names[int(idx)]
        fval = float(X_row.iloc[0, int(idx)])
        sval = float(shap_vals[int(idx)])
        label, unit = _humanize_feature(fname)
        direction = "UPWARD" if sval > 0.05 else ("DOWNWARD" if sval < -0.05 else "NEUTRAL")
        top_features.append(
            {
                "feature": fname,
                "label": label,
                "unit": unit,
                "feature_value": round(fval, 4),
                "shap_value_inr": round(sval, 2),
                "direction": direction,
            }
        )

    other_sum = float(np.sum(shap_vals[ranked_idx[top_k:]]))

    return {
        "mandi_id": mandi_id,
        "mandi_name": meta["Market Name"],
        "crop_id": crop_id,
        "variety": meta["Variety"],
        "horizon_label": h_norm,
        "horizon_days": horizon_days,
        "forecast_origin_date": resolved_date,
        "model_version": model_version,
        "current_modal_price": round(current_price, 2),
        "base_expected_change_inr": round(base_change, 2),
        "predicted_change_inr": round(predicted_change, 2),
        "predicted_modal_price": round(predicted_price, 2),
        "other_features_shap_inr": round(other_sum, 2),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "top_features": top_features,
    }
