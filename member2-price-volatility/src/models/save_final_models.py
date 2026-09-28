"""
AgroVision — Save Final Models

Trains final XGBoost models on all training data and saves them
as commodity-specific .pkl files in artifacts/.

Saves:
    - {commodity}_price_model.pkl
    - {commodity}_volatility_model.pkl
    - {commodity}_feature_config.json
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

from config import (
    ARTIFACTS_DIR,
    PRICE_COLUMN,
    PRICE_FEATURES,
    PRICE_TARGET,
    OUTLIER_CLIP_PERCENTILES,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    VOLATILITY_LABEL_MAP,
    VOLATILITY_CLASSES,
    get_commodity,
    train_path,
    volatility_dataset_train_path,
    price_model_path,
    volatility_model_path,
    feature_config_path,
)


def save_models(train_df=None, vol_train_df=None):
    """
    Train final models on all training data and save to artifacts/.

    Parameters
    ----------
    train_df : pd.DataFrame, optional
        Price training data.
    vol_train_df : pd.DataFrame, optional
        Volatility training data.
    """

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    commodity = get_commodity()

    # ==========================================================
    # PRICE MODEL
    #
    # Must match xgboost_price.py exactly:
    #   target = future_price_change (not raw future price)
    #   outlier clipping at p1/p99
    #   same hyperparameters
    # ==========================================================
    print("\n" + "=" * 60)
    print(f"TRAINING FINAL PRICE MODEL [{commodity}]")
    print("=" * 60)

    if train_df is None:
        t_path = train_path()

        if not t_path.exists():
            raise FileNotFoundError(f"File not found: {t_path}")

        train_df = pd.read_csv(t_path)

    # Use only features available in the training data
    available_features = [
        f for f in PRICE_FEATURES
        if f in train_df.columns
    ]
    # PRICE_COLUMN is already in available_features; deduplicate
    needed_cols = list(dict.fromkeys(
        available_features + [PRICE_TARGET, PRICE_COLUMN]
    ))

    price_df = train_df[
        needed_cols
    ].dropna(
        subset=[PRICE_TARGET, PRICE_COLUMN]
    ).copy()

    # ---- Create price-difference target (same as xgboost_price.py) ----
    price_df["future_price_change"] = (
        price_df[PRICE_TARGET] - price_df[PRICE_COLUMN]
    )

    # ---- Clip extreme outliers (same as xgboost_price.py) ----
    lo_pct, hi_pct = OUTLIER_CLIP_PERCENTILES
    lo_val = price_df["future_price_change"].quantile(lo_pct / 100)
    hi_val = price_df["future_price_change"].quantile(hi_pct / 100)

    n_clipped = int(
        (price_df["future_price_change"] < lo_val).sum()
        + (price_df["future_price_change"] > hi_val).sum()
    )

    price_df["future_price_change"] = (
        price_df["future_price_change"].clip(lo_val, hi_val)
    )

    print(
        f"   Outlier clipping: [{lo_val:.0f}, {hi_val:.0f}] "
        f"(p{lo_pct}/p{hi_pct}), "
        f"clipped {n_clipped} rows"
    )

    X_price = price_df[available_features]
    y_price = price_df["future_price_change"]

    # ---- Same hyperparameters as xgboost_price.py ----
    price_model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:pseudohubererror",
        random_state=42,
        n_jobs=-1,
    )

    price_model.fit(
        X_price,
        y_price
    )

    print(f"   Target: future_price_change (price-difference)")
    print(f"   Trained on {len(price_df)} rows")

    p_model_path = price_model_path()

    joblib.dump(
        price_model,
        p_model_path
    )

    print(f"   Saved: {p_model_path}")

    # ==========================================================
    # VOLATILITY MODEL
    # ==========================================================
    print("\n" + "=" * 60)
    print(f"TRAINING FINAL VOLATILITY MODEL [{commodity}]")
    print("=" * 60)

    if vol_train_df is None:
        vt_path = volatility_dataset_train_path()

        if not vt_path.exists():
            raise FileNotFoundError(
                f"File not found: {vt_path}"
            )

        vol_train_df = pd.read_csv(vt_path)

    X_vol = vol_train_df[VOLATILITY_FEATURES]

    y_vol = (
        vol_train_df[VOLATILITY_TARGET]
        .map(VOLATILITY_LABEL_MAP)
    )

    # ----------------------------------------------------------
    # XGBoost volatility classifier
    # ----------------------------------------------------------
    vol_model = XGBClassifier(
    n_estimators=500,
    max_depth=3,
    learning_rate=0.03,
    min_child_weight=3,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softmax",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
    )

    # ----------------------------------------------------------
    # Class-balanced training
    # ----------------------------------------------------------
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_vol
    )

    print("\n   Class balancing:")
    print("   LOW    -> balanced weight")
    print("   MEDIUM -> balanced weight")
    print("   HIGH   -> balanced weight")

    vol_model.fit(
        X_vol,
        y_vol,
        sample_weight=sample_weights
    )

    print(f"   Trained on {len(vol_train_df)} rows")
    print("   Class balancing: ENABLED")

    v_model_path = volatility_model_path()

    joblib.dump(
        vol_model,
        v_model_path
    )

    print(f"   Saved: {v_model_path}")

    # ==========================================================
    # FEATURE CONFIG
    # ==========================================================
    feature_config = {
        "commodity": commodity,

        "price_model": {
            "type": "XGBRegressor",
            "target": "future_price_change",
            "note": "predicts (future_modal_price - current_price); add to current price to reconstruct",
            "features": available_features,
        },

        "volatility_model": {
            "type": "XGBClassifier",
            "target": VOLATILITY_TARGET,
            "classes": VOLATILITY_CLASSES,
            "features": VOLATILITY_FEATURES,
            "class_balanced": True,
        },
    }

    fc_path = feature_config_path()

    with open(
        fc_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            feature_config,
            f,
            indent=4
        )

    print(f"\n   Saved: {fc_path}")

    # ==========================================================
    # FINAL ARTIFACTS
    # ==========================================================
    print("\n" + "=" * 60)
    print("FINAL ARTIFACTS")
    print("=" * 60)

    for file_path in ARTIFACTS_DIR.iterdir():

        if commodity in file_path.name:
            print(f"   {file_path}")


def main():
    save_models()


if __name__ == "__main__":
    main()