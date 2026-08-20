import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json

import joblib
import pandas as pd
from xgboost import XGBRegressor, XGBClassifier

from config import (
    TRAIN_PATH,
    VOLATILITY_DATASET_TRAIN_PATH,
    ARTIFACTS_DIR,
    PRICE_MODEL_PATH,
    VOLATILITY_MODEL_PATH,
    FEATURE_CONFIG_PATH,
    VOLATILITY_CONFIG_PATH,
    PRICE_FEATURES,
    PRICE_TARGET,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    VOLATILITY_LABEL_MAP,
    VOLATILITY_CLASSES,
)


def main():
    # --------------------------------------------------
    # 1. Create artifacts directory
    # --------------------------------------------------
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ==========================================================
    # PRICE MODEL
    # ==========================================================

    # --------------------------------------------------
    # 2. Load price training data
    # --------------------------------------------------
    if not TRAIN_PATH.exists():
        print(f"ERROR: File not found: {TRAIN_PATH}")
        return

    price_df = pd.read_csv(TRAIN_PATH)

    price_df = price_df[
        PRICE_FEATURES + [PRICE_TARGET]
    ].dropna().copy()

    X_price = price_df[PRICE_FEATURES]
    y_price = price_df[PRICE_TARGET]

    # --------------------------------------------------
    # 3. Train final XGBoost price model
    # --------------------------------------------------
    print("=" * 60)
    print("TRAINING FINAL PRICE MODEL")
    print("=" * 60)

    price_model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )

    price_model.fit(
        X_price,
        y_price
    )

    print("Final XGBoost price model trained.")
    print(f"Training rows used: {len(price_df)}")

    # --------------------------------------------------
    # 4. Save price model
    # --------------------------------------------------
    joblib.dump(
        price_model,
        PRICE_MODEL_PATH
    )

    print(f"Saved: {PRICE_MODEL_PATH}")

    # ==========================================================
    # VOLATILITY MODEL
    # ==========================================================

    # --------------------------------------------------
    # 5. Load volatility training data
    # --------------------------------------------------
    if not VOLATILITY_DATASET_TRAIN_PATH.exists():
        print(
            f"ERROR: File not found: "
            f"{VOLATILITY_DATASET_TRAIN_PATH}"
        )
        return

    vol_df = pd.read_csv(
        VOLATILITY_DATASET_TRAIN_PATH
    )

    X_vol = vol_df[VOLATILITY_FEATURES]
    y_vol = vol_df[VOLATILITY_TARGET]

    # --------------------------------------------------
    # 6. Encode volatility classes
    # --------------------------------------------------
    y_vol_encoded = y_vol.map(VOLATILITY_LABEL_MAP)

    # --------------------------------------------------
    # 7. Train final XGBoost volatility model
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING FINAL VOLATILITY MODEL")
    print("=" * 60)

    vol_model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softmax",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )

    vol_model.fit(
        X_vol,
        y_vol_encoded
    )

    print("Final XGBoost volatility model trained.")

    # --------------------------------------------------
    # 8. Save volatility model
    # --------------------------------------------------
    joblib.dump(
        vol_model,
        VOLATILITY_MODEL_PATH
    )

    print(f"Saved: {VOLATILITY_MODEL_PATH}")

    # --------------------------------------------------
    # 9. Save feature configuration
    # --------------------------------------------------
    feature_config = {
        "price_model": {
            "type": "XGBRegressor",
            "target": PRICE_TARGET,
            "features": PRICE_FEATURES,
        },
        "volatility_model": {
            "type": "XGBClassifier",
            "target": VOLATILITY_TARGET,
            "classes": VOLATILITY_CLASSES,
            "features": VOLATILITY_FEATURES,
        },
    }

    with open(
        FEATURE_CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            feature_config,
            file,
            indent=4
        )

    print(f"Saved: {FEATURE_CONFIG_PATH}")

    # --------------------------------------------------
    # 10. Final artifact summary
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL ARTIFACTS")
    print("=" * 60)

    for file_path in ARTIFACTS_DIR.iterdir():
        print(file_path)


if __name__ == "__main__":
    main()