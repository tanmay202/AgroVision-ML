"""
AgroVision - Model Trainer
Trains XGBoost and Random Forest models for tea arrival prediction.
Uses time-based split and log-transformed target for better performance.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

from src.config import (
    COLUMNS,
    MODELS_DIR,
    YIELD_MODEL_FILE,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CV_FOLDS,
    XGBOOST_PARAM_GRID_FAST,
)
from src.feature_engineer import get_feature_columns


def prepare_data(df):
    """
    Prepare feature matrix X and target vector y.
    Uses log-transform on target to handle skewed arrival distribution.
    Uses time-based split (last 20% chronologically) for realistic evaluation.

    Returns:
        X_train, X_test, y_train, y_test, feature_names, use_log_target
    """
    feature_cols = get_feature_columns(df)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame!")

    if not feature_cols:
        raise ValueError("No feature columns found! Check feature engineering output.")

    X = df[feature_cols].copy()
    y = df[TARGET_COLUMN].copy()

    # Log-transform the target (arrivals are highly skewed)
    # This dramatically improves model performance on skewed data
    use_log_target = True
    y_log = np.log1p(y)

    # Handle any remaining NaN/Inf in features
    nan_counts = X.isna().sum()
    if nan_counts.any():
        nan_cols = nan_counts[nan_counts > 0]
        print(f"\n   [WARN] NaN found in features, filling with median:")
        for col_name in nan_cols.index:
            median_val = X[col_name].median()
            X[col_name] = X[col_name].fillna(median_val)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Time-based split: sort by date, use last 20% as test
    # This is more realistic than random split for time-series data
    date_col = COLUMNS["date"]
    if date_col in df.columns:
        print("   Using time-based train/test split (chronological)")
        # Data should already be sorted by date from cleaner
        split_idx = int(len(X) * (1 - TEST_SIZE))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y_log.iloc[:split_idx]
        y_test = y_log.iloc[split_idx:]
    else:
        print("   Using random train/test split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    print(f"\n   [STATS] Data split:")
    print(f"      Training:  {len(X_train):,} samples")
    print(f"      Testing:   {len(X_test):,} samples")
    print(f"      Features:  {len(feature_cols)}")
    print(f"      Target:    log1p({TARGET_COLUMN})")

    return X_train, X_test, y_train, y_test, feature_cols, use_log_target


def train_baseline(X_train, y_train, X_test, y_test):
    """Train a simple Linear Regression baseline."""
    print("\n[TRAIN] Training Baseline: Linear Regression...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"   R2 (train): {train_score:.4f}")
    print(f"   R2 (test):  {test_score:.4f}")

    return model, test_score


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train a Random Forest Regressor with regularization."""
    print("\n[TRAIN] Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,             # Reduced from 10 to prevent overfitting
        min_samples_split=10,    # Regularization
        min_samples_leaf=5,      # Regularization
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"   R2 (train): {train_score:.4f}")
    print(f"   R2 (test):  {test_score:.4f}")

    return model, test_score


def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train a Gradient Boosting Regressor (sklearn)."""
    print("\n[TRAIN] Training Gradient Boosting...")
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"   R2 (train): {train_score:.4f}")
    print(f"   R2 (test):  {test_score:.4f}")

    return model, test_score


def train_xgboost(X_train, y_train, X_test, y_test, tune=True):
    """Train an XGBoost Regressor with optional hyperparameter tuning."""
    print("\n[TRAIN] Training XGBoost Regressor...")

    if tune:
        print("   Running GridSearchCV (this may take a few minutes)...")
        base_model = XGBRegressor(
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
            reg_alpha=0.1,       # L1 regularization
            reg_lambda=1.0,      # L2 regularization
        )

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=XGBOOST_PARAM_GRID_FAST,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(X_train, y_train)

        model = grid_search.best_estimator_
        print(f"\n   Best parameters: {grid_search.best_params_}")
        print(f"   Best CV R2: {grid_search.best_score_:.4f}")
    else:
        model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=RANDOM_STATE,
            verbosity=0,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"   R2 (train): {train_score:.4f}")
    print(f"   R2 (test):  {test_score:.4f}")

    return model, test_score


def train_models(df, tune_xgboost=True):
    """
    Full model training pipeline.

    Trains 4 models:
        1. Linear Regression (baseline)
        2. Random Forest
        3. Gradient Boosting
        4. XGBoost (with optional tuning)

    Selects best model by test R2 score and saves it.
    """
    print("\n" + "=" * 60)
    print("STEP 4: MODEL TRAINING")
    print("=" * 60)

    # Prepare data (with log target + time-based split)
    X_train, X_test, y_train, y_test, feature_names, use_log_target = prepare_data(df)

    # Train all models
    results = {}

    lr_model, lr_score = train_baseline(X_train, y_train, X_test, y_test)
    results["Linear Regression"] = {"model": lr_model, "r2_test": lr_score}

    rf_model, rf_score = train_random_forest(X_train, y_train, X_test, y_test)
    results["Random Forest"] = {"model": rf_model, "r2_test": rf_score}

    gb_model, gb_score = train_gradient_boosting(X_train, y_train, X_test, y_test)
    results["Gradient Boosting"] = {"model": gb_model, "r2_test": gb_score}

    xgb_model, xgb_score = train_xgboost(X_train, y_train, X_test, y_test, tune=tune_xgboost)
    results["XGBoost"] = {"model": xgb_model, "r2_test": xgb_score}

    # Select best model
    best_name = max(results, key=lambda k: results[k]["r2_test"])

    print("\n" + "-" * 40)
    print("[STATS] MODEL COMPARISON")
    print("-" * 40)
    for name, res in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(f"   {name:25s}  R2 = {res['r2_test']:.4f}{marker}")

    best_model = results[best_name]["model"]

    # Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_info = {
        "model": best_model,
        "feature_names": feature_names,
        "use_log_target": use_log_target,
        "model_name": best_name,
    }
    joblib.dump(model_info, YIELD_MODEL_FILE)
    print(f"\n[SAVE] Saved best model ({best_name}) to: {YIELD_MODEL_FILE}")

    # Store extra info for evaluation
    results["_best_name"] = best_name
    results["_best_model"] = best_model
    results["_X_train"] = X_train
    results["_X_test"] = X_test
    results["_y_train"] = y_train
    results["_y_test"] = y_test
    results["_feature_names"] = feature_names
    results["_use_log_target"] = use_log_target

    return results
