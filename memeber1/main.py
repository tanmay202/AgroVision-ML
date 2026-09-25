"""
AgroVision - Main Pipeline Runner
End-to-end pipeline: Load -> Clean -> Features -> Train -> Evaluate

Usage:
    python main.py                  # Full pipeline with XGBoost tuning
    python main.py --no-tune        # Skip hyperparameter tuning (faster)
    python main.py --predict-only   # Load saved model and predict
"""

import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_all_data
from src.data_cleaner import clean_and_save
from src.feature_engineer import engineer_features
from src.model_trainer import train_models
from src.model_evaluator import evaluate_model
from src.predict import load_model


def run_pipeline(tune_xgboost=True):
    """
    Run the complete Yield ML pipeline.

    Steps:
        1. Load raw CSV data from data/raw/
        2. Clean data and extract temporal features
        3. Engineer ML features (lags, rolling, encoding)
        4. Train models (LR, RF, XGBoost)
        5. Evaluate best model and save plots
    """
    print("=" * 60)
    print("  AgroVision - Yield ML Pipeline")
    print("  Member 1: Agricultural Data + Crop-Yield Forecasting")
    print("  Data: Tea Mandi Arrivals & Prices")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Load data
    data = load_all_data()

    # Step 2: Clean
    cleaned_df = clean_and_save(data)

    # Step 3: Feature engineering
    feature_df = engineer_features(cleaned_df)

    # Step 4: Train models
    results = train_models(feature_df, tune_xgboost=tune_xgboost)

    # Step 5: Evaluate
    evaluate_model(results)

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"[OK] PIPELINE COMPLETE in {elapsed:.1f} seconds")
    print("=" * 60)
    print(f"\nOutputs:")
    print(f"   Cleaned data    : data/processed/cleaned_dataset.csv")
    print(f"   Feature matrix  : data/features/feature_matrix.csv")
    print(f"   Trained model   : models/yield_model.pkl")
    print(f"   Evaluation      : outputs/")
    print(f"\nFor Member 3 (Engineering):")
    print(f"   Use src/predict.py to load the model and make predictions")
    print(f"   from the FastAPI service.")


def main():
    """Parse arguments and run."""
    args = sys.argv[1:]

    if "--predict-only" in args:
        print("Loading saved model for prediction...")
        model = load_model()
        print("Model loaded. Use predict.py for inference.")
        return

    tune = "--no-tune" not in args

    if not tune:
        print("[FAST] Running in fast mode (no hyperparameter tuning)\n")

    run_pipeline(tune_xgboost=tune)


if __name__ == "__main__":
    main()
