"""
AgroVision — Evaluate Saved Volatility Model

Loads the production volatility model from artifacts/
and evaluates it on the untouched volatility test dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    VOLATILITY_CLASSES,
    VOLATILITY_LABEL_MAP,
    VOLATILITY_REVERSE_LABEL_MAP,
    volatility_dataset_test_path,
    volatility_model_path,
)


def main():

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    model_path = volatility_model_path()
    test_path = volatility_dataset_test_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    # --------------------------------------------------
    # Load model and test data
    # --------------------------------------------------
    print("=" * 60)
    print("EVALUATING SAVED VOLATILITY MODEL")
    print("=" * 60)

    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df[VOLATILITY_FEATURES]
    y_test = test_df[VOLATILITY_TARGET]

    # Encode labels
    y_test_encoded = y_test.map(VOLATILITY_LABEL_MAP)

    print(f"Model : {model_path}")
    print(f"Test rows: {len(test_df)}")

    # --------------------------------------------------
    # Prediction
    # --------------------------------------------------
    predictions_encoded = model.predict(X_test)

    predictions = pd.Series(
        predictions_encoded
    ).map(
        VOLATILITY_REVERSE_LABEL_MAP
    )

    # --------------------------------------------------
    # Accuracy
    # --------------------------------------------------
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("SAVED MODEL TEST ACCURACY")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    # --------------------------------------------------
    # Classification report
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVED MODEL CLASSIFICATION REPORT")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            labels=VOLATILITY_CLASSES,
            zero_division=0,
        )
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("SAVED MODEL CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=VOLATILITY_CLASSES,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual LOW",
            "Actual MEDIUM",
            "Actual HIGH",
        ],
        columns=[
            "Predicted LOW",
            "Predicted MEDIUM",
            "Predicted HIGH",
        ],
    )

    print(cm_df)

    # --------------------------------------------------
    # Prediction distribution
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("PREDICTION DISTRIBUTION")
    print("=" * 60)

    print(
        predictions.value_counts()
        .reindex(VOLATILITY_CLASSES, fill_value=0)
        .to_string()
    )


if __name__ == "__main__":
    main()