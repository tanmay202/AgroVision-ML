import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import (
    VOLATILITY_DATASET_TRAIN_PATH,
    VOLATILITY_DATASET_TEST_PATH,
    VOLATILITY_FEATURES,
    VOLATILITY_TARGET,
    VOLATILITY_CLASSES,
)


def main():
    # --------------------------------------------------
    # 1. Load datasets
    # --------------------------------------------------
    if not VOLATILITY_DATASET_TRAIN_PATH.exists():
        print(f"ERROR: File not found: {VOLATILITY_DATASET_TRAIN_PATH}")
        return

    if not VOLATILITY_DATASET_TEST_PATH.exists():
        print(f"ERROR: File not found: {VOLATILITY_DATASET_TEST_PATH}")
        return

    train_df = pd.read_csv(VOLATILITY_DATASET_TRAIN_PATH)
    test_df = pd.read_csv(VOLATILITY_DATASET_TEST_PATH)

    print("=" * 60)
    print("VOLATILITY CLASSIFIER (Random Forest)")
    print("=" * 60)

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")
    print(f"Features: {len(VOLATILITY_FEATURES)}")

    # --------------------------------------------------
    # 2. X and y
    # --------------------------------------------------
    X_train = train_df[VOLATILITY_FEATURES]
    y_train = train_df[VOLATILITY_TARGET]

    X_test = test_df[VOLATILITY_FEATURES]
    y_test = test_df[VOLATILITY_TARGET]

    # --------------------------------------------------
    # 3. Class distribution
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAIN CLASS DISTRIBUTION")
    print("=" * 60)

    print(y_train.value_counts())

    print("\n" + "=" * 60)
    print("TEST CLASS DISTRIBUTION")
    print("=" * 60)

    print(y_test.value_counts())

    # --------------------------------------------------
    # 4. Train Random Forest classifier
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
        class_weight="balanced",
    )

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------------------
    # 5. Predictions on TEST data
    # --------------------------------------------------
    predictions = model.predict(X_test)

    # --------------------------------------------------
    # 6. Accuracy
    # --------------------------------------------------
    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("TEST ACCURACY")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    # --------------------------------------------------
    # 7. Classification report
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST CLASSIFICATION REPORT")
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
    # 8. Confusion matrix
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TEST CONFUSION MATRIX")
    print("=" * 60)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=VOLATILITY_CLASSES,
    )

    print(
        pd.DataFrame(
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
    )

    # --------------------------------------------------
    # 9. Feature importance
    # --------------------------------------------------
    importance = pd.DataFrame({
        "feature": VOLATILITY_FEATURES,
        "importance": model.feature_importances_,
    }).sort_values(
        by="importance",
        ascending=False,
    )

    print("\n" + "=" * 60)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 60)

    print(
        importance.head(10).to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()