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
    # 1. Load train/test
    # --------------------------------------------------
    if not VOLATILITY_DATASET_TRAIN_PATH.exists():
        print(f"ERROR: Missing {VOLATILITY_DATASET_TRAIN_PATH}")
        return

    if not VOLATILITY_DATASET_TEST_PATH.exists():
        print(f"ERROR: Missing {VOLATILITY_DATASET_TEST_PATH}")
        return

    train_df = pd.read_csv(VOLATILITY_DATASET_TRAIN_PATH)
    test_df = pd.read_csv(VOLATILITY_DATASET_TEST_PATH)

    print("=" * 60)
    print("TIME-AWARE VOLATILITY EVALUATION")
    print("=" * 60)

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    # --------------------------------------------------
    # 2. X / y
    # --------------------------------------------------
    X_train = train_df[VOLATILITY_FEATURES]
    y_train = train_df[VOLATILITY_TARGET]

    X_test = test_df[VOLATILITY_FEATURES]
    y_test = test_df[VOLATILITY_TARGET]

    # --------------------------------------------------
    # 3. Feature order check
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("FEATURE ORDER CHECK")
    print("=" * 60)

    if list(X_train.columns) == list(X_test.columns):
        print("PASS: Train/test feature order matches.")
    else:
        print("ERROR: Train/test feature order does not match.")
        return

    # --------------------------------------------------
    # 4. Class distributions
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
    # 5. Train classifier
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING RANDOM FOREST")
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
    # 6. Predict future test period
    # --------------------------------------------------
    predictions = model.predict(X_test)

    # --------------------------------------------------
    # 7. Accuracy
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
    # 8. Precision / Recall / F1
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
    # 9. Confusion matrix
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
    # 10. Feature importance
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