import sys
from pathlib import Path
from sklearn.utils.class_weight import compute_sample_weight
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from xgboost import XGBClassifier

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
    VOLATILITY_LABEL_MAP,
    VOLATILITY_REVERSE_LABEL_MAP,
)


def main():
    if not VOLATILITY_DATASET_TRAIN_PATH.exists():
        print(f"ERROR: Missing {VOLATILITY_DATASET_TRAIN_PATH}")
        return

    if not VOLATILITY_DATASET_TEST_PATH.exists():
        print(f"ERROR: Missing {VOLATILITY_DATASET_TEST_PATH}")
        return

    train_df = pd.read_csv(VOLATILITY_DATASET_TRAIN_PATH)
    test_df = pd.read_csv(VOLATILITY_DATASET_TEST_PATH)

    X_train = train_df[VOLATILITY_FEATURES]
    y_train = train_df[VOLATILITY_TARGET]

    X_test = test_df[VOLATILITY_FEATURES]
    y_test = test_df[VOLATILITY_TARGET]

    print("=" * 60)
    print("XGBOOST VOLATILITY CLASSIFIER")
    print("=" * 60)

    print(f"Training rows: {len(train_df)}")
    print(f"Testing rows : {len(test_df)}")

    # --------------------------------------------------
    # Convert labels to integers
    # --------------------------------------------------
    y_train_encoded = y_train.map(VOLATILITY_LABEL_MAP)
    y_test_encoded = y_test.map(VOLATILITY_LABEL_MAP)

    # --------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    model = XGBClassifier(
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

    # --------------------------------------------------
    # Class-balanced sample weights
    # --------------------------------------------------

    sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train_encoded
    )

    print("\nCLASS WEIGHTS")
    print(
    pd.DataFrame({
        "class": y_train_encoded,
        "weight": sample_weights
    }).groupby("class")["weight"].first()
    )

    # --------------------------------------------------
    # Train XGBoost
    # --------------------------------------------------
    model.fit(
    X_train,
    y_train_encoded,
    sample_weight=sample_weights
    )

    print("Training complete.")

    # --------------------------------------------------
    # Predict
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
    print("TEST ACCURACY")
    print("=" * 60)

    print(f"Accuracy: {accuracy:.4f}")

    # --------------------------------------------------
    # Classification report
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
    # Confusion matrix
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
    # Feature importance
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