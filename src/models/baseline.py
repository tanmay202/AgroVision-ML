import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    FEATURES_FINAL_PATH,
    DATE_COLUMN,
    PRICE_COLUMN,
    PRICE_TARGET,
    GROUP_COLUMNS,
)


def calculate_mape(actual, predicted):
    """
    Calculate MAPE while ignoring rows where actual price is zero.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs(
            (actual[mask] - predicted[mask])
            / actual[mask]
        )
    ) * 100


def main():
    # --------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------
    if not FEATURES_FINAL_PATH.exists():
        print(f"ERROR: File not found: {FEATURES_FINAL_PATH}")
        return

    df = pd.read_csv(FEATURES_FINAL_PATH)

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    # --------------------------------------------------
    # 2. Sort chronologically
    # --------------------------------------------------
    df = df.sort_values(
        by=GROUP_COLUMNS + [DATE_COLUMN]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 3. Baseline prediction
    # --------------------------------------------------
    actual = df[PRICE_TARGET]

    predicted = df[PRICE_COLUMN]

    # --------------------------------------------------
    # 4. Metrics
    # --------------------------------------------------
    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

    mape = calculate_mape(
        actual,
        predicted
    )

    # --------------------------------------------------
    # 5. Results
    # --------------------------------------------------
    print("=" * 60)
    print("BASELINE MODEL")
    print("=" * 60)

    print("Strategy:")
    print("Predicted future price = current Modal Price")

    print("\nMetrics:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")
    print(f"MAPE : {mape:.2f}%")

    # --------------------------------------------------
    # 6. Example predictions
    # --------------------------------------------------
    results = pd.DataFrame({
        "date": df[DATE_COLUMN],
        "current_price": df[PRICE_COLUMN],
        "actual_future_price": actual,
        "baseline_prediction": predicted
    })

    print("\n" + "=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)

    print(results.head(10))


if __name__ == "__main__":
    main()