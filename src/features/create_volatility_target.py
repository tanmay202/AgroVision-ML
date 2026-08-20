import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd

from config import (
    TRAIN_PATH,
    VOLATILITY_TRAIN_PATH,
    VOLATILITY_CONFIG_PATH,
    VOLATILITY_TARGET,
    ARTIFACTS_DIR,
)


PRICE_CHANGE_COLUMN = "price_pct_change"


def main():
    if not TRAIN_PATH.exists():
        print(f"ERROR: File not found: {TRAIN_PATH}")
        return

    df = pd.read_csv(TRAIN_PATH)

    # --------------------------------------------------
    # 1. Absolute price movement
    # --------------------------------------------------
    df["abs_price_pct_change"] = (
        df[PRICE_CHANGE_COLUMN].abs()
    )

    # --------------------------------------------------
    # 2. Separate zero and non-zero movements
    # --------------------------------------------------
    non_zero_changes = df.loc[
        df["abs_price_pct_change"] > 0,
        "abs_price_pct_change"
    ].dropna()

    print("=" * 60)
    print("VOLATILITY DATA")
    print("=" * 60)

    print(
        f"Total rows: {len(df)}"
    )

    print(
        f"Zero price-change rows: "
        f"{(df['abs_price_pct_change'] == 0).sum()}"
    )

    print(
        f"Non-zero price-change rows: "
        f"{len(non_zero_changes)}"
    )

    # --------------------------------------------------
    # 3. Data-driven thresholds
    #
    # Calculate thresholds ONLY from non-zero
    # historical price movements.
    # --------------------------------------------------
    low_threshold = non_zero_changes.quantile(1 / 3)

    high_threshold = non_zero_changes.quantile(2 / 3)

    print("\n" + "=" * 60)
    print("VOLATILITY THRESHOLDS")
    print("=" * 60)

    print(
        f"Low/Medium threshold : "
        f"{low_threshold:.4f}%"
    )

    print(
        f"Medium/High threshold: "
        f"{high_threshold:.4f}%"
    )

    # --------------------------------------------------
    # 4. Save thresholds to config
    # --------------------------------------------------
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    volatility_config = {
        "threshold_method": (
            "training-data quantiles "
            "of non-zero absolute "
            "price percentage changes"
        ),
        "low_medium_threshold_percent": round(
            float(low_threshold), 4
        ),
        "medium_high_threshold_percent": round(
            float(high_threshold), 4
        ),
        "classes": {
            "LOW": f"0% to {low_threshold:.4f}%",
            "MEDIUM": f">{low_threshold:.4f}% to {high_threshold:.4f}%",
            "HIGH": f">{high_threshold:.4f}%",
        },
    }

    with open(
        VOLATILITY_CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            volatility_config,
            file,
            indent=4
        )

    print(f"\nThresholds saved to: {VOLATILITY_CONFIG_PATH}")

    # --------------------------------------------------
    # 5. Create volatility classes
    # --------------------------------------------------
    def classify_volatility(value):
        if pd.isna(value):
            return pd.NA

        # No price movement = LOW
        if value == 0:
            return "LOW"

        if value <= low_threshold:
            return "LOW"

        if value <= high_threshold:
            return "MEDIUM"

        return "HIGH"

    df[VOLATILITY_TARGET] = (
        df["abs_price_pct_change"]
        .apply(classify_volatility)
    )

    # --------------------------------------------------
    # 6. Class distribution
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("VOLATILITY CLASS DISTRIBUTION")
    print("=" * 60)

    print(
        df[VOLATILITY_TARGET]
        .value_counts(dropna=False)
    )

    print("\nPercentages:")
    print(
        (
            df[VOLATILITY_TARGET]
            .value_counts(normalize=True)
            * 100
        ).round(2)
    )

    # --------------------------------------------------
    # 7. Examples from each class
    # --------------------------------------------------
    print("\n" + "=" * 60)
    print("EXAMPLES BY CLASS")
    print("=" * 60)

    example_columns = [
        "Market Name",
        "Variety",
        "Reported Date",
        "Modal Price (Rs./Quintal)",
        PRICE_CHANGE_COLUMN,
        "abs_price_pct_change",
        VOLATILITY_TARGET,
    ]

    for class_name in ["LOW", "MEDIUM", "HIGH"]:
        print(f"\n--- {class_name} ---")

        class_rows = (
            df[df[VOLATILITY_TARGET] == class_name]
            [example_columns]
            .head(5)
        )

        print(
            class_rows.to_string(index=False)
        )

    # --------------------------------------------------
    # 8. Save
    # --------------------------------------------------
    df.to_csv(
        VOLATILITY_TRAIN_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("OUTPUT")
    print("=" * 60)

    print(f"Saved to: {VOLATILITY_TRAIN_PATH}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()