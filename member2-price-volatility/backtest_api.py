import requests
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
)

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = "data/processed/rice_cleaned.csv"

HOLDOUT_START = pd.Timestamp("2023-02-01")
HOLDOUT_END = pd.Timestamp("2024-02-01")

N_TESTS = 100
RANDOM_STATE = 42


def volatility_class(price_range):
    if price_range <= 0.5:
        return "LOW"
    elif price_range <= 50.0:
        return "MEDIUM"
    else:
        return "HIGH"


# ---------------------------------------------------------
# 1. LOAD REAL RICE DATA
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH)

df["Reported Date"] = pd.to_datetime(
    df["Reported Date"], errors="coerce"
)

df = df.dropna(
    subset=[
        "Market Name",
        "Variety",
        "Reported Date",
        "Modal Price (Rs./Quintal)",
        "Arrivals (Tonnes)",
    ]
).copy()

df = df.sort_values(
    ["Market Name", "Variety", "Reported Date"]
).reset_index(drop=True)


# ---------------------------------------------------------
# 2. BUILD REAL HISTORICAL TEST CASES
# ---------------------------------------------------------
candidates = []

for (market, variety), group in df.groupby(
    ["Market Name", "Variety"]
):

    group = group.sort_values("Reported Date").reset_index(drop=True)

    for i in range(29, len(group) - 1):

        prediction_row = group.iloc[i]

        prediction_date = prediction_row["Reported Date"]

        # Must belong to final holdout
        if not (
            HOLDOUT_START
            <= prediction_date
            <= HOLDOUT_END
        ):
            continue

        # Actual next observation
        future = group.iloc[i + 1:]

        if future.empty:
            continue

        actual_row = future.iloc[0]

        days_to_next = (
            actual_row["Reported Date"]
            - prediction_date
        ).days

        # Same target horizon as your model: 1–7 days
        if not (1 <= days_to_next <= 7):
            continue

        # Keep actual evaluation point inside final holdout
        if actual_row["Reported Date"] > HOLDOUT_END:
            continue

        candidates.append(
            {
                "market": market,
                "variety": variety,
                "prediction_date": prediction_date,
                "group": group.copy(),
                "position": i,
                "actual_row": actual_row,
            }
        )


print("=" * 60)
print("RICE FASTAPI REAL BACKTEST")
print("=" * 60)
print(f"Available test cases: {len(candidates)}")


if len(candidates) < N_TESTS:
    N_TESTS = len(candidates)

rng = np.random.default_rng(RANDOM_STATE)

selected_indices = rng.choice(
    len(candidates),
    size=N_TESTS,
    replace=False,
)

selected = [candidates[i] for i in selected_indices]


# ---------------------------------------------------------
# 3. CALL REAL API FOR EACH TEST CASE
# ---------------------------------------------------------
results = []

for number, case in enumerate(selected, start=1):

    market = case["market"]
    variety = case["variety"]
    prediction_date = case["prediction_date"]
    group = case["group"]
    position = case["position"]
    actual_row = case["actual_row"]

    # ONLY historical data.
    # Nothing after prediction_date is sent to API.
    history = group.iloc[: position + 1].copy()

    # Last 60 observations are enough for the 30-observation
    # feature history while keeping requests reasonably small.
    history = history.tail(60)

    if len(history) < 30:
        continue

    api_history = []

    for _, row in history.iterrows():
        api_history.append(
            {
                "reported_date": row["Reported Date"].strftime(
                    "%Y-%m-%d"
                ),
                "modal_price": float(
                    row["Modal Price (Rs./Quintal)"]
                ),
                "arrivals": float(
                    row["Arrivals (Tonnes)"]
                ),
            }
        )

    payload = {
        "market": market,
        "variety": variety,
        "prediction_date": prediction_date.strftime(
            "%Y-%m-%d"
        ),
        "history": api_history,
    }

    try:
        response = requests.post(
            API_URL,
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            print(
                f"[{number}/{N_TESTS}] API ERROR "
                f"{response.status_code}"
            )
            print(response.text)
            continue

        output = response.json()

        predicted_price = float(
            output["predicted_modal_price"]
        )

        actual_price = float(
            actual_row["Modal Price (Rs./Quintal)"]
        )

        # ---------------------------------------------
        # Actual future volatility
        # ---------------------------------------------
        future_5 = group.iloc[
            position + 1 : position + 6
        ]

        if len(future_5) >= 5:
            actual_range = (
                future_5["Modal Price (Rs./Quintal)"].max()
                - future_5["Modal Price (Rs./Quintal)"].min()
            )

            actual_vol_class = volatility_class(
                actual_range
            )

            predicted_vol_class = output[
                "predicted_volatility_class"
            ]

        else:
            actual_range = np.nan
            actual_vol_class = None
            predicted_vol_class = None

        results.append(
            {
                "market": market,
                "variety": variety,
                "prediction_date": prediction_date,
                "actual_date": actual_row["Reported Date"],
                "days_to_next": (
                    actual_row["Reported Date"]
                    - prediction_date
                ).days,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "abs_error": abs(
                    predicted_price - actual_price
                ),
                "predicted_volatility": predicted_vol_class,
                "actual_volatility": actual_vol_class,
            }
        )

        print(
            f"[{number}/{N_TESTS}] "
            f"{market} / {variety} / "
            f"{prediction_date.date()} → "
            f"Pred={predicted_price:.2f}, "
            f"Actual={actual_price:.2f}"
        )

    except Exception as e:
        print(
            f"[{number}/{N_TESTS}] REQUEST FAILED: {e}"
        )


# ---------------------------------------------------------
# 4. RESULTS
# ---------------------------------------------------------
results_df = pd.DataFrame(results)

if results_df.empty:
    print("\nNo successful API predictions.")
    raise SystemExit(1)


y_true = results_df["actual_price"]
y_pred = results_df["predicted_price"]

mae = mean_absolute_error(y_true, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_true, y_pred)
)

r2 = r2_score(y_true, y_pred)


print("\n")
print("=" * 60)
print("PRICE BACKTEST RESULTS")
print("=" * 60)

print(f"Successful API tests : {len(results_df)}")
print(f"MAE                  : ₹{mae:.2f}")
print(f"RMSE                 : ₹{rmse:.2f}")
print(f"R²                   : {r2:.4f}")


# ---------------------------------------------------------
# 5. VOLATILITY RESULTS
# ---------------------------------------------------------
vol_df = results_df.dropna(
    subset=[
        "predicted_volatility",
        "actual_volatility",
    ]
)

if not vol_df.empty:

    vol_accuracy = accuracy_score(
        vol_df["actual_volatility"],
        vol_df["predicted_volatility"],
    )

    vol_macro_f1 = f1_score(
        vol_df["actual_volatility"],
        vol_df["predicted_volatility"],
        average="macro",
        labels=["LOW", "MEDIUM", "HIGH"],
        zero_division=0,
    )

    cm = confusion_matrix(
        vol_df["actual_volatility"],
        vol_df["predicted_volatility"],
        labels=["LOW", "MEDIUM", "HIGH"],
    )

    print("\n")
    print("=" * 60)
    print("VOLATILITY BACKTEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy             : {vol_accuracy:.4f}"
    )

    print(
        f"Macro F1             : {vol_macro_f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print("              LOW   MEDIUM   HIGH")
    print(
        f"Actual LOW     {cm[0][0]:4d}   {cm[0][1]:4d}   {cm[0][2]:4d}"
    )
    print(
        f"Actual MEDIUM  {cm[1][0]:4d}   {cm[1][1]:4d}   {cm[1][2]:4d}"
    )
    print(
        f"Actual HIGH    {cm[2][0]:4d}   {cm[2][1]:4d}   {cm[2][2]:4d}"
    )


# ---------------------------------------------------------
# 6. SAVE RESULTS
# ---------------------------------------------------------
results_df.to_csv(
    "api_backtest_results.csv",
    index=False,
)

print("\nResults saved to:")
print("api_backtest_results.csv")

print("\n" + "=" * 60)
print("BACKTEST COMPLETE")
print("=" * 60)