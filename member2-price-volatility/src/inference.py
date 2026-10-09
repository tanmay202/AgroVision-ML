import os, sys, json, joblib, warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC = Path(__file__).resolve().parent
PROJECT_ROOT = SRC.parent

sys.path.insert(0, str(SRC))

from config import (
    DATE_COLUMN, PRICE_COLUMN, ARRIVAL_COLUMN, GROUP_COLUMNS,
)

class InsufficientHistoryError(Exception):
    pass

class RiceInferencePipeline:
    def __init__(self, artifacts_dir=None):
        if artifacts_dir is not None:
            self.artifacts_dir = Path(artifacts_dir)
        else:
            # Multi-path discovery supporting standalone and repository deployments
            candidates = []
            env_dir = os.environ.get("AGROVISION_ARTIFACTS_DIR")
            if env_dir:
                candidates.append(Path(env_dir))
            candidates.append(PROJECT_ROOT / "artifacts")
            candidates.append(PROJECT_ROOT.parent / "artifacts")

            self.artifacts_dir = None
            for cand in candidates:
                if cand.exists() and (cand / "rice_price_final_config.json").exists() and (cand / "rice_price_model.pkl").exists():
                    self.artifacts_dir = cand
                    break

            if self.artifacts_dir is None:
                self.artifacts_dir = PROJECT_ROOT / "artifacts"

        self.price_model_path = self.artifacts_dir / "rice_price_model.pkl"
        self.price_config_path = self.artifacts_dir / "rice_price_final_config.json"

        if not self.price_model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {self.price_model_path}")
        if not self.price_config_path.exists():
            raise FileNotFoundError(f"Missing config artifact: {self.price_config_path}")

        with open(self.price_config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        if self.config.get("commodity") != "rice":
            raise ValueError(f"Invalid config at {self.price_config_path}: commodity is '{self.config.get('commodity')}', expected 'rice'")

        self.features_list = self.config.get("features", [])
        if len(self.features_list) != 48:
            raise ValueError(f"Invalid feature configuration: expected 48 features, found {len(self.features_list)}")

        loaded_obj = joblib.load(self.price_model_path)
        if isinstance(loaded_obj, dict) and "model" in loaded_obj:
            self.price_model = loaded_obj["model"]
            self.model_metadata = loaded_obj.get("metadata", {})
            self.feature_names_in_artifact = loaded_obj.get("feature_names", [])
        else:
            self.price_model = loaded_obj
            self.model_metadata = {}
            self.feature_names_in_artifact = []

        # Strictly validate feature count and artifact compatibility (refuse legacy models)
        n_features = getattr(self.price_model, "n_features_in_", None)
        if n_features is not None and n_features != 48:
            raise ValueError(
                f"Incompatible model artifact at {self.price_model_path}: "
                f"model expects {n_features} features, but production pipeline requires 48 features. "
                "Refusing to load legacy or incompatible model."
            )

    def _validate_and_clean_input(self, df, market=None, variety=None, prediction_date=None):
        if df is None or df.empty:
            raise ValueError("Empty input data.")

        df = df.copy()

        # Support case-insensitive / API column aliases
        rename_map = {}
        for col in df.columns:
            c_low = str(col).strip().lower()
            if c_low in ("market", "market_name") and "Market Name" not in df.columns:
                rename_map[col] = "Market Name"
            elif c_low == "variety" and "Variety" not in df.columns:
                rename_map[col] = "Variety"
            elif c_low in ("reported_date", "date") and DATE_COLUMN not in df.columns:
                rename_map[col] = DATE_COLUMN
            elif c_low in ("modal_price", "modal price", "price", "modal_price (rs./quintal)") and PRICE_COLUMN not in df.columns:
                rename_map[col] = PRICE_COLUMN
            elif c_low in ("arrivals", "arrival", "arrivals (tonnes)") and ARRIVAL_COLUMN not in df.columns:
                rename_map[col] = ARRIVAL_COLUMN
        if rename_map:
            df = df.rename(columns=rename_map)

        # Infer market, variety, prediction_date if not provided
        if market is None:
            if GROUP_COLUMNS[0] in df.columns and not df[GROUP_COLUMNS[0]].dropna().empty:
                market = str(df[GROUP_COLUMNS[0]].dropna().iloc[-1])
            else:
                raise ValueError(f"Market must be specified or present in '{GROUP_COLUMNS[0]}'.")

        if variety is None:
            if GROUP_COLUMNS[1] in df.columns and not df[GROUP_COLUMNS[1]].dropna().empty:
                variety = str(df[GROUP_COLUMNS[1]].dropna().iloc[-1])
            else:
                raise ValueError(f"Variety must be specified or present in '{GROUP_COLUMNS[1]}'.")

        if prediction_date is None:
            if DATE_COLUMN in df.columns and not df[DATE_COLUMN].dropna().empty:
                prediction_date = pd.to_datetime(df[DATE_COLUMN]).max()
            else:
                raise ValueError(f"Prediction date must be specified or present in '{DATE_COLUMN}'.")

        if GROUP_COLUMNS[0] not in df.columns:
            df[GROUP_COLUMNS[0]] = market
        if GROUP_COLUMNS[1] not in df.columns:
            df[GROUP_COLUMNS[1]] = variety

        required_cols = [DATE_COLUMN, PRICE_COLUMN, ARRIVAL_COLUMN] + GROUP_COLUMNS
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Filter for group and date
        pred_date = pd.to_datetime(prediction_date)
        mask = (
            (df[GROUP_COLUMNS[0]] == market) &
            (df[GROUP_COLUMNS[1]] == variety)
        )
        group_df = df[mask].copy()

        if group_df.empty:
            raise ValueError(f"No data found for Market: {market}, Variety: {variety}")

        group_df[DATE_COLUMN] = pd.to_datetime(group_df[DATE_COLUMN])
        group_df = group_df[group_df[DATE_COLUMN] <= pred_date]

        # Sort and remove duplicates (keep last observed value for a date)
        group_df = group_df.sort_values(DATE_COLUMN).drop_duplicates(subset=[DATE_COLUMN], keep='last')

        # We need at least 30 observations for the 30-day rolling/lag features
        if len(group_df) < 30:
            raise InsufficientHistoryError(f"Insufficient history. Required: 30, Found: {len(group_df)}")

        # Optional: handle missing numerical values (forward fill)
        for col in [PRICE_COLUMN, ARRIVAL_COLUMN]:
            if col in group_df.columns:
                group_df[col] = pd.to_numeric(group_df[col], errors='coerce').ffill().bfill()

        return group_df.reset_index(drop=True)

    def _generate_features(self, df):
        """Generates all 48 required features."""
        # 1. Base pipeline features
        grouped_price = df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]
        df["lag_1"] = grouped_price.shift(1)
        df["lag_7"] = grouped_price.shift(7)
        df["lag_14"] = grouped_price.shift(14)
        df["lag_30"] = grouped_price.shift(30)

        df["rolling_mean_7"] = grouped_price.shift(1).transform(lambda x: x.rolling(7).mean())
        df["rolling_mean_14"] = grouped_price.shift(1).transform(lambda x: x.rolling(14).mean())
        df["rolling_mean_30"] = grouped_price.shift(1).transform(lambda x: x.rolling(30).mean())
        df["rolling_std_7"] = grouped_price.shift(1).transform(lambda x: x.rolling(7).std()).fillna(0)
        df["rolling_std_14"] = grouped_price.shift(1).transform(lambda x: x.rolling(14).std()).fillna(0)

        df["year"] = df[DATE_COLUMN].dt.year
        df["day"] = df[DATE_COLUMN].dt.day
        df["month"] = df[DATE_COLUMN].dt.month
        df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek
        df["week_of_year"] = df[DATE_COLUMN].dt.isocalendar().week.astype(int)

        grouped_arr = df.groupby(GROUP_COLUMNS)[ARRIVAL_COLUMN]
        df["arrival_lag_1"] = grouped_arr.shift(1)
        df["arrival_lag_7"] = grouped_arr.shift(7)
        df["arrival_rolling_mean_7"] = grouped_arr.shift(1).transform(lambda x: x.rolling(7).mean())
        df["arrival_rolling_mean_14"] = grouped_arr.shift(1).transform(lambda x: x.rolling(14).mean())

        df["price_pct_change"] = grouped_price.pct_change(fill_method=None) * 100.0
        df["arrival_pct_change"] = grouped_arr.pct_change(fill_method=None) * 100.0

        # 2. Step 7 Advanced Features
        grouped = df.groupby(GROUP_COLUMNS)[PRICE_COLUMN]

        df["price_change_1"] = grouped.diff(1)
        df["price_change_7"] = grouped.diff(7)
        df["price_change_14"] = grouped.diff(14)
        df["accel_1_7"] = df["price_change_1"] - df["price_change_7"]
        df["accel_1_14"] = df["price_change_1"] - df["price_change_14"]

        def get_slope_weights(window):
            x = np.arange(window) - (window - 1) / 2.0
            return x / np.sum(x**2)

        w7 = get_slope_weights(7)
        df["trend_slope_7"] = sum(w7[i] * grouped.shift(6 - i) for i in range(7))
        w14 = get_slope_weights(14)
        df["trend_slope_14"] = sum(w14[i] * grouped.shift(13 - i) for i in range(14))
        w30 = get_slope_weights(30)
        df["trend_slope_30"] = sum(w30[i] * grouped.shift(29 - i) for i in range(30))

        df["ewma_7"] = grouped.transform(lambda x: x.ewm(span=7, adjust=False).mean())
        df["ewma_14"] = grouped.transform(lambda x: x.ewm(span=14, adjust=False).mean())
        df["ewma_30"] = grouped.transform(lambda x: x.ewm(span=30, adjust=False).mean())
        df["rolling_median_7"] = grouped.transform(lambda x: x.rolling(7).median())
        df["rolling_median_14"] = grouped.transform(lambda x: x.rolling(14).median())
        df["rolling_median_30"] = grouped.transform(lambda x: x.rolling(30).median())

        df["gap_mean_7"] = df[PRICE_COLUMN] - df["rolling_mean_7"]
        df["gap_mean_14"] = df[PRICE_COLUMN] - df["rolling_mean_14"]
        df["zscore_7"] = (df[PRICE_COLUMN] - df["rolling_mean_7"]) / (df["rolling_std_7"] + 1e-5)
        df["zscore_14"] = (df[PRICE_COLUMN] - df["rolling_mean_14"]) / (df["rolling_std_14"] + 1e-5)

        def compute_streak(df, condition_series):
            group_keys = [df[c] for c in GROUP_COLUMNS]
            block_id = (~condition_series).groupby(group_keys).cumsum()
            streak = condition_series.groupby(group_keys + [block_id]).cumsum()
            return streak

        is_move = (df["price_change_1"] != 0).astype(int)
        is_pos = (df["price_change_1"] > 0).astype(int)
        is_neg = (df["price_change_1"] < 0).astype(int)

        group_keys = [df[c] for c in GROUP_COLUMNS]
        df["move_count_7"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(7).sum())
        df["move_count_14"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(14).sum())
        df["move_count_30"] = is_move.groupby(group_keys).transform(lambda x: x.rolling(30).sum())
        df["pos_count_7"] = is_pos.groupby(group_keys).transform(lambda x: x.rolling(7).sum())
        df["neg_count_7"] = is_neg.groupby(group_keys).transform(lambda x: x.rolling(7).sum())

        df["streak_zero"] = compute_streak(df, is_move == 0)
        df["streak_pos"] = compute_streak(df, is_pos == 1)
        df["streak_neg"] = compute_streak(df, is_neg == 1)

        last_move = df["price_change_1"].replace(0, np.nan)
        df["last_non_zero_move"] = last_move.groupby(group_keys).ffill().fillna(0)
        df["last_move_abs"] = df["last_non_zero_move"].abs()

        return df

    def predict(self, df, market=None, variety=None, prediction_date=None, **kwargs):
        """
        Executes end-to-end inference for Price and Volatility.
        Returns a dict.
        """
        if market is None:
            market = kwargs.get("Market Name", kwargs.get("market"))
        if variety is None:
            variety = kwargs.get("Variety", kwargs.get("variety"))
        if prediction_date is None:
            prediction_date = kwargs.get("prediction_date")

        # Support direct inference if all 48 features are already pre-computed
        has_all_features = all(f in df.columns for f in self.features_list)
        if has_all_features and (PRICE_COLUMN in df.columns or "modal_price" in df.columns) and (DATE_COLUMN not in df.columns or len(df[df[DATE_COLUMN].notna()]) < 30):
            price_col = PRICE_COLUMN if PRICE_COLUMN in df.columns else "modal_price"
            X = df.iloc[[-1]][self.features_list].copy()
            X = X.apply(pd.to_numeric, errors="coerce")
            if X.isna().any().any():
                nan_cols = X.columns[X.isna().any()].tolist()
                raise ValueError(f"NaN detected in prediction features: {nan_cols}")

            predicted_change = float(self.price_model.predict(X)[0])
            current_price = float(df.iloc[-1][price_col])
            predicted_price = current_price + predicted_change

            if len(df) >= 5:
                last_5 = df.iloc[-5:][price_col]
                volatility_range = float(last_5.max() - last_5.min())
            else:
                volatility_range = float(df.iloc[-1].get("predicted_volatility_range", df.iloc[-1].get("volatility_range", 0.0)))

            if volatility_range <= 0.5:
                volatility_class = "LOW"
            elif volatility_range <= 50.0:
                volatility_class = "MEDIUM"
            else:
                volatility_class = "HIGH"

            mkt = market or str(df.iloc[-1].get(GROUP_COLUMNS[0], df.iloc[-1].get("market", "Unknown")))
            var = variety or str(df.iloc[-1].get(GROUP_COLUMNS[1], df.iloc[-1].get("variety", "Unknown")))
            pdate = str(prediction_date) if prediction_date else (str(df.iloc[-1].get(DATE_COLUMN, "today")) if DATE_COLUMN in df.columns else "today")

            return {
                "market": mkt,
                "variety": var,
                "prediction_date": pdate,
                "predicted_modal_price": round(predicted_price, 2),
                "predicted_volatility_range": round(volatility_range, 2),
                "predicted_volatility_class": volatility_class
            }

        # 1. Clean and filter input
        hist_df = self._validate_and_clean_input(df, market, variety, prediction_date)
        resolved_market = market if market is not None else str(hist_df[GROUP_COLUMNS[0]].iloc[-1])
        resolved_variety = variety if variety is not None else str(hist_df[GROUP_COLUMNS[1]].iloc[-1])
        resolved_pred_date = str(pd.to_datetime(prediction_date).date()) if prediction_date is not None else str(hist_df[DATE_COLUMN].max().date())

        # 2. Predict Price
        feat_df = self._generate_features(hist_df)

        # Ensure all required features are present and ordered
        missing_features = [f for f in self.features_list if f not in feat_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features for model: {missing_features}")

        # Use iloc[[-1]] to return a 1-row DataFrame and preserve dtypes (avoiding mixed-Series object upcast)
        X = feat_df.iloc[[-1]][self.features_list].copy()

        # Explicitly coerce to numeric to ensure XGBoost compatibility
        X = X.apply(pd.to_numeric, errors="coerce")

        # Validate no NaNs
        if X.isna().any().any():
            nan_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"NaN detected in prediction features: {nan_cols}")

        predicted_change = float(self.price_model.predict(X)[0])
        current_price = float(feat_df.iloc[-1][PRICE_COLUMN])
        predicted_price = current_price + predicted_change

        # 3. Predict Volatility (Persistence Baseline)
        # Use the last 5 observations
        last_5 = hist_df.iloc[-5:][PRICE_COLUMN]
        volatility_range = float(last_5.max() - last_5.min())

        if volatility_range <= 0.5:
            volatility_class = "LOW"
        elif volatility_range <= 50.0:
            volatility_class = "MEDIUM"
        else:
            volatility_class = "HIGH"

        # 4. Final Output
        return {
            "market": resolved_market,
            "variety": resolved_variety,
            "prediction_date": resolved_pred_date,
            "predicted_modal_price": round(predicted_price, 2),
            "predicted_volatility_range": round(volatility_range, 2),
            "predicted_volatility_class": volatility_class
        }
