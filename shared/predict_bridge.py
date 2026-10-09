"""
AgroVision — Shared Prediction Bridge

Single import point for Member 3 (FastAPI / Engineering team).

This module wraps both ML predict APIs into one clean interface.
Member 3 only needs to import THIS FILE — no need to know the
internal structure of member1-yield-model or member2-price-volatility.

Usage:
    from shared.predict_bridge import AgroVisionPredictor

    predictor = AgroVisionPredictor()

    # Full prediction (yield + price + volatility)
    result = predictor.predict_all(data_df, commodity="tea")
    # {
    #   "yield_arrivals_tonnes": 52.3,
    #   "predicted_price_rs_quintal": 248.50,
    #   "volatility": "LOW",
    #   "commodity": "tea"
    # }

    # Yield only (Member 1)
    yield_result = predictor.predict_yield(input_dict)
    # {"yield_arrivals_tonnes": 52.3}

    # Price + volatility only (Member 2)
    price_result = predictor.predict_price(data_df, commodity="tea")
    # {"predicted_price_rs_quintal": 248.50, "volatility": "LOW", "commodity": "tea"}
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AgroVisionPredictor:
    """
    Unified predictor for Member 3 (FastAPI) integration.

    Wraps:
      - Member 1: yield_model.pkl  → arrival forecast (tonnes)
      - Member 2: {commodity}_price_model.pkl + {commodity}_volatility_model.pkl
                                   → price forecast (₹/Quintal) + volatility (LOW/MEDIUM/HIGH)

    Parameters
    ----------
    m1_model_path : str or Path, optional
        Path to Member 1's yield_model.pkl. Defaults to standard location.
    m2_artifacts_dir : str or Path, optional
        Directory containing Member 2's .pkl files. Defaults to standard location.
    """

    def __init__(
        self,
        m1_model_path: Optional[Path] = None,
        m2_artifacts_dir: Optional[Path] = None,
    ):
        self._m1_root = PROJECT_ROOT / "member1-yield-model"
        self._m2_src = PROJECT_ROOT / "member2-price-volatility" / "src"

        self._m1_model_path = m1_model_path or (
            self._m1_root / "models" / "yield_model.pkl"
        )
        self._m2_artifacts_dir = m2_artifacts_dir or (
            PROJECT_ROOT / "member2-price-volatility" / "artifacts"
        )

        # Lazy-loaded internal modules
        self._m1_predict = None
        self._m2_predict = None
        self._rice_pipeline = None

    # ----------------------------------------------------------
    # Private loaders
    # ----------------------------------------------------------

    def _load_m1(self):
        """Lazy-load Member 1's predict module."""
        if self._m1_predict is not None:
            return

        if str(self._m1_root) not in sys.path:
            sys.path.insert(0, str(self._m1_root))

        try:
            from src.predict import load_model, predict_yield  # noqa: F401
            self._m1_load_model = load_model
            self._m1_predict_yield = predict_yield
            self._m1_model_info = load_model(str(self._m1_model_path))
            self._m1_predict = True  # marker: loaded
        except (ImportError, ModuleNotFoundError, FileNotFoundError):
            self._m1_load_model = None
            self._m1_predict_yield = None
            self._m1_model_info = {"feature_names": []}
            self._m1_predict = False

    def _load_m2(self):
        """Lazy-load Member 2's predict module (for Tea and other commodities)."""
        if self._m2_predict is not None:
            return

        if str(self._m2_src) not in sys.path:
            sys.path.insert(0, str(self._m2_src))

        from models.predict import predict as m2_predict_fn  # noqa: F401

        self._m2_predict_fn = m2_predict_fn
        self._m2_predict = True  # marker: loaded

    def _load_rice(self):
        """Lazy-load Member 2's production RiceInferencePipeline."""
        if self._rice_pipeline is not None:
            return

        if str(self._m2_src) not in sys.path:
            sys.path.insert(0, str(self._m2_src))

        from inference import RiceInferencePipeline

        if self._m2_artifacts_dir and (self._m2_artifacts_dir / "rice_price_final_config.json").exists():
            self._rice_pipeline = RiceInferencePipeline(artifacts_dir=self._m2_artifacts_dir)
        else:
            self._rice_pipeline = RiceInferencePipeline()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def predict_yield(self, input_dict: dict) -> dict:
        """
        Predict tea arrivals (yield) using Member 1's model.

        Parameters
        ----------
        input_dict : dict
            Feature values expected by the yield model.
            Required keys match `model_info["feature_names"]`.
            Unknown keys are ignored; missing keys default to 0.

        Returns
        -------
        dict
            {"yield_arrivals_tonnes": float}

        Example
        -------
        result = predictor.predict_yield({
            "Modal Price (Rs./Quintal)": 250,
            "Price_Lag_1": 240,
            "Price_RollMean_7": 245,
            ...
        })
        print(result["yield_arrivals_tonnes"])   # e.g. 52.3
        """
        self._load_m1()
        if not self._m1_predict or self._m1_predict_yield is None:
            raise RuntimeError("Member 1 yield model or predict module is not available.")
        value = self._m1_predict_yield(self._m1_model_info, input_dict)
        return {"yield_arrivals_tonnes": round(float(value), 4)}

    def predict_price(self, data_df, commodity: str = "tea", **kwargs) -> dict:
        """
        Predict price and volatility using Member 2's model.

        For commodity="rice", uses the verified production 48-feature XGBoost
        model and persistence volatility estimator (RiceInferencePipeline).
        For commodity="tea", routes to the multi-commodity model.

        Parameters
        ----------
        data_df : pandas.DataFrame
            For rice: historical time-series DataFrame (30+ observations) or
            48 pre-computed features DataFrame.
            For tea: feature DataFrame matching tea feature config.
        commodity : str
            Commodity name ("rice", "tea", etc.). Defaults to "tea".
        **kwargs :
            Optional metadata (market, variety, prediction_date).

        Returns
        -------
        dict
            {
                "predicted_price_rs_quintal": float,
                "volatility": "LOW" | "MEDIUM" | "HIGH",
                "commodity": str
            }
        """
        comm = str(commodity).strip().lower()
        if comm == "rice":
            self._load_rice()
            res = self._rice_pipeline.predict(data_df, **kwargs)
            out = {
                "predicted_price_rs_quintal": float(res["predicted_modal_price"]),
                "volatility": str(res["predicted_volatility_class"]),
                "commodity": "rice",
                "predicted_volatility_range": float(res.get("predicted_volatility_range", 0.0)),
            }
            for k in ("market", "variety", "prediction_date"):
                if k in res:
                    out[k] = res[k]
            return out
        else:
            self._load_m2()
            raw = self._m2_predict_fn(data_df, commodity=comm)
            return {
                "predicted_price_rs_quintal": raw["predicted_price"],
                "volatility": raw["volatility"],
                "commodity": raw["commodity"],
            }

    def predict_all(self, data_df, commodity: str = "tea", yield_input: Optional[dict] = None) -> dict:
        """
        Run both pipelines and return a combined prediction.

        Parameters
        ----------
        data_df : pandas.DataFrame
            Feature DataFrame for Member 2 (price + volatility model).
        commodity : str
            Commodity name (for Member 2).
        yield_input : dict, optional
            Feature dict for Member 1 (yield model).
            If None, attempts to extract from data_df's first row.

        Returns
        -------
        dict
            {
                "yield_arrivals_tonnes"    : float or None,
                "predicted_price_rs_quintal": float,
                "volatility"               : "LOW" | "MEDIUM" | "HIGH",
                "commodity"                : str
            }
        """
        results = {}

        # --- Member 2: Price + Volatility ---
        try:
            price_result = self.predict_price(data_df, commodity=commodity)
            results.update(price_result)
        except Exception as e:
            results["predicted_price_rs_quintal"] = None
            results["volatility"] = None
            results["commodity"] = commodity
            results["price_error"] = str(e)

        # --- Member 1: Yield ---
        try:
            if yield_input is None and data_df is not None and len(data_df) > 0:
                yield_input = data_df.iloc[0].to_dict()

            if yield_input:
                yield_result = self.predict_yield(yield_input)
                results.update(yield_result)
            else:
                results["yield_arrivals_tonnes"] = None
        except Exception as e:
            results["yield_arrivals_tonnes"] = None
            results["yield_error"] = str(e)

        return results

    # ----------------------------------------------------------
    # Convenience: feature listing
    # ----------------------------------------------------------

    def get_required_features(self) -> dict:
        """
        Return the list of features required by each model.
        Useful for building the FastAPI request schema.

        Returns
        -------
        dict with keys 'yield_features', 'price_features', 'volatility_features'
        """
        self._load_m1()
        self._load_m2()
        self._load_rice()

        # Import Member 2 config for feature lists
        if str(self._m2_src) not in sys.path:
            sys.path.insert(0, str(self._m2_src))
        from config import PRICE_FEATURES, VOLATILITY_FEATURES, DATE_COLUMN, PRICE_COLUMN, ARRIVAL_COLUMN, GROUP_COLUMNS  # noqa: F401

        return {
            "yield_features": self._m1_model_info.get("feature_names", []),
            "price_features": PRICE_FEATURES,
            "volatility_features": VOLATILITY_FEATURES,
            "rice_price_features": self._rice_pipeline.features_list,
            "rice_history_columns": [DATE_COLUMN, PRICE_COLUMN, ARRIVAL_COLUMN] + GROUP_COLUMNS,
        }
